# rev/common.py
"""
Shared helpers: corpus loading, model training, prediction, participant-cluster
bootstrap, band-power features and resumable caching of predictions.
"""
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.signal import welch

from rev.paths import OUT_DIR
from rev.models import MODELS

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- training recipe (identical for all experiments) ----
EPOCHS, BATCH, LR, WD = 15, 128, 1e-3, 1e-5


def load_v2(primary_only=True, raw=False):
    """Return X (memory-mapped array) and the metadata DataFrame.
    primary_only=True keeps the 14,655 windows of the primary corpus."""
    # REV_CORPUS_DIR can point to a copy of the corpus on a faster disk (e.g. /content on Colab)
    d = Path(os.environ.get("REV_CORPUS_DIR", OUT_DIR / "02_corpus_v2"))
    meta = pd.read_csv(d / "meta.csv", low_memory=False)
    X = np.load(d / ("X_raw.npy" if raw else "X_z.npy"), mmap_mode="r")
    if primary_only:
        idx = np.where(meta.in_primary.values)[0]
        meta = meta.iloc[idx].reset_index(drop=True)
        X = X[idx]                       # loads the selected windows into memory
    return X, meta


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(Xtr, ytr, seed, arch="cnn_transformer", epochs=EPOCHS, verbose=False):
    set_seed(seed)
    model = MODELS[arch]().to(DEVICE)
    cw = torch.tensor([1.0 / max((ytr == 0).sum(), 1), 1.0 / max((ytr == 1).sum(), 1)],
                      dtype=torch.float32)
    cw = (cw / cw.sum() * 2).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    dl = DataLoader(TensorDataset(torch.tensor(np.asarray(Xtr, dtype=np.float32)),
                                  torch.tensor(np.asarray(ytr, dtype=np.int64))),
                    batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if verbose:
            print(f"      epoch {ep+1:2d}/{epochs} loss={tot/n:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, X, want_feat=False):
    model.eval()
    ps, fs = [], []
    for i in range(0, len(X), 256):
        xb = torch.tensor(np.asarray(X[i:i + 256], dtype=np.float32)).to(DEVICE)
        out, f = model(xb, return_feat=True)
        ps.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        if want_feat:
            fs.append(f.cpu().numpy())
    p = np.concatenate(ps)
    return (p, np.concatenate(fs)) if want_feat else p


# ---- log band-power features (used by the dataset-identity probe) ----
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bandpower(X, fs=128):
    out = []
    for i in range(0, len(X), 2000):
        f, P = welch(np.asarray(X[i:i + 2000]), fs=fs, nperseg=256, axis=2)
        out.append(np.log(np.concatenate([P[:, :, (f >= lo) & (f < hi)].mean(axis=2)
                                          for lo, hi in BANDS], axis=1) + 1e-12))
    return np.concatenate(out)


# ---- stats ----
def cluster_boot_auc(y, p, groups, n_boot=2000, seed=0):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        s = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in s])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), vals


def auc_row(y, p, groups, **extra):
    auc = roc_auc_score(y, p)
    lo, hi, _ = cluster_boot_auc(y, p, groups)
    return dict(**extra, auc=round(auc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                n_windows=len(y), n_participants=len(np.unique(groups)))


# ---- resumable prediction cache (an interrupted run continues where it stopped) ----
def cached(path, fn):
    """Load `path` if it exists; otherwise run fn() (returns a dict of arrays) and save it."""
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    t = time.time()
    res = fn()
    np.savez_compressed(path, **res)
    print(f"      saved {path.name} ({time.time()-t:.0f}s)", flush=True)
    return res
