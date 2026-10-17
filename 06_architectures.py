# 06_architectures.py
"""
Step 06: dependence of the results on the architecture, and computational cost.

Models (all with the same training recipe: Adam 1e-3, 15 epochs, batch 128, class weights,
label smoothing 0.1):
  deep:      cnn_transformer (reference), pure_cnn, pure_transformer, eegnet, shallow_convnet
  classical: band power (50 features) + logistic regression / random forest

Three estimates per model, so that a model that fails everywhere can be recognised:
  1. WITHIN-dataset: 5-fold participant-wise, 1 seed (all four two-class datasets)
  2. POOLED: 3 participant splits (42, 0, 1), 1 seed each
  3. LODO: 5 held-out datasets x 2 seeds
The cnn_transformer values are taken from steps 03-05 (not retrained here).

Complexity: parameters, FLOPs per window (torch FlopCounter), CPU inference time per window.

GPU (Colab T4): ~60-90 min. Resumable.
Output: outputs/06_arch/RESULTS.txt, all_runs.csv, summary.csv, summary_table.csv, complexity.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out, OUT_DIR
from rev.common import load_v2, train_model, predict, bandpower, cached, DEVICE
from rev.models import MODELS, n_params

OUT = out("06_arch"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
DEEP = ["pure_cnn", "pure_transformer", "eegnet", "shallow_convnet"]
CLASSICAL = {
    "bandpower_LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "bandpower_RF": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                   n_jobs=-1, random_state=0),
}
BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]
POOL_SPLITS = [42, 0, 1]
LODO_SEEDS = [0, 1]

buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

# ------------------------------------------------ complexity first (fails fast if a model is broken)
log("=== Complexity (per 4-s window, 10 x 512) ===")
crow = []
for name in ["cnn_transformer"] + DEEP:
    m = MODELS[name]().eval()
    x = torch.zeros(1, 10, 512)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            m(x)
        flops = fc.get_total_flops()
    except Exception:
        flops = np.nan
    xb = torch.zeros(256, 10, 512)
    with torch.no_grad():
        m(xb); t = time.time()
        for _ in range(3): m(xb)
    ms = (time.time() - t) / (3 * 256) * 1000
    crow.append(dict(model=name, params=n_params(m), MFLOPs=flops / 1e6, cpu_ms_per_window=ms))
    log(f"  {name:<17} params={n_params(m):>8,}  MFLOPs={flops/1e6:8.2f}  CPU {ms:.3f} ms/window")
pd.DataFrame(crow).to_csv(OUT / "complexity.csv", index=False)

X, meta = load_v2(primary_only=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
log(f"\ncorpus {X.shape} | device {DEVICE}")
log("computing band-power features ...")
BP = bandpower(X)
t0 = time.time()
rows = []


def fit_classical(name, tr, te, seed=0):
    clf = CLASSICAL[name](); clf.fit(BP[tr], y[tr]); return clf.predict_proba(BP[te])[:, 1]


def fit_deep(name, tr, te, seed, tag):
    r = cached(PRED / f"{tag}.npz", lambda: {"p": predict(train_model(X[tr], y[tr], seed, arch=name), X[te])})
    return r["p"]


ALL = DEEP + list(CLASSICAL)
for name in ALL:
    deep = name in DEEP
    log(f"\n##### {name} #####")
    # 1) within-dataset
    for d in BOTH:
        idx = np.where(ds == d)[0]
        p = np.full(len(idx), np.nan)
        for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                   .split(idx, y[idx], groups=pid[idx])):
            tr, te = idx[a], idx[b]
            p[b] = fit_deep(name, tr, te, 0, f"{name}_within_{d}_f{k}") if deep else fit_classical(name, tr, te)
        auc = roc_auc_score(y[idx], p)
        rows.append(dict(model=name, protocol="within", dataset=d, seed=0, auc=auc))
        log(f"  within  {d:<17} AUC={auc:.3f}   ({(time.time()-t0)/60:.0f} min)")
    # 2) pooled
    for sp in POOL_SPLITS:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=sp).split(np.zeros(len(y)), y, groups=pid))
        p = fit_deep(name, tr, te, 0, f"{name}_pooled_split{sp}") if deep else fit_classical(name, tr, te)
        nb = ds[te] != "BIDS"
        rows.append(dict(model=name, protocol="pooled", dataset="ALL", seed=sp, auc=roc_auc_score(y[te], p)))
        rows.append(dict(model=name, protocol="pooled", dataset="noBIDS", seed=sp,
                         auc=roc_auc_score(y[te][nb], p[nb])))
        log(f"  pooled  split {sp:<2} AUC={rows[-2]['auc']:.3f}  noBIDS={rows[-1]['auc']:.3f}")
    # 3) LODO
    for d in sorted(np.unique(ds)):
        te = np.where(ds == d)[0]; tr = np.where(ds != d)[0]
        for seed in (LODO_SEEDS if deep else [0]):
            p = fit_deep(name, tr, te, seed, f"{name}_lodo_{d}_seed{seed}") if deep else fit_classical(name, tr, te)
            if d in BOTH:
                rows.append(dict(model=name, protocol="lodo", dataset=d, seed=seed, auc=roc_auc_score(y[te], p),
                                 pred_std=float(p.std())))
                log(f"  LODO    {d:<17} seed {seed} AUC={rows[-1]['auc']:.3f}")
            else:
                rows.append(dict(model=name, protocol="lodo_specificity", dataset=d, seed=seed,
                                 auc=float((p < 0.5).mean())))
    pd.DataFrame(rows).to_csv(OUT / "all_runs.csv", index=False)

# ------------------------------------------------ add cnn_transformer from Steps 3-5
try:
    w = pd.read_csv(OUT_DIR / "03_within" / "within_dataset_auc.csv").query("model=='cnn_transformer'")
    for _, r in w.iterrows():
        rows.append(dict(model="cnn_transformer", protocol="within", dataset=r.dataset, seed=r.seed, auc=r.auc))
    pr = pd.read_csv(OUT_DIR / "04_pooled" / "runs.csv")
    for _, r in pr.iterrows():
        rows.append(dict(model="cnn_transformer", protocol="pooled", dataset="ALL", seed=r.split, auc=r.auc_pooled))
        rows.append(dict(model="cnn_transformer", protocol="pooled", dataset="noBIDS", seed=r.split, auc=r.auc_pooled_noBIDS))
    lo = pd.read_csv(OUT_DIR / "05_lodo" / "per_seed.csv").dropna(subset=["auc"])
    for _, r in lo.iterrows():
        rows.append(dict(model="cnn_transformer", protocol="lodo", dataset=r.held_out, seed=r.seed, auc=r.auc))
except Exception as e:
    log("  (cnn_transformer reference load error:", e, ")")

df = pd.DataFrame(rows); df.to_csv(OUT / "all_runs.csv", index=False)
d2 = df[df.protocol.isin(["within", "pooled", "lodo"])]
summ = d2.groupby(["model", "protocol", "dataset"]).auc.mean().unstack(["protocol", "dataset"])
summ.to_csv(OUT / "summary.csv")

log("\n=== SUMMARY: mean AUC ===")
tab = []
for mname, g in d2.groupby("model"):
    wv = g[g.protocol == "within"].groupby("dataset").auc.mean()
    lv = g[g.protocol == "lodo"].groupby("dataset").auc.mean()
    tab.append(dict(model=mname,
                    within_SAM40=wv.get("SAM40"), within_MA=wv.get("MentalArithmetic"),
                    within_DASPS=wv.get("DASPS"), within_Mend=wv.get("Mendeley"),
                    pooled=g[(g.protocol == "pooled") & (g.dataset == "ALL")].auc.mean(),
                    pooled_noBIDS=g[(g.protocol == "pooled") & (g.dataset == "noBIDS")].auc.mean(),
                    lodo_SAM40=lv.get("SAM40"), lodo_MA=lv.get("MentalArithmetic"),
                    lodo_DASPS=lv.get("DASPS"), lodo_Mend=lv.get("Mendeley"),
                    lodo_mean4=lv.mean()))
tab = pd.DataFrame(tab); tab.to_csv(OUT / "summary_table.csv", index=False)
log(tab.round(3).to_string(index=False))
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
