# 08_adversarial_v2.py
"""
Step 08: dataset-adversarial training (DANN).

  * Standard DANN: L = L_anx + L_dataset; the gradient-reversal layer (GRL) scales the
    gradient by -lambda (lambda appears only in the GRL)
  * lambda warm-up at every step (Ganin et al. 2016): lambda(p) = lmax * (2/(1+exp(-10p)) - 1)
  * lambda_max in {0, 0.1, 0.3, 1, 3}  (0 = same network without the adversary, the baseline)
  * In each LODO fold, 20 % of the SOURCE participants are held out for validation:
      - source-validation anxiety AUC for each lambda
      - the same identity probe for every model: logistic regression on source-validation
        features, participant-wise 5-fold, identical split for lambda = 0 and the
        adversarial models
      - the majority baseline of that fold
  * Collapse check: SD of the target predictions and of the features
  * Per-epoch losses are logged (training dynamics)

4 two-class held-out datasets x 5 lambdas x 2 seeds = 40 trainings. GPU: ~40-60 min. Resumable.
Output: outputs/08_adversarial/RESULTS.txt, runs.csv, summary.csv, train_logs.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, set_seed, bandpower, cluster_boot_auc, DEVICE, EPOCHS, BATCH, LR, WD
from rev.models import CNNTransformer

OUT = out("08_adversarial"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
HOLD = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0]
SEEDS = [0, 1]


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DANN(nn.Module):
    def __init__(self, n_domains):
        super().__init__()
        self.backbone = CNNTransformer()                      # same backbone; its .head = anxiety head
        self.dom = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, n_domains))

    def forward(self, x, lam=0.0, return_feat=False):
        f = self.backbone.features(x)
        a = self.backbone.head(f)
        d = self.dom(GRL.apply(f, lam))
        return (a, d, f) if return_feat else (a, d)


def train_dann(X, y, d, n_dom, lmax, seed, tag, logs):
    set_seed(seed)
    m = DANN(n_dom).to(DEVICE)
    cw = torch.tensor([1.0 / max((y == 0).sum(), 1), 1.0 / max((y == 1).sum(), 1)], dtype=torch.float32)
    ca = nn.CrossEntropyLoss(weight=(cw / cw.sum() * 2).to(DEVICE), label_smoothing=0.1)
    cd = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y), torch.tensor(d)), batch_size=BATCH, shuffle=True)
    total = EPOCHS * len(dl); step = 0
    for ep in range(EPOCHS):
        m.train(); la = ld = acc = n = 0.0
        for xb, yb, db in dl:
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            lam = lmax * (2.0 / (1.0 + np.exp(-10.0 * step / total)) - 1.0)
            a, dd = m(xb, lam)
            loss_a, loss_d = ca(a, yb), cd(dd, db)
            opt.zero_grad(); (loss_a + loss_d).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
            k = len(yb); la += loss_a.item() * k; ld += loss_d.item() * k
            acc += (dd.argmax(1) == db).float().sum().item(); n += k
        logs.append(dict(tag=tag, epoch=ep + 1, lam_end=lam, loss_anx=la / n, loss_dom=ld / n, dom_acc_train=acc / n))
    return m


@torch.no_grad()
def infer(m, X):
    m.eval(); ps, fs = [], []
    for i in range(0, len(X), 256):
        a, _, f = m(torch.tensor(X[i:i + 256]).to(DEVICE), 0.0, return_feat=True)
        ps.append(torch.softmax(a, 1)[:, 1].cpu().numpy()); fs.append(f.cpu().numpy())
    return np.concatenate(ps), np.concatenate(fs)


def probe(F, dlab, groups):
    """Same identity probe for every model: participant-wise 5-fold LR on features."""
    accs = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, dlab, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[a], dlab[a])
        accs.append((clf.predict(F[b]) == dlab[b]).mean())
    return float(np.mean(accs))


buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
X = np.asarray(X, dtype=np.float32)
y, ds, pid = meta.y.values.astype(np.int64), meta.dataset.values, meta.participant.values
log(f"corpus {X.shape} | device {DEVICE}")
t0 = time.time(); rows = []; logs = []
logf = OUT / "train_logs.csv"
if logf.exists():
    logs = pd.read_csv(logf).to_dict("records")

for hold in HOLD:
    te = np.where(ds == hold)[0]; src = np.where(ds != hold)[0]
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(src, y[src], groups=pid[src]))
    tr, va = src[a], src[b]                                    # source train / source-val participants
    doms = sorted(np.unique(ds[src])); dmap = {k: i for i, k in enumerate(doms)}
    dtr = np.array([dmap[k] for k in ds[tr]]); dva = np.array([dmap[k] for k in ds[va]])
    majority = np.bincount(dva).max() / len(dva)
    bp_probe = probe(bandpower(X[va]), dva, pid[va])          # input band-power, SAME windows & split
    log(f"\n=== held-out {hold}: source={doms}  train={len(tr)} src-val={len(va)} target={len(te)}")
    log(f"    identity probe on src-val: majority={majority:.3f}  band-power input={bp_probe:.3f}")
    for lmax in LAMBDAS:
        for seed in SEEDS:
            tag = f"{hold}_lam{lmax}_seed{seed}"
            fp = PRED / f"{tag}.npz"
            if fp.exists():
                r = dict(np.load(fp))
            else:
                m = train_dann(X[tr], y[tr], dtr, len(doms), lmax, seed, tag, logs)
                pt, _ = infer(m, X[te]); pv, fv = infer(m, X[va])
                r = dict(pt=pt, pv=pv, fv=fv)
                np.savez_compressed(fp, **r); pd.DataFrame(logs).to_csv(logf, index=False)
            pt, pv, fv = r["pt"], r["pv"], r["fv"]
            row = dict(held_out=hold, lambda_max=lmax, seed=seed,
                       target_auc=roc_auc_score(y[te], pt), target_pred_std=float(pt.std()),
                       srcval_auc=roc_auc_score(y[va], pv), feat_std=float(fv.std(axis=0).mean()),
                       id_probe=probe(fv, dva, pid[va]), id_majority=majority, id_probe_bandpower=bp_probe)
            row["collapsed"] = row["target_pred_std"] < 0.01
            rows.append(row)
            log(f"  lam={lmax:<4} seed {seed}: target AUC={row['target_auc']:.3f}  src-val AUC={row['srcval_auc']:.3f}"
                f"  id-probe={row['id_probe']:.3f} (maj {majority:.3f})  pred_std={row['target_pred_std']:.3f}"
                f"{'  COLLAPSED' if row['collapsed'] else ''}   ({(time.time()-t0)/60:.0f} min)")
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)

df = pd.DataFrame(rows)
summ = df.groupby("lambda_max").agg(target_auc=("target_auc", "mean"), srcval_auc=("srcval_auc", "mean"),
                                    id_probe=("id_probe", "mean"), id_majority=("id_majority", "mean"),
                                    n_collapsed=("collapsed", "sum"))
summ.to_csv(OUT / "summary.csv")
log("\n=== SUMMARY by lambda_max (mean over 4 held-out x 2 seeds) ===")
log(summ.round(3).to_string())
log("\n=== Per held-out x lambda (mean over seeds) ===")
log(df.groupby(["held_out", "lambda_max"])[["target_auc", "srcval_auc", "id_probe"]].mean().round(3).to_string())

# CI for target AUC: lambda=0 vs best-srcval lambda>0, seed-averaged
log("\n=== Target AUC with participant-cluster CI (seed-averaged) ===")
for hold in HOLD:
    te = ds == hold
    for lmax in LAMBDAS:
        p = np.mean([np.load(PRED / f"{hold}_lam{lmax}_seed{s}.npz")["pt"] for s in SEEDS], 0)
        a_, lo, hi = roc_auc_score(y[te], p), *cluster_boot_auc(y[te], p, pid[te], n_boot=1000)[:2]
        log(f"  {hold:<17} lam={lmax:<4} AUC={a_:.3f} CI[{lo:.3f},{hi:.3f}]")
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
