# 11_balanced_prior.py
"""
Step 11: test of the dataset-prior mechanism. The class balance within each two-class dataset
and the exclusion of BIDS are varied separately, with a sample-size-matched control for the
subsampling.

Conditions (pooled participant-wise, splits 42/0/1/2/3, seed 0; CNN-Transformer):
  A. original                     (all 5 datasets, original class balance)
  B. noBIDS                       (BIDS removed, balance unchanged)
  C. noBIDS + balanced            (each two-class dataset 50/50; Mendeley balanced at
                                   participant level)
  D. noBIDS + size-matched random (as many windows as C, sampled at random; balance unchanged)
For each condition: pooled AUC, prior-only AUC on the same split, per-dataset AUCs.
  - In C the prior-only AUC should be ~0.5 (the prior carries no information). If the model
    is still above 0.5, the prior is not the only source.
  - C vs D separates the effect of having less data.

GPU: ~10-15 min. Output: outputs/11_balanced/RESULTS.txt, runs.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, train_model, predict, cached, DEVICE

OUT = out("11_balanced"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
SPLITS = [42, 0, 1, 2, 3]
BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
rng = np.random.default_rng(0)
log(f"device {DEVICE}")

idx_A = np.arange(len(y))
idx_B = np.where(ds != "BIDS")[0]
keep = []
for d in BOTH:
    ix = np.where(ds == d)[0]
    if d == "Mendeley":                                       # participant-level labels -> balance participants
        pl = pd.Series(y[ix], index=pid[ix]).groupby(level=0).first()
        n = min((pl == 0).sum(), (pl == 1).sum())
        sel = np.concatenate([rng.choice(pl[pl == c].index.values, n, replace=False) for c in (0, 1)])
        keep.append(ix[np.isin(pid[ix], sel)])
    else:
        n = min((y[ix] == 0).sum(), (y[ix] == 1).sum())
        keep.append(np.concatenate([rng.choice(ix[y[ix] == c], n, replace=False) for c in (0, 1)]))
idx_C = np.sort(np.concatenate(keep))
idx_D = np.sort(rng.choice(idx_B, len(idx_C), replace=False))
conds = {"A_original": idx_A, "B_noBIDS": idx_B, "C_noBIDS_balanced": idx_C, "D_noBIDS_sizematched": idx_D}
for k, v in conds.items():
    log(f"  {k:<22} windows={len(v):5d}  %anx=" + " ".join(
        f"{d[:4]}={100*y[v][ds[v]==d].mean():.0f}" for d in BOTH if (ds[v] == d).any()))

rows = []; t0 = time.time()
for cname, ix in conds.items():
    Xc, yc, dc, gc = X[ix], y[ix], ds[ix], pid[ix]
    for sp in SPLITS:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=sp).split(np.zeros(len(yc)), yc, groups=gc))
        p = cached(PRED / f"{cname}_split{sp}.npz", lambda: {"p": predict(train_model(Xc[tr], yc[tr], 0), Xc[te])})["p"]
        prev = {d: yc[tr][dc[tr] == d].mean() for d in np.unique(dc)}
        prior = np.array([prev[d] for d in dc[te]])
        r = dict(condition=cname, split=sp, auc=roc_auc_score(yc[te], p), auc_prior=roc_auc_score(yc[te], prior))
        for d in BOTH:
            m = dc[te] == d
            r[f"auc_{d}"] = roc_auc_score(yc[te][m], p[m]) if m.sum() and len(np.unique(yc[te][m])) == 2 else np.nan
        rows.append(r)
        log(f"  {cname:<22} split {sp:<2} AUC={r['auc']:.3f}  prior={r['auc_prior']:.3f}   ({(time.time()-t0)/60:.0f} min)")
df = pd.DataFrame(rows); df.to_csv(OUT / "runs.csv", index=False)
log("\n=== mean over splits ===")
log(df.drop(columns="split").groupby("condition", sort=False).agg(["mean", "std"]).round(3).to_string())
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}")
