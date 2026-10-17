# 04_pooled_repeated.py
"""
Step 04: pooled participant-wise evaluation, repeated.
  - 5 participant-level 80/20 splits (split 42 = split of the earlier analysis)
    x 3 training seeds = 15 runs
  - For each run:
      * pooled AUC (all test windows)
      * pooled AUC without BIDS
      * per-dataset AUC (held-out participants of that dataset only)
      * dataset-prior baseline on the same split (training prevalence of each dataset as score)
  - between-dataset share: pooled AUC - window-weighted mean of the per-dataset AUCs
  - the held-out features of split 42 / seed 0 are saved for the held-out t-SNE (step 15)

GPU (Colab T4): ~25-40 min. CPU: ~10 h.
Resumable: the predictions of every run are saved and skipped on a re-run.
Output: outputs/04_pooled/RESULTS.txt, runs.csv, summary.csv, split42_ci.csv
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
from rev.common import load_v2, train_model, predict, cluster_boot_auc, cached, DEVICE

OUT = out("04_pooled"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
SPLITS = [42, 0, 1, 2, 3]
SEEDS = [0, 1, 2]
BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]

buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
log(f"corpus {X.shape} | device {DEVICE}")
t0 = time.time()
rows = []
for sp in SPLITS:
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=sp)
                  .split(np.zeros(len(y)), y, groups=pid))
    prev = {d: y[tr][ds[tr] == d].mean() for d in np.unique(ds)}
    s_prior = np.array([prev[d] for d in ds[te]])
    auc_prior = roc_auc_score(y[te], s_prior)
    for seed in SEEDS:
        want_feat = (sp == 42 and seed == 0)

        def run():
            m = train_model(X[tr], y[tr], seed)
            if want_feat:
                p, f = predict(m, X[te], want_feat=True)
                return {"p": p, "te": te, "feat": f}
            return {"p": predict(m, X[te]), "te": te}

        r = cached(PRED / f"pooled_split{sp}_seed{seed}.npz", run)
        p = r["p"]; yt, dt, gt = y[te], ds[te], pid[te]
        row = dict(split=sp, seed=seed, auc_pooled=roc_auc_score(yt, p),
                   auc_pooled_noBIDS=roc_auc_score(yt[dt != "BIDS"], p[dt != "BIDS"]),
                   auc_prior=auc_prior)
        w_sum, w_auc = 0, 0.0
        for d in BOTH:
            m = dt == d
            if len(np.unique(yt[m])) == 2:
                a = roc_auc_score(yt[m], p[m]); row[f"auc_{d}"] = a
                w_sum += m.sum(); w_auc += a * m.sum()
            else:
                row[f"auc_{d}"] = np.nan          # only one class of this dataset in this split
            row[f"ntest_{d}"] = len(np.unique(gt[m]))
        row["auc_within_weighted"] = w_auc / w_sum
        row["between_dataset_share"] = row["auc_pooled"] - row["auc_within_weighted"]
        rows.append(row)
        log(f"split {sp:>2} seed {seed}: pooled={row['auc_pooled']:.3f} noBIDS={row['auc_pooled_noBIDS']:.3f} "
            f"prior={auc_prior:.3f} | " + " ".join(f"{d[:4]}={row[f'auc_{d}']:.3f}" for d in BOTH)
            + f"   ({(time.time()-t0)/60:.0f} min)")

df = pd.DataFrame(rows); df.to_csv(OUT / "runs.csv", index=False)
cols = ["auc_pooled", "auc_pooled_noBIDS", "auc_prior", "auc_within_weighted",
        "between_dataset_share"] + [f"auc_{d}" for d in BOTH]
summ = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(),
                     "min": df[cols].min(), "max": df[cols].max(),
                     "n_runs_valid": df[cols].notna().sum()})
summ.to_csv(OUT / "summary.csv")
log("\n=== SUMMARY over 15 runs (5 splits x 3 seeds) ===")
log(summ.round(3).to_string())

# cluster CIs for split 42, seed-averaged predictions
log("\n=== Split 42 (split of the earlier analysis), predictions averaged over 3 seeds, participant-cluster 95% CI ===")
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
              .split(np.zeros(len(y)), y, groups=pid))
pav = np.mean([np.load(PRED / f"pooled_split42_seed{s}.npz")["p"] for s in SEEDS], axis=0)
yt, dt, gt = y[te], ds[te], pid[te]
ci_rows = []
for name, m in [("ALL", np.ones(len(yt), bool)), ("noBIDS", dt != "BIDS")] + [(d, dt == d) for d in BOTH]:
    if len(np.unique(yt[m])) < 2:
        log(f"  {name:<17} single class in test"); continue
    a = roc_auc_score(yt[m], pav[m]); lo, hi, _ = cluster_boot_auc(yt[m], pav[m], gt[m])
    ci_rows.append(dict(subset=name, auc=a, ci_lo=lo, ci_hi=hi, n_part=len(np.unique(gt[m]))))
    log(f"  {name:<17} AUC={a:.3f}  CI[{lo:.3f},{hi:.3f}]  participants={len(np.unique(gt[m]))}")
pd.DataFrame(ci_rows).to_csv(OUT / "split42_ci.csv", index=False)

(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
