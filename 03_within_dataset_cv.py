# 03_within_dataset_cv.py
"""
Step 03: within-dataset, participant-wise performance.
Question: can the same pipeline (10 channels, 4-s windows, per-window z-score, CNN-Transformer)
recognise the label in new participants of the same dataset?
  - If yes, the pipeline is sensitive, and a LODO failure is a failure of transfer.
  - If no, the pipeline itself is weak, and LODO says nothing about transfer.

Design:
  4 two-class datasets (DASPS, SAM40, MentalArithmetic, Mendeley)
  5-fold StratifiedGroupKFold (participant = group) x 3 seeds
  Model A: CNN-Transformer (common training recipe, 15 epochs)
  Model B: band-power + logistic regression (classical reference)
  The predictions of every fold are saved, so an interrupted run resumes and skips
  finished folds.

Time: ~1.5-2.5 h on 2 CPU threads, ~30-40 min on a GPU.
Output: outputs/03_within/RESULTS.txt, within_dataset_auc.csv, preds/
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, train_model, predict, bandpower, auc_row, cached

OUT = out("03_within"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
DATASETS = ["DASPS", "SAM40", "MentalArithmetic", "Mendeley"]
SEEDS = [0, 1, 2]
N_FOLDS = 5

buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
log(f"corpus: {X.shape}")
t0 = time.time()
rows = []
for ds in DATASETS:
    m = np.where(meta.dataset.values == ds)[0]
    Xd, yd, gd = X[m], meta.y.values[m], meta.participant.values[m]
    BP = bandpower(Xd)
    log(f"\n=== {ds}: windows={len(m)} participants={len(np.unique(gd))} ===")
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = list(cv.split(Xd, yd, groups=gd))

        def run_cnn():
            p = np.full(len(yd), np.nan)
            for k, (tr, te) in enumerate(folds):
                fp = PRED / f"{ds}_cnn_seed{seed}_fold{k}.npz"
                r = cached(fp, lambda: {"p": predict(train_model(Xd[tr], yd[tr], seed), Xd[te]),
                                        "te": te})
                p[r["te"]] = r["p"]
            return p

        def run_lr():
            p = np.full(len(yd), np.nan)
            for tr, te in folds:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                clf.fit(BP[tr], yd[tr]); p[te] = clf.predict_proba(BP[te])[:, 1]
            return p

        for name, fn in [("bandpower_LR", run_lr), ("cnn_transformer", run_cnn)]:
            p = fn()
            r = auc_row(yd, p, gd, dataset=ds, model=name, seed=seed)
            rows.append(r)
            log(f"  seed {seed}  {name:<16} AUC={r['auc']:.3f}  CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
                f"   ({(time.time()-t0)/60:.0f} min elapsed)")

df = pd.DataFrame(rows)
df.to_csv(OUT / "within_dataset_auc.csv", index=False)
summ = df.groupby(["dataset", "model"]).agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                             ci_lo_mean=("ci_lo", "mean"), ci_hi_mean=("ci_hi", "mean"))
log("\n=== SUMMARY (mean over seeds) ===")
log(summ.round(3).to_string())
log("\nLODO of the earlier analysis (CNN-Transformer, 1 seed), for reference: DASPS 0.514, SAM40 0.537, MA 0.581, Mendeley 0.507")
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
