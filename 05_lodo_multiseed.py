# 05_lodo_multiseed.py
"""
Step 05: leave-one-dataset-out (LODO), 3 seeds, with uncertainty.
  - 5 held-out datasets x 3 seeds = 15 trainings (common training recipe)
  - For each held-out dataset: AUC with participant-cluster 95% CI (seed-averaged predictions),
    per-seed AUC (stability), degeneracy check, % predicted Anxious
  - Mean of the 4 two-class datasets: bootstrap CI and SESOI check (AUC 0.60), i.e. whether
    the upper CI bound lies below the smallest transfer of interest
  - Within-dataset (step 03) vs LODO table: transfer gap
  - The trained models are saved in outputs/05_lodo/models (not tracked in the repository)

GPU (Colab T4): ~30-45 min. CPU: ~10 h.
Output: outputs/05_lodo/RESULTS.txt, per_seed.csv, lodo_ci.csv, within_vs_lodo.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out, OUT_DIR
from rev.common import load_v2, train_model, predict, cluster_boot_auc, cached, DEVICE

OUT = out("05_lodo"); PRED = OUT / "preds"; MOD = OUT / "models"
PRED.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]
SESOI = 0.60
BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]

buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
DATASETS = sorted(np.unique(ds))
log(f"corpus {X.shape} | device {DEVICE}")
t0 = time.time()
per_seed, pav = [], {}
for hold in DATASETS:
    te = np.where(ds == hold)[0]; tr = np.where(ds != hold)[0]
    ps = []
    for seed in SEEDS:
        def run():
            m = train_model(X[tr], y[tr], seed)
            torch.save(m.state_dict(), MOD / f"lodo_{hold}_seed{seed}.pt")
            return {"p": predict(m, X[te])}
        p = cached(PRED / f"lodo_{hold}_seed{seed}.npz", run)["p"]
        ps.append(p)
        r = dict(held_out=hold, seed=seed, std=float(p.std()),
                 pct_pred_anx=float((p >= 0.5).mean() * 100))
        if hold in BOTH:
            r["auc"] = roc_auc_score(y[te], p)
        else:
            r["specificity"] = float((p < 0.5).mean())
        per_seed.append(r)
        log(f"{hold:<17} seed {seed}: " + (f"AUC={r['auc']:.3f}" if "auc" in r else
            f"specificity={r['specificity']:.3f}") + f"  %predAnx={r['pct_pred_anx']:.0f}"
            f"  std={r['std']:.3f}   ({(time.time()-t0)/60:.0f} min)")
    pav[hold] = np.mean(ps, axis=0)
pd.DataFrame(per_seed).to_csv(OUT / "per_seed.csv", index=False)

log("\n=== LODO, seed-averaged predictions, participant-cluster 95% CI ===")
ci_rows, boots = [], {}
for hold in BOTH:
    m = ds == hold
    a = roc_auc_score(y[m], pav[hold]); lo, hi, bv = cluster_boot_auc(y[m], pav[hold], pid[m])
    boots[hold] = bv
    sd = pd.DataFrame(per_seed).query("held_out == @hold").auc.std()
    ci_rows.append(dict(held_out=hold, auc=a, ci_lo=lo, ci_hi=hi, seed_sd=sd,
                        ci_excludes_0p5=bool(lo > 0.5 or hi < 0.5), ci_upper_below_SESOI=bool(hi < SESOI)))
    log(f"  {hold:<17} AUC={a:.3f}  CI[{lo:.3f},{hi:.3f}]  seed-SD={sd:.3f}")
k = min(len(v) for v in boots.values())
mb = np.mean([boots[h][:k] for h in BOTH], axis=0)
mean_auc = np.mean([r["auc"] for r in ci_rows]); lo, hi = np.percentile(mb, [2.5, 97.5])
ci_rows.append(dict(held_out="MEAN_4", auc=mean_auc, ci_lo=lo, ci_hi=hi,
                    ci_upper_below_SESOI=bool(hi < SESOI)))
log(f"  {'MEAN of 4':<17} AUC={mean_auc:.3f}  CI[{lo:.3f},{hi:.3f}]   "
    f"P(mean >= {SESOI}) = {np.mean(mb >= SESOI):.4f}")
pd.DataFrame(ci_rows).to_csv(OUT / "lodo_ci.csv", index=False)

# within vs LODO
wfile = OUT_DIR / "03_within" / "within_dataset_auc.csv"
if wfile.exists():
    w = pd.read_csv(wfile).query("model == 'cnn_transformer'").groupby("dataset").auc.mean()
    rows = [dict(dataset=h, within=w.get(h, np.nan),
                 lodo=[r["auc"] for r in ci_rows if r["held_out"] == h][0]) for h in BOTH]
    t = pd.DataFrame(rows); t["gap"] = t.within - t.lodo
    t.to_csv(OUT / "within_vs_lodo.csv", index=False)
    log("\n=== Within-dataset (Step 3) vs LODO (CNN-Transformer) ===")
    log(t.round(3).to_string(index=False))

(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
