# 07_positive_control.py
"""
Step 07: positive control.

SAM 40 and Mental Arithmetic both contain the same contrast, mental arithmetic vs rest/relax,
a known shared EEG effect (lower alpha and higher frontal theta during arithmetic). If the same
pipeline (10 channels, 4-s windows, z-score, CNN-Transformer) transfers this contrast from
SAM 40 to MA, the failure to transfer the anxiety labels cannot be explained by an insensitive
pipeline.

Labels (task labels, not anxiety labels):
  SAM40: all Arithmetic trials (independent of rating) = 1, Relax = 0
  MA   : Arithmetic = 1, Rest = 0
Experiments:
  A. within SAM40-arithmetic and within MA (5-fold, participant-wise)
  B. transfer SAM40-arithmetic -> MA and MA -> SAM40-arithmetic (3 seeds, cluster CI)
  C. variant: any SAM 40 stress task vs Relax -> MA
  (band power + LR is reported alongside in every case)

GPU (Colab): ~10-15 min.
Output: outputs/07_positive_control/RESULTS.txt, results.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, train_model, predict, bandpower, cluster_boot_auc, cached, DEVICE

OUT = out("07_positive_control"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]

buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

Xall, meta = load_v2(primary_only=False)
sam = meta.dataset.values == "SAM40"; ma = meta.dataset.values == "MentalArithmetic"
task = meta.task.values
sets = {
    "SAM40_arith": np.where(sam & np.isin(task, ["Arithmetic", "Relax"]))[0],
    "SAM40_anytask": np.where(sam)[0],
    "MA": np.where(ma)[0],
}
yv = np.where(np.isin(task, ["Relax", "Rest"]), 0, 1)        # task=1, rest=0 (not the anxiety label)
pid = meta.participant.values
idx_all = np.unique(np.concatenate(list(sets.values())))
Xs = np.asarray(Xall[idx_all]); pos = {g: k for k, g in enumerate(idx_all)}
def getX(ix): return Xs[[pos[i] for i in ix]]
BPs = bandpower(Xs)
def getBP(ix): return BPs[[pos[i] for i in ix]]
log(f"device {DEVICE}")
for k, v in sets.items():
    log(f"  {k:<14} windows={len(v)}  task={int(yv[v].sum())}  rest={int((yv[v]==0).sum())}  "
        f"participants={len(np.unique(pid[v]))}")
t0 = time.time(); rows = []


def report(name, model, ytrue, p, g):
    a = roc_auc_score(ytrue, p); lo, hi, _ = cluster_boot_auc(ytrue, p, g)
    rows.append(dict(experiment=name, model=model, auc=a, ci_lo=lo, ci_hi=hi))
    log(f"  {name:<28} {model:<16} AUC={a:.3f}  CI[{lo:.3f},{hi:.3f}]   ({(time.time()-t0)/60:.0f} min)")


# A. within
for sname in ["SAM40_arith", "MA"]:
    ix = sets[sname]; yy = yv[ix]; gg = pid[ix]
    pc, pl = np.full(len(ix), np.nan), np.full(len(ix), np.nan)
    for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ix, yy, gg)):
        r = cached(PRED / f"within_{sname}_f{k}.npz",
                   lambda: {"p": predict(train_model(getX(ix[a]), yy[a], 0), getX(ix[b]))})
        pc[b] = r["p"]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(getBP(ix[a]), yy[a])
        pl[b] = clf.predict_proba(getBP(ix[b]))[:, 1]
    report(f"within {sname}", "cnn_transformer", yy, pc, gg)
    report(f"within {sname}", "bandpower_LR", yy, pl, gg)

# B/C. transfer
for src, dst in [("SAM40_arith", "MA"), ("MA", "SAM40_arith"), ("SAM40_anytask", "MA")]:
    a, b = sets[src], sets[dst]
    ps = [cached(PRED / f"transfer_{src}_to_{dst}_seed{s}.npz",
                 lambda: {"p": predict(train_model(getX(a), yv[a], s), getX(b))})["p"] for s in SEEDS]
    for s, p in zip(SEEDS, ps):
        log(f"    seed {s}: AUC={roc_auc_score(yv[b], p):.3f}")
    report(f"transfer {src} -> {dst}", "cnn_transformer", yv[b], np.mean(ps, 0), pid[b])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(getBP(a), yv[a])
    report(f"transfer {src} -> {dst}", "bandpower_LR", yv[b], clf.predict_proba(getBP(b))[:, 1], pid[b])

pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
log("\nReference (anxiety labels, Step 5 LODO): SAM40 0.506, MA 0.555")
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}")
