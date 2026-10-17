# 16_state_only_lodo.py
"""
Step 16: state-only LODO with 3 seeds and participant-cluster CIs.
Earlier single-seed result: DASPS 0.477, SAM40 0.523, MentalArithmetic 0.538.

Only the 3 state-labelled datasets (DASPS, SAM40, MentalArithmetic); each one is held out
and the model is trained on the other 2. CNN-Transformer, common training recipe, seeds 0/1/2.
Note: restricting to state labels does not make the constructs equivalent (stress rating vs
imagery-induced affect vs cognitive load); it only removes the mixing of trait and state labels.

GPU (Colab): ~10-15 min. Resumable.
Output: outputs/16_state_only/RESULTS.txt, per_seed.csv, ci.csv, fig_state_only.png
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, train_model, predict, cluster_boot_auc, cached, DEVICE

OUT = out("16_state_only"); PRED = OUT / "preds"; PRED.mkdir(exist_ok=True)
STATE = ["DASPS", "SAM40", "MentalArithmetic"]
SEEDS = [0, 1, 2]
OLD = {"DASPS": 0.477, "SAM40": 0.523, "MentalArithmetic": 0.538}
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True)
keep = np.where(meta.dataset.isin(STATE).values)[0]
X = np.asarray(X[keep]); meta = meta.iloc[keep].reset_index(drop=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
log(f"state-only corpus {X.shape} | device {DEVICE}")
t0 = time.time(); per, ci = [], []
for hold in STATE:
    te = np.where(ds == hold)[0]; tr = np.where(ds != hold)[0]
    ps = []
    for s in SEEDS:
        p = cached(PRED / f"state_{hold}_seed{s}.npz", lambda: {"p": predict(train_model(X[tr], y[tr], s), X[te])})["p"]
        ps.append(p); per.append(dict(held_out=hold, seed=s, auc=roc_auc_score(y[te], p), pred_std=float(p.std())))
        log(f"  {hold:<17} seed {s}: AUC={per[-1]['auc']:.3f}   ({(time.time()-t0)/60:.0f} min)")
    pav = np.mean(ps, 0); a = roc_auc_score(y[te], pav); lo, hi, _ = cluster_boot_auc(y[te], pav, pid[te])
    ci.append(dict(held_out=hold, auc=a, ci_lo=lo, ci_hi=hi, old_single_seed=OLD[hold],
                   seed_sd=pd.DataFrame(per).query("held_out == @hold").auc.std()))
    log(f"  {hold:<17} seed-avg AUC={a:.3f} CI[{lo:.3f},{hi:.3f}]  (earlier 1-seed: {OLD[hold]:.3f})")
pd.DataFrame(per).to_csv(OUT / "per_seed.csv", index=False)
c = pd.DataFrame(ci); c.to_csv(OUT / "ci.csv", index=False)
log(f"\n  mean of 3 = {c.auc.mean():.3f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = {"DASPS": "DASPS", "SAM40": "SAM 40", "MentalArithmetic": "Mental Arith."}
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    for i, r in enumerate(c.itertuples()):
        ax.errorbar(r.auc, i, xerr=[[r.auc - r.ci_lo], [r.ci_hi - r.auc]], fmt="o", color="#2a78d6", ms=6, elinewidth=2)
        ax.text(r.ci_hi + 0.01, i, f"{r.auc:.3f}", va="center", fontsize=7)
    ax.axvline(0.5, color="#6b6b6b", ls="--", lw=1); ax.axvline(0.60, color="#e34948", ls=":", lw=1.2)
    ax.set_yticks(range(len(c))); ax.set_yticklabels([names[h] for h in c.held_out]); ax.invert_yaxis()
    ax.set_xlim(0.38, 0.72); ax.set_xlabel("State-only LODO AUC (3 seeds, 95% cluster CI)")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_state_only.png", dpi=300, facecolor="white"); plt.close(fig)
except Exception as e:
    log("figure error:", e)
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}")
