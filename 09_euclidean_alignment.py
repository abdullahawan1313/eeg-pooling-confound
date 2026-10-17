# 09_euclidean_alignment.py
"""
Step 09: Euclidean alignment (He & Wu, 2019) as a second correction method.

EA: for each PARTICIPANT, the mean spatial covariance R of their windows is computed and every
    window is mapped X -> R^(-1/2) X. This whitens the spatial signature of each participant
    (and hence of each dataset). No labels are used, but the unlabelled windows of the test
    participants are, so EA is a transductive, subject-level adaptation and is reported
    separately from plain LODO.

Pipeline: X_raw (uV) -> EA per participant -> per-channel per-window z-score (+/-10 clip)
Evaluation: within (4 datasets), pooled (3 splits), LODO (5 x 2 seeds), as in step 06.
Extra: dataset-identity probe (band-power LR, participant-wise 5-fold) before and after EA.

GPU: ~20-30 min. Output: outputs/09_ea/RESULTS.txt, runs.csv, summary.csv, identity.csv
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
from rev.common import load_v2, bandpower, DEVICE
from rev.loaders import zscore_clip
from rev.protocols import evaluate, summary_table

OUT = out("09_ea")
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def euclidean_align(Xr, groups):
    Xa = np.empty_like(Xr, dtype=np.float32)
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        W = Xr[ix].astype(np.float64)
        W = W - W.mean(axis=2, keepdims=True)
        R = np.einsum("nct,ndt->cd", W, W) / (W.shape[0] * W.shape[2])
        ev, V = np.linalg.eigh(R)
        ev = np.clip(ev, ev.max() * 1e-8, None)
        Rinv = V @ np.diag(ev ** -0.5) @ V.T
        Xa[ix] = np.einsum("cd,ndt->nct", Rinv, W).astype(np.float32)
    return Xa


def identity_probe(F, ds, pid):
    lab = pd.factorize(ds)[0]; accs = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, lab, pid):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[a], lab[a])
        accs.append((clf.predict(F[b]) == lab[b]).mean())
    return float(np.mean(accs)), float(np.bincount(lab).max() / len(lab))


t0 = time.time()
Xr, meta = load_v2(primary_only=True, raw=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
log(f"device {DEVICE} | raw corpus {Xr.shape}")
Xz, _ = load_v2(primary_only=True)
Xea = zscore_clip(euclidean_align(np.asarray(Xr), pid)).astype(np.float32)
del Xr
acc0, maj = identity_probe(bandpower(Xz), ds, pid)
acc1, _ = identity_probe(bandpower(Xea), ds, pid)
log(f"identity probe (band-power LR, participant-wise): baseline z-score={acc0:.3f}  EA={acc1:.3f}  majority={maj:.3f}")
del Xz

rows = evaluate(Xea, y, ds, pid, "EA", OUT / "preds", log=log)
df = pd.DataFrame(rows); df.to_csv(OUT / "runs.csv", index=False)
pd.DataFrame([dict(identity_baseline=acc0, identity_EA=acc1, majority=maj)]).to_csv(OUT / "identity.csv", index=False)
tab = summary_table(df); tab.to_csv(OUT / "summary.csv", index=False)
log("\n=== EA summary (compare with Step 5/6 cnn_transformer: LODO mean 0.536) ===")
log(tab.round(3).T.to_string())
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
