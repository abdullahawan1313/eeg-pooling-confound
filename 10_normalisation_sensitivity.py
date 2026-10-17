# 10_normalisation_sensitivity.py
"""
Step 10: normalisation sensitivity.
The per-channel, per-window z-score removes absolute amplitude and inter-channel power ratios
(e.g. frontal alpha asymmetry). Two scalings that keep them are tested.

Variants (from X_raw in uV; the same rule for training and test data, no statistic is fitted
on test data):
  A. window_global : remove the per-channel mean (DC), then divide the WHOLE window (all
                     channels) by one scalar, the overall SD of that window. Inter-channel ratios
                     are kept, absolute amplitude is removed. The scalar comes from the window
                     itself (no leakage).
  B. fixed_scale   : remove the DC, then divide by a fixed a-priori constant (10 uV); clip +/-20.
                     Absolute amplitude and inter-channel ratios are both kept. No fitted statistic.
Reference: main pipeline (per-channel per-window z-score), steps 03-06.

Evaluation: within (4), pooled (3 splits), LODO (5 x 2 seeds), as in steps 06 and 09.
GPU: ~40-60 min. Output: outputs/10_norm/RESULTS.txt, runs.csv, summary.csv
"""
import sys, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out
from rev.common import load_v2, DEVICE
from rev.protocols import evaluate, summary_table

OUT = out("10_norm")
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

t0 = time.time()
Xr, meta = load_v2(primary_only=True, raw=True)
Xr = np.asarray(Xr, dtype=np.float32)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
log(f"device {DEVICE} | raw corpus {Xr.shape}")
Xr -= Xr.mean(axis=2, keepdims=True)                         # per-channel DC removal (both variants)

rows = []
# ---- A: window-global scalar (ratio preserving)
s = Xr.std(axis=(1, 2), keepdims=True); s[s == 0] = 1
XA = np.clip(Xr / s, -10, 10).astype(np.float32)
rows += evaluate(XA, y, ds, pid, "window_global", OUT / "preds", log=log)
del XA
pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
# ---- B: fixed a-priori scale (absolute amplitude preserving)
XB = np.clip(Xr / 10.0, -20, 20).astype(np.float32)
del Xr
rows += evaluate(XB, y, ds, pid, "fixed_scale", OUT / "preds", log=log)
df = pd.DataFrame(rows); df.to_csv(OUT / "runs.csv", index=False)
tab = summary_table(df); tab.to_csv(OUT / "summary.csv", index=False)
log("\n=== Normalisation sensitivity (reference main pipeline: pooled 0.670, LODO mean 0.536) ===")
log(tab.round(3).T.to_string())
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
