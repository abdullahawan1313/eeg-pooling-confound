# 12_sam40_task_labels.py
"""
Step 12: sensitivity of the results to the SAM 40 labelling.

Primary : Relax = Normal; a stress trial is Anxious only if its self-rating is >= 5
          (other stress trials are dropped)
Task    : Relax = Normal; EVERY stress trial (Stroop, Arithmetic, Mirror image) = Anxious
          (rating ignored)
The other 4 datasets are unchanged. Evaluation: within SAM40, pooled (3 splits),
LODO (5 x 2 seeds).

GPU: ~15-20 min. Output: outputs/12_sam40_task/RESULTS.txt, runs.csv, summary.csv
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

OUT = out("12_sam40_task")
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

t0 = time.time()
X, meta = load_v2(primary_only=False)          # includes the low-rated SAM 40 stress trials
X = np.asarray(X)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
s = ds == "SAM40"
log(f"device {DEVICE} | corpus {X.shape} | SAM40 windows={s.sum()} (%Anxious={100*y[s].mean():.1f}; primary was 63.0)")
rows = evaluate(X, y, ds, pid, "SAM40_taskbased", OUT / "preds", log=log, within_datasets=["SAM40"])
df = pd.DataFrame(rows); df.to_csv(OUT / "runs.csv", index=False)
tab = summary_table(df); tab.to_csv(OUT / "summary.csv", index=False)
log("\n=== SAM40 task-based labels (primary reference: within SAM40 0.659, LODO SAM40 0.506, pooled 0.670) ===")
log(tab.round(3).T.to_string())
(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"Saved -> {OUT}  ({(time.time()-t0)/60:.0f} min)")
