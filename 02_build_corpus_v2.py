# 02_build_corpus_v2.py
"""
Step 02: builds the harmonised corpus used by all later steps. No training.
Time: 10-25 min (the BIDS .set files are the slowest). RAM: ~2 GB.

Writes outputs/02_corpus_v2/:
  X_raw.npy   (N,10,512) float32  filtered and resampled, uV, not normalised
                                   (used by the normalisation and spectral analyses)
  X_z.npy     (N,10,512) float32  per-channel, per-window z-score (clipped at +/-10)
  meta.csv    one row per window: dataset, participant, y, rec (recording id), task,
              rating, in_primary, win_idx, + valence/arousal/seg (DASPS), trial (SAM40)
  VERIFY.txt  log, including an optional comparison with the corpus of the earlier analysis

in_primary=True  -> primary corpus (14,655 windows)
in_primary=False -> low-rated (<5) SAM 40 stress trials (sensitivity analysis and positive
                    control only)

Run: python 02_build_corpus_v2.py
"""
import sys, time, io
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import OLD_CORPUS, out
from rev.loaders import LOADERS, zscore_clip

OUT = out("02_corpus_v2")
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

t0 = time.time()
Xr, rows = [], []
for ds, fn in LOADERS.items():
    t = time.time()
    log(f">>> loading {ds} ...")
    recs = fn()
    for r in recs:
        n = len(r["x"])
        Xr.append(r["x"])
        for k in range(n):
            rows.append(dict(dataset=ds, participant=r["pid"], y=r["y"], rec=r["rec"],
                             task=r["task"], rating=r["rating"], in_primary=r["in_primary"],
                             win_idx=k, **r["extra"]))
    nw = sum(len(r["x"]) for r in recs)
    log(f"    {ds}: recordings={len(recs)} windows={nw}  ({time.time()-t:.0f}s)")

X_raw = np.concatenate(Xr, axis=0)                     # float64
meta = pd.DataFrame(rows)
log(f"\nTotal windows: {len(meta)}  | in_primary: {int(meta.in_primary.sum())}")

X_z = zscore_clip(X_raw).astype(np.float32)            # per-channel per-window z-score (float64 -> float32)
X_raw = X_raw.astype(np.float32)

# ---------------- amplitude summary (one part of the dataset signature) ----------------
log("\nPer-dataset amplitude (median channel SD in raw uV, before z-score):")
sd = X_raw.std(axis=2)                                  # (N,10)
for ds in LOADERS:
    m = (meta.dataset == ds).values
    log(f"  {ds:<17} median SD = {np.median(sd[m]):9.3f}   IQR = "
        f"[{np.percentile(sd[m],25):.3f}, {np.percentile(sd[m],75):.3f}]")

# ---------------- optional check against the corpus of the earlier analysis ----------------
if not OLD_CORPUS.exists():
    log("\nVERIFY skipped: the corpus of the earlier analysis is not available (optional check).")
else:
  log("\nVERIFY: in_primary windows vs earlier corpus.npz")
  old = np.load(OLD_CORPUS, allow_pickle=True)
  oX, oy, od, op = old["X"], old["y"], old["dataset"].astype(str), old["participant"].astype(str)
  prim = meta.in_primary.values
  all_ok = True
  for ds in LOADERS:
      mo = od == ds
      mn = prim & (meta.dataset.values == ds)
      if mo.sum() != mn.sum():
          log(f"  [FAIL] {ds}: old windows={mo.sum()} new={mn.sum()}")
          all_ok = False
          continue
      same_y = np.array_equal(oy[mo], meta.y.values[mn])
      same_p = np.array_equal(op[mo], meta.participant.values[mn])
      diff = float(np.abs(oX[mo] - X_z[mn]).max())
      ok = same_y and same_p and diff < 1e-3
      all_ok &= ok
      log(f"  [{'OK  ' if ok else 'FAIL'}] {ds:<17} n={mo.sum():5d} labels_same={same_y} "
          f"participants_same={same_p} max|dX|={diff:.2e}")
  log("RESULT: " + ("corpus matches the earlier corpus (primary windows)."
                    if all_ok else "MISMATCH with the earlier corpus."))

np.save(OUT / "X_raw.npy", X_raw)
np.save(OUT / "X_z.npy", X_z)
meta.to_csv(OUT / "meta.csv", index=False)
log(f"\nSaved -> {OUT}   ({time.time()-t0:.0f}s total)")
(OUT / "VERIFY.txt").write_text(buf.getvalue(), encoding="utf-8")
