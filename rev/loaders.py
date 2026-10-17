# rev/loaders.py
"""
Dataset loaders: band-pass 0.5-50 Hz, resampling to 128 Hz, the ten common channels,
4-s windows with 50 % overlap (BIDS: the first 4 s of each epoch), and the label rule
of each dataset.
  1. Windows are returned un-normalised (raw, uV); the z-score is applied later.
  2. Each recording carries metadata (recording id, task, rating, window index).
  3. Low-rated SAM 40 stress trials are also loaded (flag in_primary=False) so that
     the task-based labelling and the positive control can use them.

Each loader returns a list of dicts, one per RECORDING:
  {"x": (n_win, 10, 512) float32 raw, "y": int, "pid": str, "rec": str,
   "task": str, "rating": float, "in_primary": bool, "extra": dict}
"""
import re
from math import gcd
from collections import defaultdict
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from scipy.io import loadmat

from rev.paths import DATASET_DIRS

TARGET_FS = 128
BANDPASS = (0.5, 50.0)
WIN = 512
STEP = 256
CLIP_SD = 10.0
COMMON = ["F7", "F3", "F4", "F8", "T7", "T8", "P7", "P8", "O1", "O2"]
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
EMOTIV14 = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SAM40_THR = 5


# ------------------------------------------------------------------ common DSP
def bandpass_resample(x, fs_in):
    lo, hi = BANDPASS
    b, a = butter(4, [lo / (fs_in / 2), hi / (fs_in / 2)], btype="band")
    x = filtfilt(b, a, x, axis=1)
    if fs_in != TARGET_FS:
        g = gcd(int(fs_in), int(TARGET_FS))
        x = resample_poly(x, TARGET_FS // g, int(fs_in) // g, axis=1)
    return x


def windows_raw(x):
    """(ch, n) -> (n_win, ch, 512) raw float64, step 256 samples (50 % overlap)."""
    out = [x[:, s:s + WIN] for s in range(0, x.shape[1] - WIN + 1, STEP)]
    return np.array(out, dtype=np.float64) if out else np.zeros((0, x.shape[0], WIN))


def zscore_clip(w):
    """Per-channel, per-window z-score, clipped at +/-10 SD. w: (..., ch, 512)."""
    mu = w.mean(axis=-1, keepdims=True)
    sd = w.std(axis=-1, keepdims=True)
    sd[sd == 0] = 1.0
    return np.clip((w - mu) / sd, -CLIP_SD, CLIP_SD)


def _rec(x, y, pid, rec, task, rating=np.nan, in_primary=True, **extra):
    return dict(x=x.astype(np.float64), y=int(y), pid=pid, rec=rec, task=task,
                rating=float(rating), in_primary=bool(in_primary), extra=extra)


# ------------------------------------------------------------------ SAM40
def _sam40_ratings(path):
    import pandas as pd
    x = pd.read_excel(path, header=None)
    r = {}
    for i in range(2, x.shape[0]):
        s = x.iloc[i, 0]
        if pd.isna(s):
            continue
        s = int(s)
        for task, base in [("Maths", 1), ("Symmetry", 2), ("Stroop", 3)]:
            tr = [x.iloc[i, base], x.iloc[i, base + 3], x.iloc[i, base + 6]]
            r[(s, task)] = [float(t) if not pd.isna(t) else np.nan for t in tr]
    return r


def load_sam40():
    root = DATASET_DIRS["SAM40"]
    names = []
    with open(root / "Data" / "Coordinates.locs") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                names.append(p[-1].strip())
    up = [c.upper() for c in names]
    ch_idx = [up.index(c.upper()) for c in COMMON]
    ratings = _sam40_ratings(root / "Data" / "scales.xls")
    col = {"Arithmetic": "Maths", "Mirror_image": "Symmetry", "Stroop": "Stroop"}
    stress = ["Arithmetic", "Mirror_image", "Stroop"]
    rx = re.compile(r"(\w+?)_sub_(\d+)_trial(\d+)\.mat", re.IGNORECASE)
    recs = []
    for f in sorted((root / "Data" / "raw_data").glob("*.mat")):
        m = rx.match(f.name)
        if not m:
            continue
        task, subj, trial = m.group(1), int(m.group(2)), int(m.group(3))
        if task.lower().startswith("relax"):
            label, rating, prim, tname = 0, np.nan, True, "Relax"
        elif task in stress or task.lower() in [t.lower() for t in stress]:
            key = col.get(task) or col.get(task.capitalize())
            rl = ratings.get((subj, key), [np.nan] * 3)
            rating = rl[trial - 1] if trial - 1 < len(rl) else np.nan
            prim = (not np.isnan(rating)) and rating >= SAM40_THR
            label, tname = 1, task
        else:
            continue
        d = loadmat(f)["Data"][ch_idx, :]
        w = windows_raw(bandpass_resample(d, 128))
        if len(w) == 0:
            continue
        recs.append(_rec(w, label, f"SAM40_s{subj:02d}", f"SAM40_s{subj:02d}_{tname}_t{trial}",
                         tname, rating, prim, trial=trial))
    return recs


# ------------------------------------------------------------------ Mental Arithmetic
def load_ma():
    import mne
    root = DATASET_DIRS["MentalArithmetic"] / "eeg-during-mental-arithmetic-tasks-1.0.0"

    def clean(n):
        n = n.replace("EEG", "").strip().upper()
        return RENAME.get(n, n)

    def read(p):
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        cl = {clean(c): i for i, c in enumerate(raw.ch_names)}
        idx = [cl[c] for c in COMMON]
        return raw.get_data()[idx, :], raw.info["sfreq"]            # volts (converted to uV after filtering)

    edfs = sorted(root.glob("Subject*.edf"))
    subs = sorted(set(re.match(r"(Subject\d+)_", f.name).group(1) for f in edfs))
    recs = []
    for s in subs:
        fr, ft = root / f"{s}_1.edf", root / f"{s}_2.edf"
        if not (fr.exists() and ft.exists()):
            continue
        r, fsr = read(fr)
        t, fst = read(ft)
        # filter on the volt scale, then convert to uV
        r, t = bandpass_resample(r, fsr) * 1e6, bandpass_resample(t, fst) * 1e6
        L = min(r.shape[1], t.shape[1])
        for sig, lab, tn in [(r[:, :L], 0, "Rest"), (t[:, :L], 1, "Arithmetic")]:
            w = windows_raw(sig)
            if len(w):
                recs.append(_rec(w, lab, f"MA_{s}", f"MA_{s}_{tn}", tn))
    return recs


# ------------------------------------------------------------------ DASPS
def load_dasps():
    import mat73
    p = DATASET_DIRS["DASPS"] / "DASPS+HAM labels" / "DASPS+HAM labels.mat"
    R = mat73.loadmat(p)["Regim_datasub"]
    trials, labs = R["trial"], R["label"]
    ch_idx = [EMOTIV14.index(c) for c in COMMON]
    recs = []
    for i in range(len(trials)):
        sig = np.array(trials[i])
        if sig.ndim == 3:
            sig = sig.squeeze()
        va = np.array(labs[i]).flatten()[:2]
        val, aro = float(va[0]), float(va[1])
        label = 1 if (val <= 4 and aro >= 5) else 0
        w = windows_raw(bandpass_resample(sig[ch_idx, :], 128))
        if len(w) == 0:
            continue
        subj = i // 12 + 1
        recs.append(_rec(w, label, f"DASPS_s{subj:02d}", f"DASPS_s{subj:02d}_seg{i % 12:02d}",
                         "ImageryRecall", np.nan, True, valence=val, arousal=aro, seg=i % 12))
    return recs


# ------------------------------------------------------------------ Mendeley
def load_mendeley():
    root = DATASET_DIRS["Mendeley"] / "Anxiety_Dataset" / "Multi-channel Wireless EEG Recordings"
    ch_idx = [EMOTIV14.index(c) for c in COMMON]
    recs = []
    for folder, label in [("Anxiety_Positive", 1), ("Anxiety_Negative", 0)]:
        for f in sorted((root / folder).glob("*.mat")):
            m = loadmat(f)
            x = np.array(m[[k for k in m if not k.startswith("__")][0]])
            if x.shape[0] != 14 and x.shape[1] == 14:
                x = x.T
            w = windows_raw(bandpass_resample(x[ch_idx, :], 128))
            if len(w):
                recs.append(_rec(w, label, f"MEN_{f.stem}", f"MEN_{f.stem}", "Recording"))
    return recs


# ------------------------------------------------------------------ BIDS
def load_bids():
    import mne
    root = DATASET_DIRS["BIDS"]
    by = defaultdict(dict)
    for p in root.rglob("*.set"):
        ms, me = re.search(r"sub-(\w+?)_", p.name), re.search(r"ses-(\w+?)_", p.name)
        if ms:
            by[ms.group(1)][me.group(1) if me else "t1"] = p
    recs = []
    for subj, sess in sorted(by.items()):
        ses = "t1" if "t1" in sess else ("t2" if "t2" in sess else list(sess)[0])
        ep = mne.io.read_epochs_eeglab(sess[ses], verbose="ERROR")
        up = [c.upper() for c in ep.ch_names]
        idx = [up.index(c.upper()) for c in COMMON]
        data = ep.get_data()[:, idx, :]                               # Volt
        ws = []
        for e in range(data.shape[0]):
            w = bandpass_resample(data[e], ep.info["sfreq"])        # filter on the volt scale
            if w.shape[1] >= WIN:
                ws.append(w[:, :WIN] * 1e6)                            # -> uV
        if ws:
            recs.append(_rec(np.stack(ws), 0, f"BIDS_s{subj}", f"BIDS_s{subj}_ses-{ses}",
                             "RestEyesClosed"))
    return recs


LOADERS = {"SAM40": load_sam40, "MentalArithmetic": load_ma, "DASPS": load_dasps,
           "Mendeley": load_mendeley, "BIDS": load_bids}       # dataset order used throughout
