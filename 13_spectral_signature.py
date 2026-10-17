# 13_spectral_signature.py
"""
Step 13: physiological characterisation of the dataset signature. CPU, ~5-10 min, no training.
Uses X_raw (uV).

1. Average spectrum of each dataset (figure)
2. Spectral parameterisation (FOOOF) per participant: aperiodic exponent and offset,
   alpha peak frequency and power
3. Which features carry dataset identity? Participant-level LR: aperiodic vs alpha vs both
4. Frontal alpha asymmetry (FAA = ln alpha F4 - ln alpha F3; F8-F7), a classic anxiety marker:
   AUC against the label WITHIN each dataset (no training) + participant-cluster CI
5. Age vs alpha peak frequency (age is available for MA and BIDS)

Output: outputs/13_spectral/RESULTS.txt, participant_features.csv, faa_auc.csv, fig_spectra.png
"""
import sys, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import out, META
from rev.common import load_v2, cluster_boot_auc
from rev.loaders import COMMON

OUT = out("13_spectral")
buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

X, meta = load_v2(primary_only=True, raw=True)
y, ds, pid = meta.y.values, meta.dataset.values, meta.participant.values
DSETS = ["BIDS", "DASPS", "Mendeley", "MentalArithmetic", "SAM40"]
log(f"raw corpus {X.shape}")

# ---------------- PSD per window
P = []
for i in range(0, len(X), 2000):
    f, p = welch(np.asarray(X[i:i + 2000], dtype=np.float64), fs=128, nperseg=256, axis=2)
    P.append(p)
P = np.concatenate(P)                                     # (N, 10, 129)
logP = np.log10(P + 1e-12)
alpha = (f >= 8) & (f <= 13)
ch = {c: i for i, c in enumerate(COMMON)}

# ---------------- 1. figure
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fm = (f >= 1) & (f <= 45)
    for d in DSETS:
        parts = np.unique(pid[ds == d])
        S = np.array([logP[pid == g].mean(axis=(0, 1)) for g in parts])[:, fm]
        mu, se = S.mean(0), S.std(0) / np.sqrt(len(S))
        ax.plot(f[fm], mu, label=f"{d} (n={len(parts)})"); ax.fill_between(f[fm], mu - 1.96 * se, mu + 1.96 * se, alpha=.2)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("log10 power (uV^2/Hz)")
    ax.set_title("Participant-averaged spectra (10 common channels)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_spectra.png", dpi=300); plt.close(fig)
    log("saved fig_spectra.png")
except Exception as e:
    log("figure error:", e)

# ---------------- 2. FOOOF per participant
# NOTE: in the DASPS data as provided, power below ~4 Hz is attenuated (pre-filtering by the
#       provider), so FOOOF is fitted on the common valid range 4-40 Hz for all datasets.
from fooof import FOOOF
FIT_RANGE = [4, 40]
rows = []
for g in np.unique(pid):
    m = pid == g
    spec = P[m].mean(axis=(0, 1))
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, spec, FIT_RANGE)
    off, exp_ = fm.aperiodic_params_
    pk = fm.peak_params_
    a = pk[(pk[:, 0] >= 7) & (pk[:, 0] <= 14)] if len(pk) else np.zeros((0, 3))
    cf, pw = (a[np.argmax(a[:, 1])][:2] if len(a) else (np.nan, np.nan))
    rows.append(dict(participant=g, dataset=ds[m][0], label_mean=y[m].mean(), offset=off, exponent=exp_,
                     alpha_cf=cf, alpha_pw=pw, r2=fm.r_squared_,
                     rel_alpha=P[m][:, :, alpha].sum(2).mean() / P[m][:, :, (f >= 1) & (f <= 45)].sum(2).mean(),
                     low_ratio=P[m][:, :, (f >= 1) & (f < 3)].mean() / P[m][:, :, (f >= 5) & (f < 7)].mean(),
                     hf_ratio=P[m][:, :, (f >= 30) & (f < 40)].mean() / P[m][:, :, (f >= 8) & (f < 13)].mean()))
F = pd.DataFrame(rows); F.to_csv(OUT / "participant_features.csv", index=False)
log("\n=== FOOOF per dataset (participant level, mean +/- sd) ===")
log(F.groupby("dataset")[["offset", "exponent", "alpha_cf", "alpha_pw", "rel_alpha", "low_ratio", "hf_ratio", "r2"]]
    .agg(["mean", "std"]).round(2).to_string())

# ---------------- 3. which features carry identity?
log("\n=== Dataset identity from participant-level features (5-fold LR) ===")
lab = pd.factorize(F.dataset)[0]; maj = np.bincount(lab).max() / len(lab)
Fi = F.fillna(F.median(numeric_only=True))
for name, cols in [("aperiodic (offset, exponent)", ["offset", "exponent"]),
                   ("alpha (cf, pw, rel)", ["alpha_cf", "alpha_pw", "rel_alpha"]),
                   ("filter/noise (1-3 Hz & 30-40 Hz ratios)", ["low_ratio", "hf_ratio"]),
                   ("all", ["offset", "exponent", "alpha_cf", "alpha_pw", "rel_alpha", "low_ratio", "hf_ratio"])]:
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          Fi[cols].values, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    log(f"  {name:<30} acc={acc:.3f}  (majority {maj:.3f})")

# ---------------- 4. FAA within each dataset
log("\n=== Frontal alpha asymmetry vs label, WITHIN each dataset (no training) ===")
log("  hypothesis: anxious -> lower FAA (relatively more right-frontal activity). AUC uses -FAA.")
la = np.log(P[:, :, alpha].mean(2) + 1e-12)
fr = []
for pair in [("F4", "F3"), ("F8", "F7")]:
    faa = la[:, ch[pair[0]]] - la[:, ch[pair[1]]]
    for d in ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]:
        m = ds == d
        a_ = roc_auc_score(y[m], -faa[m]); lo, hi, _ = cluster_boot_auc(y[m], -faa[m], pid[m], n_boot=1000)
        fr.append(dict(pair=f"{pair[0]}-{pair[1]}", dataset=d, auc=a_, ci_lo=lo, ci_hi=hi))
        log(f"  {pair[0]}-{pair[1]}  {d:<17} AUC={a_:.3f}  CI[{lo:.3f},{hi:.3f}]")
pd.DataFrame(fr).to_csv(OUT / "faa_auc.csv", index=False)

# ---------------- 5. age vs alpha peak
log("\n=== Age vs alpha peak frequency (MA + BIDS) ===")
try:
    ma = pd.read_csv(META["MA_subject_info"]); ma["participant"] = "MA_" + ma.Subject
    b = pd.read_csv(META["BIDS_participants"], sep="\t"); b["participant"] = b.participant_id.str.replace("sub-", "BIDS_s")
    ages = pd.concat([ma[["participant", "Age"]].rename(columns={"Age": "age"}), b[["participant", "age"]]])
    A = F.merge(ages, on="participant").dropna(subset=["alpha_cf"])
    r, pval = spearmanr(A.age, A.alpha_cf)
    log(f"  n={len(A)}  Spearman rho(age, alpha CF)={r:.2f}  p={pval:.3g}")
    log(A.groupby("dataset")[["age", "alpha_cf", "exponent"]].mean().round(2).to_string())
except Exception as e:
    log("  age analysis skipped:", e)

(OUT / "RESULTS.txt").write_text(buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}")
