# 17_old_figures_redraw.py
"""
Step 17: figures carried over from the earlier version of the manuscript, redrawn with the
new results in the style of step 15. No training. CPU, ~3-6 min (band power and spectra are
the slowest).

Output: outputs/15_figures/ (next to the step 15 figures, 300 dpi PNG + PDF)
  fig_label_balance     per-dataset Normal/Anxious windows, % Anxious
  fig_dataset_identity  identity accuracy with fold-wise error bar, recall of each dataset
  fig_pooled_vs_lodo    pooled (15 runs) + dataset prior + LODO 95% CI
  fig_lodo_confusion    3-seed average predictions (threshold 0.5)
  fig_state_only        step 16: 3 seeds, 95% cluster CI
  fig_spectra           step 13 figure in the same style (overwritten)

Required inputs: outputs/02_corpus_v2, 04_pooled/runs.csv, 05_lodo/(lodo_ci.csv, preds/),
                 16_state_only/ci.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import OUT_DIR, out

O = out("15_figures")
R = lambda *p: OUT_DIR.joinpath(*p)
# same palette and style as 15_figures.py
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DCOL = {"BIDS": C[0], "DASPS": C[1], "Mendeley": C[2], "MentalArithmetic": C[3], "SAM40": C[4]}
DNAME = {"BIDS": "BIDS rest", "DASPS": "DASPS", "Mendeley": "Mendeley", "MentalArithmetic": "Mental Arith.", "SAM40": "SAM 40"}
ORDER = ["DASPS", "SAM40", "MentalArithmetic", "Mendeley", "BIDS"]
NORMAL, ANX = "#9aa7b8", "#e34948"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "DejaVu Sans", "lines.linewidth": 2})
SESOI = 0.60


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(O / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  saved", name)


def corpus_dir():
    import os
    return Path(os.environ.get("REV_CORPUS_DIR", OUT_DIR / "02_corpus_v2"))


meta_all = pd.read_csv(corpus_dir() / "meta.csv", low_memory=False)
prim = np.where(meta_all.in_primary.values)[0]
meta = meta_all.iloc[prim].reset_index(drop=True)
ds, y = meta.dataset.values, meta.y.values.astype(int)
print(f"primary corpus: {len(meta)} windows")

# ------------------------------------------------------------------ Fig 1: label balance
try:
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for i, d in enumerate(ORDER):
        m = ds == d; n0, n1 = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        ax.bar(i, n0, color=NORMAL, width=0.62, label="Normal" if i == 0 else None)
        ax.bar(i, n1, bottom=n0, color=ANX, width=0.62, label="Anxious" if i == 0 else None)
        ax.text(i, n0 + n1 + 90, f"{100 * n1 / (n0 + n1):.0f}% Anxious\nn = {n0 + n1:,}", ha="center",
                va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([DNAME[d] for d in ORDER])
    ax.set_ylabel("4-s windows"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max((ds == d).sum() for d in ORDER) * 1.28)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    save(fig, "fig_label_balance")
except Exception as e:
    print("label balance skipped:", e)

# ------------------------------------------------------------------ Fig 2: dataset identity
try:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from rev.common import load_v2, bandpower
    Xz, mz = load_v2(primary_only=True)
    F = bandpower(Xz)
    lab, names = pd.factorize(mz.dataset.values); pid = mz.participant.values
    accs, pred = [], np.empty(len(lab), dtype=int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, lab, pid):   # same as Step 9
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[a], lab[a])
        pred[b] = clf.predict(F[b]); accs.append((pred[b] == lab[b]).mean())
    accs = np.array(accs); maj = np.bincount(lab).max() / len(lab)
    pd.DataFrame(dict(fold=range(1, 6), accuracy=accs)).to_csv(O / "fig_dataset_identity_folds.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1.6]))
    a1.bar([0, 1], [maj, accs.mean()], color=[NORMAL, C[0]], width=0.6)
    a1.errorbar(1, accs.mean(), yerr=accs.std(ddof=1), color=INK, capsize=4, lw=1.2)
    a1.scatter(1 + np.linspace(-0.12, 0.12, len(accs)), accs, s=10, color=INK, zorder=3)
    a1.text(0, maj + 0.03, f"{100 * maj:.1f}%", ha="center", fontsize=7.5, color=INK)
    a1.text(1, accs.max() + 0.04, f"{100 * accs.mean():.1f}% ± {100 * accs.std(ddof=1):.1f}", ha="center",
            fontsize=7.5, color=INK)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Majority\nclass", "Band-power\nLR"]); a1.set_ylim(0, 1.08)
    a1.set_ylabel("Dataset-identity accuracy"); a1.grid(axis="x", visible=False)
    a1.set_title("a  Participant-wise 5-fold CV", fontsize=8, loc="left")
    rec = [(pred[lab == k] == k).mean() for k in range(len(names))]
    o = np.argsort(rec)
    a2.barh(range(len(names)), np.array(rec)[o], color=[DCOL[names[k]] for k in o], height=0.6)
    for j, k in enumerate(o):
        a2.text(rec[k] + 0.01, j, f"{100 * rec[k]:.0f}%", va="center", fontsize=7, color=INK)
    a2.set_yticks(range(len(names))); a2.set_yticklabels([DNAME[names[k]] for k in o]); a2.set_xlim(0, 1.1)
    a2.set_xlabel("Recall (held-out participants)"); a2.grid(axis="y", visible=False)
    a2.set_title("b  Recall of each dataset", fontsize=8, loc="left")
    save(fig, "fig_dataset_identity")
    print(f"    identity acc = {accs.mean():.3f} ± {accs.std(ddof=1):.3f} (majority {maj:.3f})")
except Exception as e:
    print("dataset identity skipped:", e)

# ------------------------------------------------------------------ Fig 3: pooled vs LODO
try:
    r4 = pd.read_csv(R("04_pooled", "runs.csv"))
    lc = pd.read_csv(R("05_lodo", "lodo_ci.csv")).set_index("held_out")
    rows = [("Pooled\n(15 runs)", r4.auc_pooled.values, C[0]),
            ("Dataset prior\nonly (no EEG)", r4.auc_prior.values, NORMAL)]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, (lab_, v, col) in enumerate(rows):
        ax.bar(i, v.mean(), color=col, width=0.6)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=INK, capsize=4, lw=1.2)
        ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=8, color=INK, alpha=0.5, zorder=3)
        ax.text(i, v.max() + 0.025, f"{v.mean():.3f} ± {v.std(ddof=1):.3f}", ha="center", fontsize=7, color=INK)
    m = lc.loc["MEAN_4"]
    ax.bar(2, m.auc, color=C[1], width=0.6)
    ax.errorbar(2, m.auc, yerr=[[m.auc - m.ci_lo], [m.ci_hi - m.auc]], color=INK, capsize=4, lw=1.2)
    four = [d for d in ["DASPS", "SAM40", "MentalArithmetic", "Mendeley"] if d in lc.index]
    for j, d in enumerate(four):
        ax.scatter(2 - 0.15 + 0.1 * j, lc.loc[d, "auc"], s=18, color=DCOL[d], edgecolor=INK, lw=0.6, zorder=3,
                   label=f"{DNAME[d]} ({lc.loc[d, 'auc']:.3f})")
    ax.text(2, 0.32, f"{m.auc:.3f}\n[{m.ci_lo:.3f}, {m.ci_hi:.3f}]", ha="center", va="bottom", fontsize=7,
            color="white", fontweight="bold")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, zorder=0)
    ax.axhline(SESOI, color=C[7], ls=":", lw=1.2, zorder=0)
    ax.text(1.5, SESOI + 0.005, "SESOI 0.60", color=C[7], fontsize=6.5, ha="center", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows] + ["Leave-one-dataset-out\n(mean of 4, 95% CI)"])
    ax.set_ylim(0.3, 0.9); ax.set_ylabel("AUC"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=6.3, ncol=1, handletextpad=0.2, title="LODO, held-out dataset",
              title_fontsize=6.3)
    save(fig, "fig_pooled_vs_lodo")
except Exception as e:
    print("pooled vs LODO skipped:", e)

# ------------------------------------------------------------------ Fig 4: LODO confusion matrices
try:
    P = R("05_lodo", "preds")
    fig, axes = plt.subplots(1, 5, figsize=(8.0, 2.2))
    rows = []
    for ax, d in zip(axes, ORDER):
        te = np.where(ds == d)[0]
        ps = [np.load(P / f"lodo_{d}_seed{s}.npz")["p"] for s in range(3)]
        assert all(len(p) == len(te) for p in ps), f"{d}: prediction length != corpus windows"
        p = np.mean(ps, axis=0); yt = y[te]; yh = (p >= 0.5).astype(int)
        cm = np.array([[((yt == a) & (yh == b)).sum() for b in (0, 1)] for a in (0, 1)])
        rowsum = cm.sum(1, keepdims=True); pct = np.divide(cm, rowsum, out=np.zeros_like(cm, float), where=rowsum > 0)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for a in (0, 1):
            for b in (0, 1):
                if rowsum[a, 0] == 0:
                    ax.text(b, a, "—", ha="center", va="center", fontsize=7, color=MUTED); continue
                ax.text(b, a, f"{100 * pct[a, b]:.0f}%\n({cm[a, b]:,})", ha="center", va="center", fontsize=6.3,
                        color="white" if pct[a, b] > 0.6 else INK)
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) == 2 else np.nan
        sub = f"AUC {auc:.3f}" if np.isfinite(auc) else "one class only"
        ax.set_title(f"{DNAME[d]}\n{sub}\n{100 * yh.mean():.0f}% predicted Anxious", fontsize=6.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["N", "A"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["N", "A"])
        ax.grid(False); ax.set_xlabel("Predicted", fontsize=7)
        if d == ORDER[0]: ax.set_ylabel("True", fontsize=7)
        rows.append(dict(held_out=d, auc=auc, pct_pred_anxious=100 * yh.mean(), pred_sd=float(p.std()),
                         tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1]))
    pd.DataFrame(rows).to_csv(O / "fig_lodo_confusion_values.csv", index=False)
    save(fig, "fig_lodo_confusion")
except Exception as e:
    print("LODO confusion skipped:", e)

# ------------------------------------------------------------------ Fig 9: state-only LODO
try:
    c = pd.read_csv(R("16_state_only", "ci.csv"))
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for i, r in enumerate(c.itertuples()):
        ax.bar(i, r.auc, color=DCOL[r.held_out], width=0.58)
        ax.errorbar(i, r.auc, yerr=[[r.auc - r.ci_lo], [r.ci_hi - r.auc]], color=INK, capsize=4, lw=1.2)
        ax.text(i, 0.315, f"{r.auc:.3f}\n[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", ha="center", va="bottom", fontsize=6.0,
                color="white", fontweight="bold")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, zorder=0)
    ax.axhline(SESOI, color=C[7], ls=":", lw=1.2, zorder=0)
    ax.text(-0.42, SESOI + 0.005, "SESOI 0.60", color=C[7], fontsize=6.5, ha="left", va="bottom")
    ax.set_xticks(range(len(c))); ax.set_xticklabels([f"{DNAME[h]}\nheld out" for h in c.held_out])
    ax.set_ylim(0.3, 0.75); ax.set_ylabel("AUC (3 seeds, 95% cluster CI)"); ax.grid(axis="x", visible=False)
    save(fig, "fig_state_only")
except Exception as e:
    print("state-only skipped:", e)

# ------------------------------------------------------------------ spectra (Step 13 figure, same style)
try:
    from scipy.signal import welch
    from rev.common import load_v2
    Xr, mr = load_v2(primary_only=True, raw=True)
    pid_r, ds_r = mr.participant.values, mr.dataset.values
    S = []
    for i in range(0, len(Xr), 2000):
        f, p = welch(np.asarray(Xr[i:i + 2000], dtype=np.float64), fs=128, nperseg=256, axis=2)
        S.append(np.log10(p.mean(1) + 1e-12))                  # mean over 10 channels
    S = np.concatenate(S); fm = (f >= 1) & (f <= 45)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for d in ["BIDS", "DASPS", "Mendeley", "MentalArithmetic", "SAM40"]:
        parts = np.unique(pid_r[ds_r == d])
        M = np.array([S[pid_r == g].mean(0) for g in parts])[:, fm]
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.plot(f[fm], mu, color=DCOL[d], lw=1.6, label=f"{DNAME[d]} (n = {len(parts)})")
        ax.fill_between(f[fm], mu - 1.96 * se, mu + 1.96 * se, color=DCOL[d], alpha=0.18, lw=0)
    ax.axvspan(8, 13, color=GRID, alpha=0.6, zorder=0); ax.text(10.5, ax.get_ylim()[1], "alpha", ha="center",
                                                                 va="top", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("log$_{10}$ power (µV$^2$/Hz)"); ax.set_xlim(1, 45)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "fig_spectra")
except Exception as e:
    print("spectra skipped:", e)

print(f"Saved -> {O}")
