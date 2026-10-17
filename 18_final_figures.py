# 18_final_figures.py
"""
Step 18: final numbered figures of the article (9 main + 2 supplementary) in one folder.
Run 15_figures.py and 17_old_figures_redraw.py first. No training. CPU, ~3-5 min.

Built here (merged panels):
  Fig 2 = (a) dataset identity  (b) recall of each dataset  (c) participant-averaged spectra
  Fig 3 = (a) pooled vs dataset prior vs LODO  (b) pooled AUC decomposition
The other figures are copied from outputs/15_figures/ under their final names.

Output: outputs/18_final_figures/  Fig1..Fig9, FigS1, FigS2 (PNG 300 dpi + PDF) + FIGURE_LIST.txt
"""
import sys, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import OUT_DIR, out

O = out("18_final_figures")
F15 = OUT_DIR / "15_figures"
R = lambda *p: OUT_DIR.joinpath(*p)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DCOL = {"BIDS": C[0], "DASPS": C[1], "Mendeley": C[2], "MentalArithmetic": C[3], "SAM40": C[4]}
DNAME = {"BIDS": "BIDS rest", "DASPS": "DASPS", "Mendeley": "Mendeley", "MentalArithmetic": "Mental Arith.", "SAM40": "SAM 40"}
NORMAL = "#9aa7b8"
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


def panel(ax, letter, text):
    ax.set_title(f"{letter}  {text}", fontsize=8, loc="left", color=INK)


# ================================================================== Fig 2 (identity + recall + spectra)
try:
    from scipy.signal import welch
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from rev.common import load_v2, bandpower
    Xz, mz = load_v2(primary_only=True)
    Fb = bandpower(Xz); del Xz
    lab, names = pd.factorize(mz.dataset.values); pid = mz.participant.values
    accs, pred = [], np.empty(len(lab), dtype=int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Fb, lab, pid):   # same as Steps 9/17
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Fb[a], lab[a])
        pred[b] = clf.predict(Fb[b]); accs.append((pred[b] == lab[b]).mean())
    accs = np.array(accs); maj = np.bincount(lab).max() / len(lab)
    rec = np.array([(pred[lab == k] == k).mean() for k in range(len(names))])

    Xr, mr = load_v2(primary_only=True, raw=True)
    S = []
    for i in range(0, len(Xr), 2000):
        f, p = welch(np.asarray(Xr[i:i + 2000], dtype=np.float64), fs=128, nperseg=256, axis=2)
        S.append(np.log10(p.mean(1) + 1e-12))
    S = np.concatenate(S); del Xr; fm = (f >= 1) & (f <= 45)

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[1, 1.15])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    a1.bar([0, 1], [maj, accs.mean()], color=[NORMAL, C[0]], width=0.6)
    a1.errorbar(1, accs.mean(), yerr=accs.std(ddof=1), color=INK, capsize=4, lw=1.2)
    a1.scatter(1 + np.linspace(-0.12, 0.12, len(accs)), accs, s=10, color=INK, zorder=3)
    a1.text(0, maj + 0.03, f"{100 * maj:.1f}%", ha="center", fontsize=7.5, color=INK)
    a1.text(1, accs.max() + 0.04, f"{100 * accs.mean():.1f}% ± {100 * accs.std(ddof=1):.1f}", ha="center",
            fontsize=7.5, color=INK)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Majority\nclass", "Band-power\nLR"]); a1.set_ylim(0, 1.1)
    a1.set_ylabel("Dataset-identity accuracy"); a1.grid(axis="x", visible=False)
    panel(a1, "a", "Dataset identity (5-fold, by participant)")
    o = np.argsort(rec)
    a2.barh(range(len(names)), rec[o], color=[DCOL[names[k]] for k in o], height=0.6)
    for j, k in enumerate(o):
        a2.text(rec[k] + 0.01, j, f"{100 * rec[k]:.0f}%", va="center", fontsize=7, color=INK)
    a2.set_yticks(range(len(names))); a2.set_yticklabels([DNAME[names[k]] for k in o]); a2.set_xlim(0, 1.1)
    a2.set_xlabel("Recall (held-out participants)"); a2.grid(axis="y", visible=False)
    panel(a2, "b", "Recall of each dataset")
    pid_r, ds_r = mr.participant.values, mr.dataset.values
    for d in ["BIDS", "DASPS", "Mendeley", "MentalArithmetic", "SAM40"]:
        parts = np.unique(pid_r[ds_r == d])
        M = np.array([S[pid_r == g].mean(0) for g in parts])[:, fm]
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        a3.plot(f[fm], mu, color=DCOL[d], lw=1.6, label=f"{DNAME[d]} (n = {len(parts)})")
        a3.fill_between(f[fm], mu - 1.96 * se, mu + 1.96 * se, color=DCOL[d], alpha=0.18, lw=0)
    a3.axvspan(8, 13, color=GRID, alpha=0.6, zorder=0)
    a3.text(10.5, a3.get_ylim()[1], "alpha", ha="center", va="top", fontsize=6.5, color=MUTED)
    a3.set_xlabel("Frequency (Hz)"); a3.set_ylabel("log$_{10}$ power (µV$^2$/Hz)"); a3.set_xlim(1, 45)
    a3.legend(fontsize=7, loc="upper right", ncol=2)
    panel(a3, "c", "Participant-averaged spectra (10 common channels, mean ± 95% CI)")
    save(fig, "Fig2_dataset_identity_spectra")
    print(f"    identity {accs.mean():.3f} ± {accs.std(ddof=1):.3f}; recall " +
          ", ".join(f"{names[k]} {rec[k]:.2f}" for k in range(len(names))))
except Exception as e:
    print("Fig 2 skipped:", e)

# ================================================================== Fig 3 (pooled vs LODO + decomposition)
try:
    r4 = pd.read_csv(R("04_pooled", "runs.csv"))
    lc = pd.read_csv(R("05_lodo", "lodo_ci.csv")).set_index("held_out")
    b = pd.read_csv(R("11_balanced", "runs.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1.3]))
    # (a)
    for i, (v, col) in enumerate([(r4.auc_pooled.values, C[0]), (r4.auc_prior.values, NORMAL)]):
        a1.bar(i, v.mean(), color=col, width=0.6)
        a1.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=INK, capsize=4, lw=1.2)
        a1.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=8, color=INK, alpha=0.5, zorder=3)
        a1.text(i, v.max() + 0.02, f"{v.mean():.3f}\n± {v.std(ddof=1):.3f}", ha="center", fontsize=6.8, color=INK)
    m = lc.loc["MEAN_4"]
    a1.bar(2, m.auc, color=C[1], width=0.6)
    a1.errorbar(2, m.auc, yerr=[[m.auc - m.ci_lo], [m.ci_hi - m.auc]], color=INK, capsize=4, lw=1.2)
    for j, d in enumerate(["DASPS", "SAM40", "MentalArithmetic", "Mendeley"]):
        a1.scatter(2 - 0.15 + 0.1 * j, lc.loc[d, "auc"], s=16, color=DCOL[d], edgecolor=INK, lw=0.6, zorder=3,
                   label=f"{DNAME[d]} ({lc.loc[d, 'auc']:.3f})")
    a1.text(2, 0.32, f"{m.auc:.3f}\n[{m.ci_lo:.3f},\n{m.ci_hi:.3f}]", ha="center", va="bottom", fontsize=6.2,
            color="white", fontweight="bold")
    a1.axhline(0.5, color=MUTED, ls="--", lw=1, zorder=0); a1.axhline(SESOI, color=C[7], ls=":", lw=1.2, zorder=0)
    a1.text(1.5, SESOI + 0.005, "SESOI 0.60", color=C[7], fontsize=6.3, ha="center", va="bottom")
    a1.set_xticks(range(3)); a1.set_xticklabels(["Pooled\n(15 runs)", "Dataset\nprior only", "LODO\n(mean of 4)"])
    a1.set_ylim(0.3, 1.0); a1.set_ylabel("AUC"); a1.grid(axis="x", visible=False)
    a1.legend(loc="upper center", fontsize=6, ncol=2, title="LODO, held-out dataset", title_fontsize=6,
              handletextpad=0.2, columnspacing=0.8)
    panel(a1, "a", "Pooled vs leave-one-dataset-out")
    # (b)
    rows = [("Pooled, all 5 datasets", r4.auc_pooled), ("Dataset prior only (no EEG)", r4.auc_prior),
            ("Pooled, BIDS removed", r4.auc_pooled_noBIDS),
            ("BIDS removed + class-balanced", b[b.condition == "C_noBIDS_balanced"].auc),
            ("BIDS removed + size-matched", b[b.condition == "D_noBIDS_sizematched"].auc),
            ("Within-dataset share of pooled", r4.auc_within_weighted)]
    for i, (lab_, v) in enumerate(rows):
        yj = i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        a2.scatter(v, yj, s=12, color=C[0], alpha=0.45, edgecolor="white", linewidth=0.5)
        a2.scatter([v.mean()], [i], s=50, color=INK, marker="D", zorder=3)
        a2.text(max(v.max(), v.mean()) + 0.012, i, f"{v.mean():.3f}", va="center", fontsize=6.8, color=INK)
    a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[0] for r in rows]); a2.invert_yaxis()
    a2.axvline(0.5, color=MUTED, ls="--", lw=1, zorder=0)
    a2.set_xlabel("Held-out-participant AUC (dots = runs, ◆ = mean)"); a2.set_xlim(0.3, 0.85)
    panel(a2, "b", "Where the pooled AUC comes from")
    save(fig, "Fig3_pooled_vs_lodo_decomposition")
except Exception as e:
    print("Fig 3 skipped:", e)

# ================================================================== copy the rest with final names
COPY = [("fig_label_balance", "Fig1_label_balance"),
        ("fig_within_vs_lodo", "Fig4_within_vs_lodo"),
        ("fig_lodo_confusion", "Fig5_lodo_confusion"),
        ("fig_positive_control", "Fig6_positive_control"),
        ("fig_adversarial", "Fig7_adversarial"),
        ("fig_state_only", "Fig8_state_only"),
        ("fig_robustness", "Fig9_robustness"),
        ("fig_architectures", "FigS1_architectures"),
        ("fig_tsne_heldout", "FigS2_tsne_heldout")]
for src, dst in COPY:
    for ext in ("png", "pdf"):
        p = F15 / f"{src}.{ext}"
        if p.exists():
            shutil.copy2(p, O / f"{dst}.{ext}")
        else:
            print(f"  MISSING {p.name} (run steps 15 and 17 first)")
    print("  copied", dst)

(O / "FIGURE_LIST.txt").write_text("""Final figures (main text)
Fig 1  Label balance per dataset
Fig 2  (a) dataset identity (b) per-dataset recall (c) spectra
Fig 3  (a) pooled vs prior vs LODO (b) pooled decomposition
Fig 4  Within-dataset vs LODO, 95% CI
Fig 5  LODO confusion matrices, 3-seed average
Fig 6  Positive control: arithmetic vs rest
Fig 7  Dataset-adversarial training, lambda sweep
Fig 8  State-only LODO, 3 seeds
Fig 9  Robustness: LODO across 12 variants
Supplementary (Online Resource 1)
Fig S1 Architectures, within vs LODO (step 06 runs; within values differ slightly from Fig 4, which uses step 03)
Fig S2 t-SNE, held-out participants only (descriptive)
""", encoding="utf-8")
print(f"Saved -> {O}")
