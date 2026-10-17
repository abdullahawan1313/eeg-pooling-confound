# 15_figures.py
"""
Step 15: figures from the saved results (outputs/03..13). No training.
CPU, ~2-3 min (t-SNE is the slowest).

Figures (outputs/15_figures/, 300 dpi PNG + PDF):
  fig_pooled_decomposition : pooled AUC vs dataset prior vs no-BIDS vs balanced
  fig_within_vs_lodo       : per dataset, within (participant-wise) vs LODO, 95% CI
  fig_architectures        : 7 models, within vs LODO (SAM40, MA)
  fig_positive_control     : arithmetic-vs-rest transfer vs anxiety-label LODO
  fig_adversarial          : lambda sweep: target/source AUC, identity probe, loss curves
  fig_robustness           : LODO in every variant (EA, normalisation, task labels,
                             architectures) + SESOI
  fig_tsne_heldout         : t-SNE of HELD-OUT participants only
(fig_spectra is produced by step 13.)
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import OUT_DIR, out

O = out("15_figures")
R = lambda *p: OUT_DIR.joinpath(*p)
# validated categorical palette (fixed order; dataset identity always the same colour)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DCOL = {"BIDS": C[0], "DASPS": C[1], "Mendeley": C[2], "MentalArithmetic": C[3], "SAM40": C[4]}
DNAME = {"BIDS": "BIDS rest", "DASPS": "DASPS", "Mendeley": "Mendeley", "MentalArithmetic": "Mental Arith.", "SAM40": "SAM 40"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "DejaVu Sans", "lines.linewidth": 2})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(O / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  saved", name)


def chance(ax, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(0.5, color=MUTED, ls="--", lw=1, zorder=0)


def first(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ 1. pooled decomposition
try:
    r4 = pd.read_csv(R("04_pooled", "runs.csv"))
    b = pd.read_csv(R("11_balanced", "runs.csv"))
    rows = [("Pooled, all 5 datasets", r4.auc_pooled), ("Dataset prior only (no EEG)", r4.auc_prior),
            ("Pooled, BIDS removed", r4.auc_pooled_noBIDS),
            ("BIDS removed + class-balanced", b[b.condition == "C_noBIDS_balanced"].auc),
            ("BIDS removed + size-matched", b[b.condition == "D_noBIDS_sizematched"].auc),
            ("Within-dataset share of pooled", r4.auc_within_weighted)]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, (lab, v) in enumerate(rows):
        yj = i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        ax.scatter(v, yj, s=14, color=C[0], alpha=0.45, edgecolor="white", linewidth=0.5)
        ax.scatter([v.mean()], [i], s=60, color=INK, marker="D", zorder=3)
        ax.text(max(v.max(), v.mean()) + 0.012, i, f"mean {v.mean():.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    chance(ax); ax.set_xlabel("Held-out-participant AUC"); ax.set_xlim(0.3, 0.85)
    save(fig, "fig_pooled_decomposition")
except Exception as e:
    print("pooled decomposition skipped:", e)

# ------------------------------------------------------------------ 2. within vs LODO (CNN-Transformer)
try:
    wf = first(R("03_within", "within_dataset_auc.csv"), R("03_within_cpu", "within_dataset_auc.csv"))
    w = pd.read_csv(wf).query("model == 'cnn_transformer'").groupby("dataset")[["auc", "ci_lo", "ci_hi"]].mean()
    l = pd.read_csv(R("05_lodo", "lodo_ci.csv")).set_index("held_out")
    order = ["SAM40", "MentalArithmetic", "Mendeley", "DASPS"]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, d in enumerate(order):
        for off, src, col, lab in [(-0.13, w, C[0], "Within dataset (new participants)"),
                                   (0.13, l, C[1], "Leave-one-dataset-out")]:
            a, lo, hi = src.loc[d, "auc"], src.loc[d, "ci_lo"], src.loc[d, "ci_hi"]
            ax.errorbar(a, i + off, xerr=[[a - lo], [hi - a]], fmt="o", color=col, ms=6, capsize=0,
                        elinewidth=2, label=lab if i == 0 else None)
        ax.plot([w.loc[d, "auc"], l.loc[d, "auc"]], [i - 0.13, i + 0.13], color=GRID, lw=1, zorder=0)
    ax.axvline(0.60, color=C[7], ls=":", lw=1.2); ax.text(0.605, -0.62, "SESOI 0.60", color=C[7], fontsize=6.5, va="center")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([DNAME[d] for d in order]); ax.set_ylim(3.5, -0.8)
    chance(ax); ax.set_xlabel("AUC (95% participant-cluster bootstrap CI)"); ax.set_xlim(0.35, 0.85)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, "fig_within_vs_lodo")
except Exception as e:
    print("within vs LODO skipped:", e)

# ------------------------------------------------------------------ 3. architectures
try:
    a = pd.read_csv(R("06_arch", "all_runs.csv"))
    # CNN-Transformer within-dataset: use the main Step 3 run (GPU, 3 seeds) so that all figures agree
    w3 = R("03_within", "within_dataset_auc.csv")
    if w3.exists():
        w3 = pd.read_csv(w3).query("model == 'cnn_transformer'").groupby("dataset").auc.mean()
        a = a[~((a.model == "cnn_transformer") & (a.protocol == "within"))]
        a = pd.concat([a, pd.DataFrame(dict(model="cnn_transformer", protocol="within",
                                            dataset=w3.index, auc=w3.values))], ignore_index=True)
    models = ["cnn_transformer", "pure_cnn", "pure_transformer", "eegnet", "shallow_convnet", "bandpower_LR", "bandpower_RF"]
    mname = {"cnn_transformer": "CNN–Transformer", "pure_cnn": "CNN only", "pure_transformer": "Transformer only",
             "eegnet": "EEGNet", "shallow_convnet": "ShallowConvNet", "bandpower_LR": "Band-power + LR",
             "bandpower_RF": "Band-power + RF"}
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    for ax, d in zip(axs, ["SAM40", "MentalArithmetic"]):
        for i, m in enumerate(models):
            g = a[(a.model == m) & (a.dataset == d)]
            wv, lv = g[g.protocol == "within"].auc.mean(), g[g.protocol == "lodo"].auc.mean()
            ax.plot([wv, lv], [i, i], color=GRID, lw=1.5, zorder=0)
            ax.scatter(wv, i, color=C[0], s=36, zorder=3, label="Within dataset" if i == 0 else None)
            ax.scatter(lv, i, color=C[1], s=36, zorder=3, label="Leave-one-dataset-out" if i == 0 else None)
        chance(ax); ax.set_title(DNAME[d], fontsize=8, loc="left"); ax.set_xlim(0.38, 0.8); ax.set_xlabel("AUC")
    axs[0].set_yticks(range(len(models))); axs[0].set_yticklabels([mname[m] for m in models]); axs[0].invert_yaxis()
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.6, 1.06))
    save(fig, "fig_architectures")
except Exception as e:
    print("architectures skipped:", e)

# ------------------------------------------------------------------ 4. positive control
try:
    pc = pd.read_csv(R("07_positive_control", "results.csv")).query("model == 'cnn_transformer'")
    l = pd.read_csv(R("05_lodo", "lodo_ci.csv")).set_index("held_out")
    items = [(r.experiment.replace("SAM40_arith", "SAM40").replace("SAM40_anytask", "SAM40 (any task)")
              .replace("within ", "Within ").replace("transfer ", "Transfer "), r.auc, r.ci_lo, r.ci_hi, C[0])
             for r in pc.itertuples()]
    items += [("Anxiety labels, LODO → Mental Arith.", *l.loc["MentalArithmetic", ["auc", "ci_lo", "ci_hi"]], C[1]),
              ("Anxiety labels, LODO → SAM 40", *l.loc["SAM40", ["auc", "ci_lo", "ci_hi"]], C[1])]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for i, (lab, au, lo, hi, col) in enumerate(items):
        ax.errorbar(au, i, xerr=[[au - lo], [hi - au]], fmt="o", color=col, ms=6, elinewidth=2)
        ax.text(hi + 0.01, i, f"{au:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items]); ax.invert_yaxis()
    chance(ax); ax.set_xlim(0.38, 0.85); ax.set_xlabel("AUC (95% CI)")
    save(fig, "fig_positive_control")
except Exception as e:
    print("positive control skipped:", e)

# ------------------------------------------------------------------ 5. adversarial
try:
    ad = pd.read_csv(R("08_adversarial", "runs.csv"))
    lg = pd.read_csv(R("08_adversarial", "train_logs.csv"))
    s = ad.groupby("lambda_max").agg(t=("target_auc", "mean"), v=("srcval_auc", "mean"), i=("id_probe", "mean"),
                                     m=("id_majority", "mean"), c=("collapsed", "sum")).reset_index()
    xs = np.arange(len(s)); xl = [f"{x:g}" for x in s.lambda_max]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axs[0]
    ax.plot(xs, s.v, "-o", color=C[0], label="Source held-out"); ax.plot(xs, s.t, "-o", color=C[1], label="Target dataset")
    chance(ax, True); ax.set_xticks(xs); ax.set_xticklabels(xl); ax.set_xlabel("λ max"); ax.set_ylabel("Anxiety AUC")
    ax.set_ylim(0.4, 0.8); ax.legend(fontsize=7); ax.set_title("a  Task signal", fontsize=8, loc="left")
    for x, c in zip(xs, s.c):
        if c: ax.text(x, 0.415, f"{int(c)}/8", ha="center", fontsize=6.5, color=C[7])
    ax.text(-0.3, 0.44, "collapsed runs:", fontsize=6, color=C[7])
    ax = axs[1]
    ax.plot(xs, s.i, "-o", color=C[6], label="Identity probe"); ax.plot(xs, s.m, "--", color=MUTED, lw=1, label="Majority baseline")
    ax.set_xticks(xs); ax.set_xticklabels(xl); ax.set_xlabel("λ max"); ax.set_ylabel("Dataset-identity accuracy")
    ax.set_ylim(0.3, 1.0); ax.legend(fontsize=7); ax.set_title("b  Dataset identity (same probe)", fontsize=8, loc="left")
    ax = axs[2]
    lg["lam"] = lg.tag.str.extract(r"lam([\d.]+)_").astype(float)
    for lam, col in [(0.0, C[0]), (0.3, C[3]), (1.0, C[1]), (3.0, C[7])]:
        g = lg[lg.lam == lam].groupby("epoch").loss_anx.mean()
        ax.plot(g.index, g.values, color=col, label=f"λ={lam:g}")
    ax.axhline(np.log(2), color=MUTED, ls="--", lw=1); ax.text(1, 0.35, "dashed: ln 2 = chance-level loss", fontsize=6, color=MUTED)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Anxiety loss (train)"); ax.legend(fontsize=6)
    ax.set_title("c  Training dynamics", fontsize=8, loc="left")
    save(fig, "fig_adversarial")
except Exception as e:
    print("adversarial skipped:", e)

# ------------------------------------------------------------------ 6. robustness of LODO
try:
    rows = []
    l5 = pd.read_csv(R("05_lodo", "per_seed.csv")).dropna(subset=["auc"])
    rows.append(("Main pipeline (CNN–Transformer)", l5.groupby("held_out").auc.mean()))
    a = pd.read_csv(R("06_arch", "all_runs.csv"))
    for m, n in [("eegnet", "EEGNet"), ("shallow_convnet", "ShallowConvNet"), ("pure_cnn", "CNN only"),
                 ("pure_transformer", "Transformer only"), ("bandpower_LR", "Band-power + LR"), ("bandpower_RF", "Band-power + RF")]:
        rows.append((n, a[(a.model == m) & (a.protocol == "lodo")].groupby("dataset").auc.mean()))
    for f, v, n in [("09_ea", "EA", "Euclidean alignment (transductive)"),
                    ("10_norm", "window_global", "Ratio-preserving scaling"),
                    ("10_norm", "fixed_scale", "Absolute-amplitude scaling"),
                    ("12_sam40_task", "SAM40_taskbased", "SAM 40 task-based labels")]:
        rr = pd.read_csv(R(f, "runs.csv"))
        rows.append((n, rr[(rr.variant == v) & (rr.protocol == "lodo")].groupby("dataset").auc.mean()))
    ad = pd.read_csv(R("08_adversarial", "runs.csv"))
    best = ad[ad.lambda_max == 0.1].groupby("held_out").target_auc.mean()
    rows.append(("Dataset-adversarial (λ=0.1)", best))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (n, ser) in enumerate(rows):
        for d, v in ser.items():
            if d in DCOL:
                ax.scatter(v, i, s=18, color=DCOL[d], alpha=0.9, edgecolor="white", linewidth=0.5,
                           label=DNAME[d] if i == 0 else None)
        ax.scatter(ser.mean(), i, marker="|", s=160, color=INK, linewidth=2)
    ax.axvline(0.60, color=C[7], ls=":", lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    chance(ax); ax.set_xlabel("Leave-one-dataset-out AUC (│ = mean of 4)"); ax.set_xlim(0.35, 0.75)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=6.5, ncol=4)
    save(fig, "fig_robustness")
except Exception as e:
    print("robustness skipped:", e)

# ------------------------------------------------------------------ 7. held-out t-SNE
try:
    from sklearn.manifold import TSNE
    meta = pd.read_csv(first(Path(__import__("os").environ.get("REV_CORPUS_DIR", "_")) / "meta.csv",
                             R("02_corpus_v2", "meta.csv")), low_memory=False)
    meta = meta[meta.in_primary].reset_index(drop=True)
    z = np.load(R("04_pooled", "preds", "pooled_split42_seed0.npz"))
    te, F = z["te"], z["feat"]
    E = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F)
    yt, dt = meta.y.values[te], meta.dataset.values[te]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.1))
    for k, col, lab in [(0, "#8a8a85", "Normal"), (1, C[6], "Anxious")]:
        m = yt == k; axs[0].scatter(E[m, 0], E[m, 1], s=4, color=col, alpha=0.6, label=lab, linewidths=0)
    for d in DCOL:
        m = dt == d; axs[1].scatter(E[m, 0], E[m, 1], s=4, color=DCOL[d], alpha=0.6, label=DNAME[d], linewidths=0)
    for ax, t in zip(axs, ["a  Coloured by label", "b  Coloured by dataset"]):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(t, fontsize=8, loc="left")
        ax.legend(markerscale=3, fontsize=6.5, loc="best")
    save(fig, "fig_tsne_heldout")
except Exception as e:
    print("t-SNE skipped:", e)
print("done ->", O)
