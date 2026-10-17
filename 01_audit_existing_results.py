# 01_audit_existing_results.py
"""
Step 01 (optional): audit of the earlier single-split analysis. No training (1-2 min, CPU).
Needs the corpus (corpus.npz) and the saved predictions of the earlier analysis, which are not
part of this repository (set OLD_PROJECT_DIR, see rev/paths.py). No other step depends on it.

Checks:
  A. Dataset composition and windows per participant
  B. Dataset-prior baseline: whole-corpus composition and the actual 80/20 participant
     split (seed 42)
  C. LODO AUC per dataset with participant-cluster bootstrap 95% CI, CI of the mean of 4,
     and the SESOI (AUC 0.60) check
  D. Whether the earlier adversarial models produced degenerate (constant) predictions
  E. Fraction of the earlier t-SNE points that came from training participants
  F. Percentage predicted Anxious per held-out dataset
  G. Demographics available in the dataset files

Run:    python 01_audit_existing_results.py
Output: outputs/01_audit/*.csv and outputs/01_audit/AUDIT_SUMMARY.txt
"""
import sys, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev.paths import OLD_CORPUS, OLD_CACHE, META, out

OUT = out("01_audit")
SEED = 42            # random seed of the earlier analysis
N_BOOT = 2000
SESOI = 0.60         # smallest effect size of interest (AUC)
BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]

log_buf = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); log_buf.write(s + "\n")


# ---------------------------------------------------------------- helpers
def cluster_boot_auc(y, p, groups, n_boot=N_BOOT, seed=0):
    """Participant-cluster bootstrap: participants resample (with replacement)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        s = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in s])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    vals = np.array(vals)
    return np.percentile(vals, [2.5, 97.5]), vals


def degeneracy(p):
    return dict(std=float(np.std(p)), n_unique=int(len(np.unique(np.round(p, 6)))),
                pmin=float(p.min()), pmax=float(p.max()),
                frac_pred_anx=float((p >= 0.5).mean()))


# ---------------------------------------------------------------- load
log("=" * 70); log(" STEP 1: AUDIT OF THE EARLIER ANALYSIS"); log("=" * 70)
d = np.load(OLD_CORPUS, allow_pickle=True)          # X is not loaded (saves memory)
y, dataset, participant = d["y"].astype(int), d["dataset"].astype(str), d["participant"].astype(str)
N = len(y)
log(f"corpus windows = {N}")

# ---------------------------------------------------------------- A
log("\n[A] Table 1 re-check + windows per participant")
rows = []
for ds in sorted(set(dataset)):
    m = dataset == ds
    pids = participant[m]
    per = pd.Series(pids).value_counts()
    # participant-level label (Mendeley/BIDS: constant per participant)
    pl = pd.DataFrame({"p": pids, "y": y[m]}).groupby("p").y.mean()
    rows.append(dict(dataset=ds, n_participants=len(per), windows=int(m.sum()),
                     pct_anxious=round(100 * y[m].mean(), 1),
                     win_per_part_min=int(per.min()), win_per_part_median=float(per.median()),
                     win_per_part_max=int(per.max()),
                     participants_all_anxious=int((pl == 1).sum()),
                     participants_all_normal=int((pl == 0).sum()),
                     participants_mixed=int(((pl > 0) & (pl < 1)).sum())))
tA = pd.DataFrame(rows); tA.to_csv(OUT / "A_table1_recheck.csv", index=False)
log(tA.to_string(index=False))

# ---------------------------------------------------------------- B
log("\n[B] Dataset-PRIOR baseline (score = anxious prevalence of the dataset)")
prev_all = {ds: y[dataset == ds].mean() for ds in set(dataset)}
s_all = np.array([prev_all[x] for x in dataset])
auc_b1 = roc_auc_score(y, s_all)
nb = dataset != "BIDS"
auc_b1_nobids = roc_auc_score(y[nb], s_all[nb])
log(f"  B1 whole corpus (Table-1 composition): AUC = {auc_b1:.3f}  | without BIDS: {auc_b1_nobids:.3f}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(np.zeros(N), y, groups=participant))
prev_tr = {ds: (y[tr][dataset[tr] == ds].mean() if (dataset[tr] == ds).any() else y[tr].mean())
           for ds in set(dataset)}
s_te = np.array([prev_tr[x] for x in dataset[te]])
auc_b2 = roc_auc_score(y[te], s_te)
ci_b2, _ = cluster_boot_auc(y[te], s_te, participant[te])
m_te_nb = dataset[te] != "BIDS"
auc_b2_nb = roc_auc_score(y[te][m_te_nb], s_te[m_te_nb])
log(f"  B2 actual 80/20 split (seed {SEED}), prevalence from TRAIN participants:")
log(f"     prior-only test AUC = {auc_b2:.3f}  95% cluster CI [{ci_b2[0]:.3f}, {ci_b2[1]:.3f}]"
    f"  | without BIDS: {auc_b2_nb:.3f}")
log(f"     (earlier pooled CNN-Transformer AUC = 0.770 on this same split)")
comp = []
for ds in sorted(set(dataset)):
    mt = dataset[te] == ds
    comp.append(dict(dataset=ds, test_participants=len(set(participant[te][mt])),
                     test_windows=int(mt.sum()), test_pct_anx=round(100 * y[te][mt].mean(), 1),
                     train_prevalence=round(prev_tr[ds], 3)))
tB = pd.DataFrame(comp); tB.to_csv(OUT / "B_pooled_split_composition.csv", index=False)
log(tB.to_string(index=False))
pd.DataFrame([dict(B1_prior_auc=auc_b1, B1_prior_auc_noBIDS=auc_b1_nobids,
                   B2_prior_auc_actual_split=auc_b2, B2_ci_lo=ci_b2[0], B2_ci_hi=ci_b2[1],
                   B2_prior_auc_noBIDS=auc_b2_nb)]).to_csv(OUT / "B_prior_baseline.csv", index=False)

# ---------------------------------------------------------------- C + D + F
def audit_preds(fname, tag):
    log(f"\n[C/D/F] {tag}  ({fname})")
    c = np.load(OLD_CACHE / fname, allow_pickle=True)
    rows, boots = [], {}
    for ds in sorted(set(dataset)):
        p, yy = c[f"{ds}__p"], c[f"{ds}__y"].astype(int)
        g = participant[dataset == ds]
        assert len(g) == len(p) and np.array_equal(yy, y[dataset == ds]), f"order mismatch {ds}"
        dg = degeneracy(p)
        r = dict(model=tag, held_out=ds, n_windows=len(p), n_participants=len(set(g)), **dg)
        if ds in BOTH:
            auc = roc_auc_score(yy, p)
            ci, bv = cluster_boot_auc(yy, p, g)
            boots[ds] = bv
            r.update(auc=round(auc, 4), ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4),
                     ci_excludes_0p5=bool(ci[0] > 0.5 or ci[1] < 0.5),
                     ci_upper_below_SESOI=bool(ci[1] < SESOI))
        rows.append(r)
        flag = "  <-- DEGENERATE (almost constant output)" if dg["std"] < 0.01 else ""
        a = f"AUC={r['auc']:.3f} CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]" if "auc" in r else "single-class"
        log(f"  {ds:<17} {a:<30} std={dg['std']:.4f} unique={dg['n_unique']:<5} "
            f"%predAnx={100*dg['frac_pred_anx']:.0f}{flag}")
    # mean of 4 with bootstrap (independent per dataset)
    k = min(len(v) for v in boots.values())
    mean_boot = np.mean([boots[ds][:k] for ds in BOTH], axis=0)
    mean_auc = np.mean([r["auc"] for r in rows if "auc" in r])
    lo, hi = np.percentile(mean_boot, [2.5, 97.5])
    log(f"  MEAN(4)           AUC={mean_auc:.3f} CI[{lo:.3f},{hi:.3f}]  "
        f"P(boot mean >= SESOI {SESOI}) = {np.mean(mean_boot >= SESOI):.3f}")
    rows.append(dict(model=tag, held_out="MEAN_bothclass", auc=round(mean_auc, 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    return rows

allrows = audit_preds("lodo_preds.npz", "LODO_standard")
allrows += audit_preds("lodo_adv_preds.npz", "LODO_adversarial_lambda1")
tC = pd.DataFrame(allrows); tC.to_csv(OUT / "C_lodo_auc_ci_degeneracy.csv", index=False)

# ---------------------------------------------------------------- E
log("\n[E] Earlier t-SNE sample: fraction of points from TRAINING participants")
idx = np.random.default_rng(SEED).permutation(N)[:3000]      # same sampling as the earlier analysis
tr_set = set(participant[tr])
frac_train = np.mean([participant[i] in tr_set for i in idx])
log(f"  {100*frac_train:.1f}% of the 3000 t-SNE points came from training participants "
    f"(only {100*(1-frac_train):.1f}% held-out).")
f = np.load(OLD_CACHE / "pooled_feats.npz", allow_pickle=True)
same = np.array_equal(f["y"].astype(int), y[idx])
log(f"  labels in the cache match this reconstruction: {same}")

# ---------------------------------------------------------------- G
log("\n[G] Demographics (available in the dataset files)")
demo = []
try:
    ma = pd.read_csv(META["MA_subject_info"])
    demo.append(dict(dataset="MentalArithmetic", n=len(ma), age_mean=round(ma.Age.mean(), 1),
                     age_sd=round(ma.Age.std(), 1), age_min=ma.Age.min(), age_max=ma.Age.max(),
                     female=int((ma.Gender == "F").sum()), male=int((ma.Gender == "M").sum()),
                     source="subject-info.csv"))
except Exception as e:
    log("  MA meta error:", e)
try:
    b = pd.read_csv(META["BIDS_participants"], sep="\t")
    used = sorted({p.replace("BIDS_s", "sub-") for p in set(participant[dataset == "BIDS"])})
    bu = b[b.participant_id.isin(used)]
    demo.append(dict(dataset="BIDS (used subset)", n=len(bu), age_mean=round(bu.age.mean(), 1),
                     age_sd=round(bu.age.std(), 1), age_min=bu.age.min(), age_max=bu.age.max(),
                     female=int((bu.sex == "f").sum()), male=int((bu.sex == "m").sum()),
                     source="participants.tsv"))
    log(f"  BIDS used IDs: {used[0]} .. {used[-1]} ({len(used)})")
except Exception as e:
    log("  BIDS meta error:", e)
try:
    r = pd.read_excel(META["DASPS_ratings"], header=None).iloc[1:, [1, 5, 6]].dropna()
    r.columns = ["id", "hamA_pre", "hamA_post"]
    r["hamA_pre_score"] = r.hamA_pre.astype(str).str.extract(r"(\d+)").astype(float)
    r["hamA_pre_level"] = r.hamA_pre.astype(str).str.split(":").str[-1].str.strip()
    r.to_csv(OUT / "G_dasps_hamA.csv", index=False)
    log(f"  DASPS HAM-A (pre): mean={r.hamA_pre_score.mean():.1f}, levels="
        f"{r.hamA_pre_level.value_counts().to_dict()}")
except Exception as e:
    log("  DASPS meta error:", e)
for ds in ["DASPS", "SAM40", "Mendeley"]:
    demo.append(dict(dataset=ds, source="not in the dataset files (taken from the dataset publication)"))
tG = pd.DataFrame(demo); tG.to_csv(OUT / "G_demographics_partial.csv", index=False)
log(tG.to_string(index=False))

(OUT / "AUDIT_SUMMARY.txt").write_text(log_buf.getvalue(), encoding="utf-8")
log(f"\nSaved -> {OUT}")
