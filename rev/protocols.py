# rev/protocols.py
"""
Runs the same evaluation set (within-dataset / pooled / LODO) on any input variant.
Used by steps 09-12 so that all variants are compared under an identical protocol.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from rev.common import train_model, predict, cached

BOTH = ["DASPS", "Mendeley", "MentalArithmetic", "SAM40"]


def evaluate(X, y, ds, pid, variant, pred_dir, log=print, within=True, within_datasets=BOTH,
             pooled_splits=(42, 0, 1), lodo_seeds=(0, 1), lodo_holds=None, arch="cnn_transformer"):
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if within:
        for d in within_datasets:
            idx = np.where(ds == d)[0]
            if len(np.unique(y[idx])) < 2:
                continue
            p = np.full(len(idx), np.nan)
            for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                       .split(idx, y[idx], groups=pid[idx])):
                tr, te = idx[a], idx[b]
                p[b] = cached(pred_dir / f"{variant}_within_{d}_f{k}.npz",
                              lambda: {"p": predict(train_model(X[tr], y[tr], 0, arch=arch), X[te])})["p"]
            rows.append(dict(variant=variant, protocol="within", dataset=d, run=0, auc=roc_auc_score(y[idx], p)))
            log(f"  [{variant}] within {d:<17} AUC={rows[-1]['auc']:.3f}")
    for sp in pooled_splits:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=sp).split(np.zeros(len(y)), y, groups=pid))
        p = cached(pred_dir / f"{variant}_pooled_split{sp}.npz",
                   lambda: {"p": predict(train_model(X[tr], y[tr], 0, arch=arch), X[te])})["p"]
        yt, dt = y[te], ds[te]
        rows.append(dict(variant=variant, protocol="pooled", dataset="ALL", run=sp, auc=roc_auc_score(yt, p)))
        nb = dt != "BIDS"
        if nb.sum() and len(np.unique(yt[nb])) == 2 and (~nb).sum():
            rows.append(dict(variant=variant, protocol="pooled", dataset="noBIDS", run=sp,
                             auc=roc_auc_score(yt[nb], p[nb])))
        for d in BOTH:
            m = dt == d
            if m.sum() and len(np.unique(yt[m])) == 2:
                rows.append(dict(variant=variant, protocol="pooled_perdataset", dataset=d, run=sp,
                                 auc=roc_auc_score(yt[m], p[m])))
        log(f"  [{variant}] pooled split {sp:<2} AUC={roc_auc_score(yt, p):.3f}")
    holds = lodo_holds if lodo_holds is not None else sorted(np.unique(ds))
    for d in holds:
        te = np.where(ds == d)[0]; tr = np.where(ds != d)[0]
        for s in lodo_seeds:
            p = cached(pred_dir / f"{variant}_lodo_{d}_seed{s}.npz",
                       lambda: {"p": predict(train_model(X[tr], y[tr], s, arch=arch), X[te])})["p"]
            if len(np.unique(y[te])) == 2:
                rows.append(dict(variant=variant, protocol="lodo", dataset=d, run=s, auc=roc_auc_score(y[te], p)))
                log(f"  [{variant}] LODO {d:<17} seed {s} AUC={rows[-1]['auc']:.3f}")
            else:
                rows.append(dict(variant=variant, protocol="lodo_specificity", dataset=d, run=s,
                                 auc=float((p < 0.5).mean())))
    return rows


def summary_table(df):
    out = []
    for v, g in df.groupby("variant", sort=False):
        w = g[g.protocol == "within"].groupby("dataset").auc.mean()
        l = g[g.protocol == "lodo"].groupby("dataset").auc.mean()
        pp = g[g.protocol == "pooled_perdataset"].groupby("dataset").auc.mean()
        out.append(dict(variant=v,
                        within_SAM40=w.get("SAM40"), within_MA=w.get("MentalArithmetic"),
                        within_DASPS=w.get("DASPS"), within_Mend=w.get("Mendeley"),
                        pooled=g[(g.protocol == "pooled") & (g.dataset == "ALL")].auc.mean(),
                        pooled_noBIDS=g[(g.protocol == "pooled") & (g.dataset == "noBIDS")].auc.mean(),
                        pooled_in_SAM40=pp.get("SAM40"), pooled_in_MA=pp.get("MentalArithmetic"),
                        lodo_SAM40=l.get("SAM40"), lodo_MA=l.get("MentalArithmetic"),
                        lodo_DASPS=l.get("DASPS"), lodo_Mend=l.get("Mendeley"),
                        lodo_mean=l.mean() if len(l) else np.nan))
    return pd.DataFrame(out)
