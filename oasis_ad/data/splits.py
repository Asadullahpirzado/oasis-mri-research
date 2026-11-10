"""Participant-level, repeated, stratified CV with nested inner partitions.

For each (repeat, fold):
    test   = outer held-out fold                    -> scoring ONLY
    train  = the other participants, split further into
        fit    -> gradient updates
        val    -> early stopping + temperature scaling
        calib  -> conformal quantile + rejection threshold
Nothing fitted on val/calib/test ever leaks into the numbers reported for the test fold.
All splitting is done on participant IDs; there is exactly one scan per participant (MR1),
and the group key is the participant ID, so repeat scans can never straddle two partitions.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ..utils import age_band


def _strata(df, label_col, bands, use_age, min_count):
    lab = df[label_col].astype(int).astype(str)
    if use_age:
        s = lab + "_" + df["age"].map(lambda a: age_band(a, bands))
        if s.value_counts().min() >= min_count:
            return s
    return lab


def make_outer_splits(labeled, cfg):
    """labeled: DataFrame with participant_id, label, age. Returns long table repeat/fold/participant_id/role."""
    k, R, seed = cfg.cv.n_splits, cfg.cv.n_repeats, cfg.cv.seed
    bands = [tuple(b) for b in cfg.eval.age_bands]
    strata = _strata(labeled, "label", bands, cfg.cv.stratify_age_band, k)
    if strata.value_counts().min() < k:           # tiny class (e.g. CDR>=1 in the 3-class task) -> fall back to binary strata
        strata = (labeled["label"].astype(int) > 0).astype(int).astype(str)
    rows = []
    for r in range(R):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + 100 * r)
        for f, (tr, te) in enumerate(sgkf.split(labeled, strata, groups=labeled["participant_id"])):
            tr_ids = labeled.iloc[tr]
            te_ids = labeled.iloc[te]
            inner = make_inner_split(tr_ids, cfg, seed=seed + 100 * r + f)
            for pid in te_ids["participant_id"]:
                rows.append((r, f, pid, "test"))
            for role, ids in inner.items():
                for pid in ids:
                    rows.append((r, f, pid, role))
    sp = pd.DataFrame(rows, columns=["repeat", "fold", "participant_id", "role"])
    check_no_leakage(sp)
    return sp


def make_inner_split(train_df, cfg, seed):
    y = train_df["label"].astype(int).values
    ids = train_df["participant_id"].values
    strat = y if np.bincount(y).min() >= 3 else (y > 0).astype(int)
    if np.bincount(strat).min() < 3:
        strat = None
    hold = cfg.cv.val_frac + cfg.cv.calib_frac
    fit, hold_ids, _, y_hold = train_test_split(ids, y, test_size=hold, stratify=strat, random_state=seed)
    s2 = y_hold if (np.bincount(y_hold).min() >= 2) else None
    val, calib = train_test_split(hold_ids, test_size=cfg.cv.calib_frac / hold, stratify=s2, random_state=seed)
    return {"fit": fit, "val": val, "calib": calib}


def check_no_leakage(sp):
    for (r, f), g in sp.groupby(["repeat", "fold"]):
        roles = g.groupby("role")["participant_id"].apply(set).to_dict()
        keys = list(roles)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not (roles[keys[i]] & roles[keys[j]]), f"LEAKAGE between {keys[i]} and {keys[j]} in repeat {r} fold {f}"
        # each participant exactly once per repeat as test
    t = sp[sp["role"] == "test"].groupby("repeat")["participant_id"].apply(lambda s: s.is_unique)
    assert t.all(), "a participant appears in two test folds of one repeat"


def get_ids(sp, repeat, fold, role):
    m = (sp["repeat"] == repeat) & (sp["fold"] == fold) & (sp["role"] == role)
    return sp.loc[m, "participant_id"].tolist()


def split_summary(sp, labeled):
    lab = labeled.set_index("participant_id")["label"]
    rows = []
    for (r, f), g in sp.groupby(["repeat", "fold"]):
        d = {"repeat": r, "fold": f}
        for role, gg in g.groupby("role"):
            d[f"n_{role}"] = len(gg)
            d[f"pos_{role}"] = int((lab.loc[gg["participant_id"]].astype(int) > 0).sum())
        rows.append(d)
    return pd.DataFrame(rows)
