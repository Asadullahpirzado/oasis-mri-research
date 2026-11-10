"""Simple baselines every image model must beat, evaluated with the SAME outer participant-level folds.

Features: age, sex, normalised whole-brain volume (nWBV), estimated total intracranial volume (eTIV).
MMSE / CDR / education / SES are never used (see data.audit.FORBIDDEN_FEATURES).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from .data.audit import FORBIDDEN_FEATURES
from .utils import log

FEATURE_SETS = {
    "age_only": ["age"],
    "age_sex": ["age", "sex_num"],
    "volume_age_sex": ["age", "sex_num", "nwbv", "etiv"],
    "volume_only": ["nwbv", "etiv"],
    "tissue_only": ["csf_frac", "gm_frac", "wm_frac"],                       # image-derived (segmentation) features
    "tissue_age_sex": ["age", "sex_num", "csf_frac", "gm_frac", "wm_frac"],
}


def make_estimator(kind, seed):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=seed, n_jobs=1))


def run_baselines(labeled, splits, cfg, paths, tissue=None):
    """Writes step4_baselines/<name>/fold_outputs.csv (same schema as the CNN) and returns list of names.
    Includes a majority-class baseline (the floor every model must beat) and optional tissue-fraction baselines."""
    from sklearn.dummy import DummyClassifier
    names = []
    if tissue is not None and len(tissue):
        labeled = labeled.merge(tissue, on="participant_id", how="left")
    lab = labeled.set_index("participant_id")
    n_cls = 2 if cfg.task.mode == "binary" else 3

    # majority-class (prior) baseline
    rows = []
    for (r, f), g in splits.groupby(["repeat", "fold"]):
        tr = g.loc[g["role"].isin(["fit", "val", "calib"]), "participant_id"]
        te = g.loc[g["role"] == "test", "participant_id"]
        est = DummyClassifier(strategy="prior").fit(np.zeros((len(tr), 1)), lab.loc[tr, "label"].astype(int).values)
        P = np.zeros((len(te), n_cls)); P[:, est.classes_.astype(int)] = est.predict_proba(np.zeros((len(te), 1)))
        for i, pid in enumerate(te):
            d = {"repeat": r, "fold": f, "participant_id": pid, "role": "test", "label": int(lab.loc[pid, "label"])}
            d.update({f"p_cal_{c}": float(P[i, c]) for c in range(n_cls)})
            rows.append(d)
    d = os.path.join(paths.baselines, "majority__prior"); os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "fold_outputs.csv"), index=False); names.append("majority__prior")

    for fs_name, cols in FEATURE_SETS.items():
        assert not (set(cols) & FORBIDDEN_FEATURES), "forbidden feature used"
        if any(c not in lab.columns or lab[c].isna().all() for c in cols):
            log(f"baseline '{fs_name}' skipped (features not available: {[c for c in cols if c not in lab.columns]})")
            continue
        for kind in ("logreg", "rf"):
            name = f"{fs_name}__{kind}"
            rows = []
            for (r, f), g in splits.groupby(["repeat", "fold"]):
                train_ids = g.loc[g["role"].isin(["fit", "val", "calib"]), "participant_id"]
                test_ids = g.loc[g["role"] == "test", "participant_id"]
                Xtr, ytr = lab.loc[train_ids, cols].values.astype(float), lab.loc[train_ids, "label"].astype(int).values
                Xte, yte = lab.loc[test_ids, cols].values.astype(float), lab.loc[test_ids, "label"].astype(int).values
                est = make_estimator(kind, cfg.cv.seed + r)
                est.fit(Xtr, ytr)
                P = np.zeros((len(Xte), n_cls))
                P[:, est.classes_.astype(int)] = est.predict_proba(Xte)
                for i, pid in enumerate(test_ids):
                    d = {"repeat": r, "fold": f, "participant_id": pid, "role": "test", "label": int(yte[i])}
                    for c in range(n_cls):
                        d[f"p_cal_{c}"] = float(P[i, c])
                    rows.append(d)
            d = os.path.join(paths.baselines, name); os.makedirs(d, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(d, "fold_outputs.csv"), index=False)
            names.append(name)
    log(f"baselines written: {names}")
    return names


def age_matched_ids(labeled, caliper=3, seed=0):
    """1:1 nearest-neighbour matching on age (without replacement) between label==1 and label==0 participants.
    Works for the binary task; for the 3-class task the binary (>0) split is matched."""
    rng = np.random.default_rng(seed)
    pos = labeled[labeled["label"].astype(int) > 0].sample(frac=1, random_state=seed)
    neg = labeled[labeled["label"].astype(int) == 0].copy()
    used, pairs = set(), []
    for _, p in pos.iterrows():
        cand = neg[~neg["participant_id"].isin(used)].copy()
        cand["d"] = (cand["age"] - p["age"]).abs()
        cand = cand[cand["d"] <= caliper].sort_values("d")
        if len(cand):
            q = cand.iloc[0]
            used.add(q["participant_id"]); pairs.append((p["participant_id"], q["participant_id"], p["age"], q["age"]))
    df = pd.DataFrame(pairs, columns=["pos_id", "neg_id", "pos_age", "neg_age"])
    return df
