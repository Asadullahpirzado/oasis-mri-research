"""External validation of the FROZEN model bundle on another cohort (e.g. OASIS-3), if access and label compatibility allow.

Input table (CSV): participant_id, path, label, [age], [sex]
  path  : volume file readable by nibabel (.nii / .nii.gz / Analyze .img) in the SAME kind of space as OASIS-1 processed scans
          (atlas-registered, brain-masked). Otherwise the QC check will flag/abstain - that is the intended behaviour.
  label : 0 / 1 with exactly the same meaning as the training labels (CDR 0 vs CDR > 0). Check the label definition first!
Nothing is re-fitted: temperature, conformal q-hat, rejection threshold and OOD thresholds all come from the OASIS-1 bundle.
"""
import numpy as np
import pandas as pd

from .data.oasis_io import load_volume
from .inference import Predictor
from . import uncertainty as U


def evaluate_external(bundle_dir, table_csv, n_boot=500, with_cam=False):
    pred = Predictor(bundle_dir)
    tbl = pd.read_csv(table_csv)
    names = pred.b["class_names"]
    rows = []
    for _, r in tbl.iterrows():
        raw = load_volume(r["path"])
        age = float(r["age"]) if "age" in tbl and pd.notna(r.get("age")) else None
        sex = r["sex"] if "sex" in tbl and pd.notna(r.get("sex")) else None
        res = pred.predict_raw(raw, age, sex, with_cam=with_cam)
        p = np.array([res["probabilities"][n] for n in names])
        rows.append({"participant_id": r["participant_id"], "label": int(r["label"]), **{f"p_{i}": p[i] for i in range(len(p))},
                     "conformal_set": ",".join(str(names.index(n)) for n in res["conformal_set"]),
                     "abstain": res["abstain"], "qc_fail": res["qc"]["hard_fail"], "ood_flag": res["ood"]["flag"],
                     "confidence": res["confidence"]})
    df = pd.DataFrame(rows)
    P = df[[c for c in df.columns if c.startswith("p_")]].values
    y = df["label"].values.astype(int)
    out = {"n": len(df), "overall": U.metrics_from_probs(y, P) if len(np.unique(y)) > 1 else {}}
    out["calibration"] = U.calibration_summary(P, y)
    sets = [[int(x) for x in s.split(",") if x != ""] for s in df["conformal_set"]]
    out["conformal"] = U.coverage_table(sets, y)["overall"]
    out["conformal"]["target_coverage"] = 1 - pred.b["alpha"]
    keep = ~df["abstain"].values
    out["abstention_rate"] = float(df["abstain"].mean())
    out["qc_fail_rate"] = float(df["qc_fail"].mean()); out["ood_flag_rate"] = float(df["ood_flag"].mean())
    if keep.sum() >= 5 and len(np.unique(y[keep])) > 1:
        out["on_non_abstained"] = U.metrics_from_probs(y[keep], P[keep])
        out["on_non_abstained"]["n"] = int(keep.sum())
    out["note"] = ("External performance should be compared with the internal out-of-fold results; a drop in AUROC, worse calibration, "
                   "conformal coverage below target and a high OOD/QC flag rate are all evidence of dataset shift.")
    return out, df
