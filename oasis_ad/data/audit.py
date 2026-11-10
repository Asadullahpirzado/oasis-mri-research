"""Data audit: build the manifest, freeze the CDR-based labels, quantify age confounding."""
import os
import numpy as np
import pandas as pd

from ..utils import participant_id, save_json, log, _clean

# Variables that must NEVER be model inputs (they are tied to how CDR is assigned or are not image/age/volume data)
FORBIDDEN_FEATURES = {"mmse", "cdr", "educ", "ses", "delay"}

LABEL_NAMES = {
    "binary": {0: "CDR 0 (no dementia)", 1: "CDR > 0 (very mild to moderate dementia)"},
    "three_class": {0: "CDR 0", 1: "CDR 0.5", 2: "CDR >= 1"},
}


def n_classes(cfg):
    return 2 if cfg.task.mode == "binary" else 3


def assign_labels(cdr):
    """FROZEN label mapping (do not change after the audit).
    binary: CDR == 0 -> 0, CDR > 0 -> 1.   three_class: 0 / 0.5 / >=1 (CDR 2 merged into >=1).
    CDR 0.5 is a research grouping, NOT a confirmed MCI diagnosis."""
    cdr = pd.Series(cdr, dtype="float64")
    b = (cdr > 0).astype("Int64").where(cdr.notna())
    t = pd.Series(np.where(cdr.isna(), np.nan, np.where(cdr == 0, 0, np.where(cdr <= 0.5, 1, 2))), index=cdr.index)
    return b, t


def build_manifest(clin, file_index, cfg):
    """One row per clinical session with a `role`:
       labeled | ood_young | excluded_repeat_scan | excluded_no_mri | excluded_no_cdr."""
    df = clin.copy()
    df["participant_id"] = df["session_id"].map(participant_id)
    df = df.merge(file_index, on="session_id", how="left")
    df["has_mri"] = df["t88_img"].notna()
    df["is_mr1"] = df["session_id"].str.endswith("_MR1")
    min_age = cfg.task.min_age

    role = []
    for _, r in df.iterrows():
        if cfg.data.use_mr1_only and not r["is_mr1"]:
            role.append("excluded_repeat_scan")
        elif not r["has_mri"]:
            role.append("excluded_no_mri")
        elif pd.notna(r["age"]) and r["age"] < min_age:
            role.append("ood_young")
        elif pd.isna(r["age"]) or pd.isna(r["cdr"]):
            role.append("excluded_no_cdr")
        else:
            role.append("labeled")
    df["role"] = role

    b, t = assign_labels(df["cdr"])
    df["label_binary"] = b
    df["label_3class"] = t
    key = "label_binary" if cfg.task.mode == "binary" else "label_3class"
    df["label"] = df[key]
    df["sex_num"] = (df["sex"] == "M").astype(float).where(df["sex"].isin(["M", "F"])) if "sex" in df else np.nan
    df.loc[df["role"] != "labeled", ["label", "label_binary", "label_3class"]] = np.nan

    lab = df[df["role"] == "labeled"]
    assert lab["participant_id"].is_unique, "Duplicate participants in the labelled cohort!"
    return df


def audit_report(clin, manifest, cfg, out_dir):
    rep = {"task_mode": cfg.task.mode, "min_age": cfg.task.min_age}
    rep["clinical_rows_total"] = int(len(clin))
    rep["unique_participants"] = int(manifest["participant_id"].nunique())
    rep["role_counts"] = manifest["role"].value_counts().to_dict()
    rep["missing_values_clinical"] = clin.isna().sum().to_dict()
    lab = manifest[manifest["role"] == "labeled"]
    rep["cohort_flow"] = {
        "all_sessions": int(len(clin)),
        "mr1_sessions": int(manifest["is_mr1"].sum()),
        "with_processed_mri": int((manifest["is_mr1"] & manifest["has_mri"]).sum()),
        f"age>={cfg.task.min_age}": int(((manifest["is_mr1"] & manifest["has_mri"]) & (manifest["age"] >= cfg.task.min_age)).sum()),
        "with_valid_cdr (analysis cohort)": int(len(lab)),
    }
    rep["cdr_distribution_labeled"] = lab["cdr"].value_counts().sort_index().to_dict()
    rep["label_counts_binary"] = lab["label_binary"].value_counts().sort_index().to_dict()
    rep["label_counts_3class"] = lab["label_3class"].value_counts().sort_index().to_dict()
    rep["age_labeled"] = lab["age"].describe().to_dict()
    n_md = int((lab["label_3class"] == 2).sum())
    rep["three_class_warning"] = ("CDR>=1 group has only %d participants: report 3-class results descriptively only." % n_md
                                  if n_md < 30 else "ok")
    rep["young_ood_n"] = int((manifest["role"] == "ood_young").sum())
    rep["young_with_cdr_values"] = int(((manifest["role"] == "ood_young") & manifest["cdr"].notna()).sum())
    rep["label_mapping"] = {"binary": "CDR=0 -> 0 ; CDR>0 -> 1",
                            "three_class": "CDR=0 -> 0 ; CDR=0.5 -> 1 ; CDR>=1 (incl. 2) -> 2",
                            "note": "CDR 0.5 is NOT a confirmed MCI diagnosis; CDR>=1 is not proof of Alzheimer's pathology."}
    rep["forbidden_model_inputs"] = sorted(FORBIDDEN_FEATURES)
    rep["cohort_rationale"] = ("Age >= %d: in OASIS-1 clinical dementia ratings are only provided for older adults, and younger "
                               "participants have no CDR labels. They are therefore kept as a held-out natural distribution shift "
                               "(OOD) set and are never used for training or scoring the classifier." % cfg.task.min_age)

    # age confounding
    from scipy import stats
    ac = {}
    if len(lab) > 5:
        rho, p = stats.spearmanr(lab["age"], lab["cdr"])
        ac["spearman_age_vs_cdr"] = {"rho": float(rho), "p": float(p)}
        g = [lab.loc[lab["label_3class"] == k, "age"].dropna() for k in (0, 1, 2)]
        g = [x for x in g if len(x) > 1]
        if len(g) >= 2:
            H, p = stats.kruskal(*g)
            ac["kruskal_age_by_class"] = {"H": float(H), "p": float(p)}
        a0, a1 = lab.loc[lab["label_binary"] == 0, "age"], lab.loc[lab["label_binary"] == 1, "age"]
        if len(a0) > 1 and len(a1) > 1:
            ac["mean_age_by_binary_class"] = {"CDR0": float(a0.mean()), "CDR>0": float(a1.mean())}
            u, p = stats.mannwhitneyu(a0, a1)
            ac["mannwhitney_age_binary"] = {"U": float(u), "p": float(p)}
        if "nwbv" in lab:
            rho, p = stats.spearmanr(lab["age"], lab["nwbv"], nan_policy="omit")
            ac["spearman_age_vs_nwbv"] = {"rho": float(rho), "p": float(p)}
    rep["age_confounding"] = ac
    rep = _clean(rep)
    save_json(rep, os.path.join(out_dir, "audit_report.json"))
    return rep


def plot_age_confounding(manifest, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = manifest[manifest["role"] == "labeled"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, name in ((0, "CDR 0"), (1, "CDR 0.5"), (2, "CDR >= 1")):
        s = lab.loc[lab["label_3class"] == k, "age"]
        if len(s):
            ax[0].hist(s, bins=10, alpha=0.55, label=f"{name} (n={len(s)})")
    ax[0].set_xlabel("Age"); ax[0].set_title("Age distribution per CDR group"); ax[0].legend()
    data = [lab.loc[lab["label_binary"] == k, "age"].dropna() for k in (0, 1)]
    ax[1].boxplot(data); ax[1].set_xticklabels(["CDR 0", "CDR > 0"]); ax[1].set_title("Age by binary class")
    if "nwbv" in lab:
        sc = ax[2].scatter(lab["age"], lab["nwbv"], c=lab["label_binary"], cmap="coolwarm", alpha=0.7, s=18)
        ax[2].set_xlabel("Age"); ax[2].set_ylabel("nWBV"); ax[2].set_title("Brain volume vs age (red = CDR>0)")
    fig.suptitle("Age confounding - OASIS-1 analysis cohort"); fig.tight_layout()
    p = os.path.join(out_dir, "age_confounding.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p
