"""Pooled out-of-fold evaluation (calibration, conformal prediction, baselines vs MRI, age control, selective prediction).

Every participant is a TEST case exactly once per CV repeat. Metrics are computed on the pooled out-of-fold
predictions of each repeat, averaged over repeats, with participant-level bootstrap CIs.
"""
import os, glob
import numpy as np
import pandas as pd

from .utils import save_json, load_json, age_band
from . import uncertainty as U


# ─────────────────────────── loading ───────────────────────────
def load_fold_outputs(folder):
    fs = sorted(glob.glob(os.path.join(folder, "**", "fold_outputs.csv"), recursive=True))
    if not fs:
        raise FileNotFoundError(f"no fold_outputs.csv under {folder}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def prob_cols(df, prefix="p_cal_"):
    return sorted([c for c in df.columns if c.startswith(prefix)], key=lambda c: int(c.split("_")[-1]))


class PredSet:
    """Aligned out-of-fold predictions: ids (N,), y (N,), P_list = list over repeats of (N, C)."""
    def __init__(self, name, ids, y, P_list, meta):
        self.name, self.ids, self.y, self.P_list, self.meta = name, ids, y, P_list, meta

    @property
    def P_ens(self):
        return np.mean(self.P_list, axis=0)


def to_predset(df, name, manifest, ids=None):
    t = df[df["role"] == "test"]
    cols = prob_cols(t)
    ids = sorted(t["participant_id"].unique()) if ids is None else ids
    P_list, y = [], None
    for r in sorted(t["repeat"].unique()):
        g = t[t["repeat"] == r].set_index("participant_id").loc[ids]
        P_list.append(g[cols].values.astype(float))
        y = g["label"].values.astype(int)
    meta = manifest.set_index("participant_id").loc[ids, ["age", "sex"]].reset_index()
    return PredSet(name, np.array(ids), y, P_list, meta)


def ensemble_predset(ps):
    """Deep ensemble of the CV repeats: each participant's probabilities are averaged over the models (one per repeat)
    that never saw them. With a single repeat this equals the single model."""
    return PredSet(ps.name + "_deep_ensemble", ps.ids, ps.y, [ps.P_ens], ps.meta)


def subgroup_table(predsets, bands, n_boot=300, min_n=20):
    """Classification metrics per subgroup (age band, sex) - shows whether performance depends on cohort characteristics."""
    from .utils import age_band
    rows = []
    age = predsets[0].meta["age"].values.astype(float)
    sex = predsets[0].meta["sex"].values
    groups = {"all": np.ones(len(age), bool)}
    for lo, hi in bands:
        groups[f"age {lo}-{hi}"] = (age >= lo) & (age <= hi)
    for sx in ("F", "M"):
        groups[f"sex {sx}"] = sex == sx
    for ps in predsets:
        for gname, m in groups.items():
            y = ps.y[m]
            row = {"model": ps.name, "subgroup": gname, "n": int(m.sum()), "n_positive": int((y > 0).sum()), "note": ""}
            if m.sum() >= min_n and len(np.unique(y)) > 1:
                sm = U.bootstrap_participants(y, [P[m] for P in ps.P_list], U.metrics_from_probs, n_boot=n_boot)
                for k in ("balanced_acc", "auroc", "sensitivity", "specificity"):
                    if k in sm:
                        row[k] = round(sm[k]["mean"], 3); row[k + "_CI"] = f"[{sm[k]['ci_lo']:.2f}, {sm[k]['ci_hi']:.2f}]"
            else:
                row["note"] = "too few participants / single class - not estimated"
            rows.append(row)
    df = pd.DataFrame(rows)
    if "note" not in df.columns:
        df["note"] = ""
    return df


# ─────────────────────────── discrimination ───────────────────────────
def summarize_predset(ps, n_boot):
    return U.bootstrap_participants(ps.y, ps.P_list, U.metrics_from_probs, n_boot=n_boot)


def compare_models(predsets, n_boot, out_dir):
    rows = []
    for ps in predsets:
        s = summarize_predset(ps, n_boot)
        row = {"model": ps.name}
        for k in ("balanced_acc", "macro_f1", "auroc", "sensitivity", "specificity"):
            if k in s:
                row[k] = round(s[k]["mean"], 4)
                row[k + "_CI"] = f"[{s[k]['ci_lo']:.3f}, {s[k]['ci_hi']:.3f}]"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    return df


def age_matched_metrics(predsets, pairs, n_boot):
    keep = set(pairs["pos_id"]) | set(pairs["neg_id"])
    rows = []
    for ps in predsets:
        m = np.array([i in keep for i in ps.ids])
        if m.sum() < 10:
            continue
        sub = PredSet(ps.name, ps.ids[m], ps.y[m], [P[m] for P in ps.P_list], None)
        # binarise for the matched analysis (matched on CDR>0 vs 0)
        s = U.bootstrap_participants(sub.y, sub.P_list, U.metrics_from_probs, n_boot=min(n_boot, 500))
        rows.append({"model": ps.name, "n_matched": int(m.sum()),
                     "balanced_acc": round(s["balanced_acc"]["mean"], 4),
                     "auroc": round(s.get("auroc", {"mean": np.nan})["mean"], 4),
                     "auroc_CI": f"[{s['auroc']['ci_lo']:.3f}, {s['auroc']['ci_hi']:.3f}]" if "auroc" in s else ""})
    return pd.DataFrame(rows)


def age_dependence(ps, age):
    """How strongly do the model scores track age? (Spearman of P(class>0) with age, overall and within class)."""
    from scipy import stats
    P = ps.P_ens
    score = 1.0 - P[:, 0]
    out = {"spearman_score_age_all": float(stats.spearmanr(score, age)[0])}
    for c in np.unique(ps.y):
        m = ps.y == c
        if m.sum() > 5:
            out[f"spearman_score_age_class{c}"] = float(stats.spearmanr(score[m], age[m])[0])
    return out


def incremental_value_over_age(ps, age, n_splits=5, seed=0):
    """Cross-validated AUROC of logistic regression: age alone vs age + MRI-model logit (out-of-fold scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    y = (ps.y > 0).astype(int)
    P = np.clip(ps.P_ens, 1e-6, 1 - 1e-6)
    logit = np.log((1 - P[:, 0]) / P[:, 0])
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    res = {}
    for name, X in (("age_only", age.reshape(-1, 1)), ("age_plus_mri_logit", np.c_[age, logit]), ("mri_logit_only", logit.reshape(-1, 1))):
        pr = cross_val_predict(LogisticRegression(max_iter=1000), (X - X.mean(0)) / (X.std(0) + 1e-9), y, cv=cv, method="predict_proba")[:, 1]
        res[name] = float(roc_auc_score(y, pr))
    res["gain_over_age"] = res["age_plus_mri_logit"] - res["age_only"]
    return res


# ─────────────────────────── calibration ───────────────────────────
def calibration_report(df_model, n_bins=10):
    """Before/after temperature scaling on the pooled TEST predictions of each repeat."""
    t = df_model[df_model["role"] == "test"]
    lc = [c for c in t.columns if c.startswith("logit_")]
    lc = sorted(lc, key=lambda c: int(c.split("_")[-1]))
    rows = []
    for r, g in t.groupby("repeat"):
        y = g["label"].values.astype(int)
        raw = U.softmax(g[lc].values)
        cal = g[prob_cols(g)].values
        mc = g[prob_cols(g, "p_mc_")].values
        for tag, P in (("raw (T=1)", raw), ("temperature-scaled", cal), ("temp-scaled + MC dropout", mc)):
            rows.append({"repeat": r, "variant": tag, **U.calibration_summary(P, y, n_bins)})
    # deep ensemble over repeats (each participant averaged over models that never saw them)
    pc = prob_cols(t)
    if t["repeat"].nunique() > 1:
        g = t.groupby("participant_id")[pc + ["label"]].mean()
        y = g["label"].round().values.astype(int)
        rows.append({"repeat": -1, "variant": "deep ensemble of repeats (temp-scaled members)", **U.calibration_summary(g[pc].values, y, n_bins)})
    df = pd.DataFrame(rows)
    out = df.groupby("variant").mean(numeric_only=True).drop(columns=["repeat"]).reset_index()
    order = {"raw (T=1)": 0, "temperature-scaled": 1, "temp-scaled + MC dropout": 2}
    out["_o"] = out["variant"].map(lambda v: order.get(v, 3)); out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return out, df


def uncertainty_vs_error(df_model):
    from scipy import stats
    t = df_model[df_model["role"] == "test"]
    P = t[prob_cols(t, "p_mc_")].values
    err = (P.argmax(1) != t["label"].values).astype(float)
    out = {}
    for name in ("mc_entropy", "mc_mutual_info"):
        rho, p = stats.spearmanr(t[name].values, err)
        out[name] = {"spearman_with_error": float(rho), "p": float(p)}
    out["error_rate_overall"] = float(err.mean())
    q = pd.qcut(t["mc_entropy"], 5, duplicates="drop", labels=False)
    out["error_by_entropy_quintile"] = {int(k): float(v) for k, v in pd.Series(err).groupby(q.values).mean().items()}
    return out


# ─────────────────────────── conformal ───────────────────────────
def conformal_report(df_model, manifest, cfg):
    """Per fold: q_hat from the inner CALIBRATION set -> sets for that fold's TEST participants. Pool + subgroup coverage."""
    alpha = 1 - cfg.uncertainty.target_coverage
    bands = [tuple(b) for b in cfg.eval.age_bands]
    meta = manifest.set_index("participant_id")
    per_repeat, rows_all = [], []
    for (r, f), g in df_model.groupby(["repeat", "fold"]):
        cal = g[g["role"] == "calib"]; te = g[g["role"] == "test"]
        cols = prob_cols(g)
        q = U.conformal_qhat(cal[cols].values, cal["label"].values.astype(int), alpha)
        sets = U.prediction_sets(te[cols].values, q)
        d = te[["repeat", "fold", "participant_id", "label"]].copy()
        d["qhat"] = q
        d["set"] = [",".join(map(str, s)) for s in sets]
        d["set_size"] = [len(s) for s in sets]
        d["covered"] = [int(t in s) for s, t in zip(sets, te["label"].astype(int))]
        rows_all.append(d)
    allr = pd.concat(rows_all, ignore_index=True)
    allr["age"] = meta.loc[allr["participant_id"], "age"].values
    allr["sex"] = meta.loc[allr["participant_id"], "sex"].values
    allr["age_band"] = allr["age"].map(lambda a: age_band(a, bands))
    out = {}
    for r, g in allr.groupby("repeat"):
        sets = [[int(x) for x in s.split(",") if x != ""] for s in g["set"]]
        out[f"repeat{r}"] = U.coverage_table(sets, g["label"].values.astype(int),
                                              {"age_band": g["age_band"].values, "sex": g["sex"].values, "class": g["label"].values})
    summary = {"target_coverage": cfg.uncertainty.target_coverage, "alpha": alpha}
    # average over repeats
    keys = out["repeat0"].keys()
    avg = {}
    for k in keys:
        avg[k] = {m: float(np.nanmean([out[r][k][m] for r in out if k in out[r]])) for m in out["repeat0"][k]}
    summary["pooled_mean_over_repeats"] = avg
    # bootstrap CI of overall coverage over participants (repeat-averaged coverage per participant)
    cov_p = allr.groupby("participant_id")["covered"].mean().values
    rng = np.random.default_rng(0)
    b = [cov_p[rng.integers(0, len(cov_p), len(cov_p))].mean() for _ in range(cfg.eval.n_boot)]
    summary["overall_coverage_CI95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    summary["note"] = ("Conformal guarantees are MARGINAL and assume exchangeability; subgroup coverage can fall short. "
                       "Each fold's q_hat uses only that fold's inner calibration set.")
    return summary, allr


# ─────────────────────────── selective prediction ───────────────────────────
def risk_coverage(conf, correct):
    order = np.argsort(-conf, kind="stable")
    c = np.asarray(correct)[order].astype(float)
    k = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / k
    cov = k / len(c)
    return cov, risk


def aurc(conf, correct):
    cov, risk = risk_coverage(conf, correct)
    return float(np.mean(risk))                       # mean selective risk over all coverage levels


def selective_acc_at(conf, correct, coverage):
    n = len(conf); k = max(1, int(np.ceil(coverage * n)))
    order = np.argsort(-conf, kind="stable")[:k]
    return float(np.asarray(correct)[order].mean())


def selective_report(df_model, cfg):
    """Rejection threshold per fold from the inner calibration set; evaluated on the fold's test set.
    Also pooled risk-coverage curves for confidence and MC-entropy rankings."""
    tau_cov = cfg.uncertainty.selective_coverage
    res_rows = []
    for (r, f), g in df_model.groupby(["repeat", "fold"]):
        cal = g[g["role"] == "calib"]; te = g[g["role"] == "test"]
        cols = prob_cols(g)
        conf_cal = cal[cols].values.max(1)
        tau = float(np.quantile(conf_cal, 1 - tau_cov))        # keep the most confident `tau_cov` share of calibration cases
        P = te[cols].values; conf = P.max(1); pred = P.argmax(1); y = te["label"].values.astype(int)
        keep = conf >= tau
        res_rows.append({"repeat": r, "fold": f, "tau": tau, "n_test": len(te), "coverage": float(keep.mean()),
                         "acc_all": float((pred == y).mean()),
                         "acc_kept": float((pred[keep] == y[keep]).mean()) if keep.any() else np.nan})
    folds = pd.DataFrame(res_rows)
    t = df_model[df_model["role"] == "test"]
    out = {"rejection_rule": f"abstain when calibrated confidence < tau (tau keeps {tau_cov:.0%} of the fold's calibration cases)",
           "per_fold_mean": folds[["tau", "coverage", "acc_all", "acc_kept"]].mean().to_dict()}
    curves = {}
    for r, g in t.groupby("repeat"):
        P = g[prob_cols(g)].values; y = g["label"].values.astype(int); correct = (P.argmax(1) == y)
        conf = P.max(1)
        ent = g["mc_entropy"].values
        rr = {"AURC_confidence": aurc(conf, correct), "AURC_mc_entropy": aurc(-ent, correct),
              "AURC_random": float(1 - correct.mean()),
              "AURC_oracle": float(np.mean(1 - np.cumsum(np.sort(correct)[::-1]) / np.arange(1, len(correct) + 1)))}
        for c in (0.9, 0.8):
            rr[f"selective_acc@{int(c*100)}"] = selective_acc_at(conf, correct, c)
        rr["full_coverage_acc"] = float(correct.mean())
        curves[f"repeat{r}"] = rr
    out["pooled_by_repeat"] = curves
    out["pooled_mean"] = pd.DataFrame(curves).T.mean().to_dict()
    return out, folds


# ─────────────────────────── plots ───────────────────────────
def plot_reliability(df_model, path, n_bins=10):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = df_model[df_model["role"] == "test"]
    lc = sorted([c for c in t.columns if c.startswith("logit_")], key=lambda c: int(c.split("_")[-1]))
    y = t["label"].values.astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, (title, P) in zip(ax, (("Before temperature scaling", U.softmax(t[lc].values)), ("After temperature scaling", t[prob_cols(t)].values))):
        pts = U.reliability_points(P, y, n_bins, adaptive=True)
        a.plot([0.4, 1], [0.4, 1], "k--", label="perfect")
        a.plot(pts[:, 0], pts[:, 1], "o-", label="model (adaptive bins)")
        a.set_xlabel("confidence"); a.set_ylabel("accuracy"); a.set_title(f"{title}\nECE={U.ece(P, y, n_bins, True):.3f}")
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_risk_coverage(df_model, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = df_model[df_model["role"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r, g in t.groupby("repeat"):
        P = g[prob_cols(g)].values; y = g["label"].values.astype(int); correct = P.argmax(1) == y
        for lab, score in (("confidence", P.max(1)), ("MC entropy", -g["mc_entropy"].values)):
            cov, risk = risk_coverage(score, correct)
            ax.plot(cov, risk, alpha=0.7, color="C0" if lab == "confidence" else "C1", label=lab if r == 0 else None)
    ax.set_xlabel("coverage (share of cases predicted)"); ax.set_ylabel("risk (error rate on predicted cases)")
    ax.set_title("Risk-coverage curves (pooled out-of-fold, one line per CV repeat)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
