"""Temperature scaling, calibration metrics, split conformal prediction, pooled metrics and bootstrap CIs (numpy only)."""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score,
                             recall_score, precision_score)


# ─────────────────────────── probabilities ───────────────────────────
def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def fit_temperature(logits, labels, iters=200):
    """Minimise NLL over log-temperature (keeps T>0). Fitted on the inner VALIDATION set only."""
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(labels, dtype=int)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return 1.0

    def nll(logT):
        p = softmax(z, np.exp(logT))
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    grid = np.linspace(-2.0, 3.0, 201)                      # T in [0.135, 20]
    vals = [nll(g) for g in grid]
    g0 = grid[int(np.argmin(vals))]
    lo, hi = g0 - 0.05, g0 + 0.05                           # golden-section refinement
    for _ in range(iters // 4):
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(a) < nll(b): hi = b
        else: lo = a
    return float(np.exp((lo + hi) / 2))


# ─────────────────────────── calibration metrics ───────────────────────────
def ece(probs, labels, n_bins=10, adaptive=False):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); acc = (pred == labels).astype(float)
    n = len(labels)
    if adaptive:
        order = np.argsort(conf); bins = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
    e = 0.0
    for b in bins:
        if len(b):
            e += len(b) / n * abs(acc[b].mean() - conf[b].mean())
    return float(e)


def nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier(probs, labels):
    oh = np.eye(probs.shape[1])[labels]
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def calibration_summary(probs, labels, n_bins=10):
    return {"ECE_equal_width": ece(probs, labels, n_bins, False), "ECE_adaptive": ece(probs, labels, n_bins, True),
            "Brier": brier(probs, labels), "NLL": nll(probs, labels)}


def reliability_points(probs, labels, n_bins=10, adaptive=False):
    conf = probs.max(axis=1); acc = (probs.argmax(axis=1) == labels).astype(float)
    if adaptive:
        bins = np.array_split(np.argsort(conf), n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
    pts = [(conf[b].mean(), acc[b].mean(), len(b)) for b in bins if len(b)]
    return np.array(pts)


# ─────────────────────────── conformal prediction ───────────────────────────
def conformal_qhat(probs_calib, y_calib, alpha):
    """Split conformal, score s = 1 - p(true class). Finite-sample corrected quantile (method='higher')."""
    s = 1.0 - probs_calib[np.arange(len(y_calib)), y_calib]
    n = len(s)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def prediction_sets(probs, qhat):
    return [np.where(1.0 - p <= qhat)[0].tolist() for p in probs]


def coverage_table(sets, y, groups=None):
    """overall + per-group empirical coverage, mean set size, share of empty / multi-class sets."""
    cov = np.array([int(t in s) for s, t in zip(sets, y)], dtype=float)
    size = np.array([len(s) for s in sets], dtype=float)

    def one(m):
        return {"n": int(m.sum()), "coverage": float(cov[m].mean()) if m.any() else np.nan,
                "avg_set_size": float(size[m].mean()) if m.any() else np.nan,
                "empty_frac": float((size[m] == 0).mean()) if m.any() else np.nan,
                "multi_frac": float((size[m] > 1).mean()) if m.any() else np.nan}

    out = {"overall": one(np.ones(len(y), bool))}
    if groups is not None:
        for name, g in groups.items():
            g = np.asarray(g)
            for v in sorted(set(g.tolist())):
                out[f"{name}={v}"] = one(g == v)
    return out


# ─────────────────────────── discrimination metrics ───────────────────────────
def metrics_from_probs(y, P):
    """balanced accuracy, macro-F1, AUROC (+ per-class precision/recall). P: (N, C)."""
    y = np.asarray(y, dtype=int); pred = P.argmax(axis=1); C = P.shape[1]
    out = {"balanced_acc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    try:
        if C == 2:
            out["auroc"] = float(roc_auc_score(y, P[:, 1]))
            out["auprc"] = float(average_precision_score(y, P[:, 1]))
        else:
            present = np.unique(y)
            out["auroc"] = float(roc_auc_score(y, P[:, present] / P[:, present].sum(1, keepdims=True),
                                               multi_class="ovr", average="macro", labels=present)) if len(present) > 1 else np.nan
    except ValueError:
        out["auroc"] = np.nan
    rec = recall_score(y, pred, average=None, labels=list(range(C)), zero_division=0)
    prec = precision_score(y, pred, average=None, labels=list(range(C)), zero_division=0)
    for c in range(C):
        out[f"recall_c{c}"] = float(rec[c]); out[f"precision_c{c}"] = float(prec[c])
    if C == 2:                                     # explicit names for the binary task (class 1 = CDR>0 is 'positive')
        out["sensitivity"] = out["recall_c1"]; out["specificity"] = out["recall_c0"]
    return out


def bootstrap_participants(y, P_list, fn, n_boot=1000, seed=0, key=None):
    """Participant-level bootstrap. y: (N,), P_list: list over CV repeats of (N, C) arrays aligned to the SAME participants.
    fn(y, P) -> dict. Point estimate = mean over repeats; CI = percentile CI of the bootstrap mean."""
    rng = np.random.default_rng(seed)
    N = len(y)
    point = {}
    for P in P_list:
        for k, v in fn(y, P).items():
            point.setdefault(k, []).append(v)
    point = {k: float(np.nanmean(v)) for k, v in point.items()}
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        agg = {}
        for P in P_list:
            for k, v in fn(y[idx], P[idx]).items():
                agg.setdefault(k, []).append(v)
        for k in boots:
            boots[k].append(np.nanmean(agg[k]))
    res = {}
    for k in point:
        b = np.array(boots[k], dtype=float)
        res[k] = {"mean": point[k], "ci_lo": float(np.nanpercentile(b, 2.5)) if len(b) else np.nan,
                  "ci_hi": float(np.nanpercentile(b, 97.5)) if len(b) else np.nan}
    return res


def paired_bootstrap_diff(y, P_list_a, P_list_b, metric="auroc", n_boot=1000, seed=0):
    """Paired participant-level bootstrap of metric(A) - metric(B) (same resampled participants for both)."""
    rng = np.random.default_rng(seed)
    N = len(y)

    def m(Pl, idx):
        return np.nanmean([metrics_from_probs(y[idx], P[idx])[metric] for P in Pl])

    full = np.arange(N)
    d0 = m(P_list_a, full) - m(P_list_b, full)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(m(P_list_a, idx) - m(P_list_b, idx))
    ds = np.array(ds)
    return {"diff": float(d0), "ci_lo": float(np.percentile(ds, 2.5)), "ci_hi": float(np.percentile(ds, 97.5)),
            "p_a_not_better": float((ds <= 0).mean())}
