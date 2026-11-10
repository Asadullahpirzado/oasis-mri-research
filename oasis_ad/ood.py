"""Out-of-distribution / robustness analysis.

Scores (higher = more unusual):
  * mahalanobis  - class-conditional means, tied Ledoit-Wolf shrinkage covariance on penultimate embeddings of the
                   TRAINING (fit) participants of the fold; score = min distance over classes
  * mc_entropy   - predictive entropy of MC-dropout probabilities
  * mc_mutual_info - epistemic part of the uncertainty
Each score is converted to a z-score with the fold's inner CALIBRATION set, and the OOD flag threshold is the
`thresh_percentile` of calibration scores -> the false-alarm rate on clean test scans can be measured honestly.

Shift sets:
  * natural   - younger OASIS-1 participants (< min_age). Used ONLY here, never for training or scoring.
  * synthetic - TorchIO corruptions (noise, blur, motion, ghosting, bias field, low resolution) at several severities.
These show ROBUSTNESS only; they do not prove generalisation to other scanners or cohorts.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from .train import predict_logits, mc_logits
from .uncertainty import softmax, entropy
from .data.dataset import VolumeStore


@torch.no_grad()
def embeddings(model, store, ids, device, bs=4):
    from torch.utils.data import DataLoader
    from .data.dataset import VolDataset
    model.eval()
    out = []
    for x, _ in DataLoader(VolDataset(store, ids, None), batch_size=bs):
        out.append(model.embed(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embeddings_from_arrays(model, vols, device, bs=4):
    """vols: list of (D,H,W) float32 arrays (already preprocessed)."""
    model.eval()
    out = []
    for i in range(0, len(vols), bs):
        x = torch.from_numpy(np.stack(vols[i:i + bs])).unsqueeze(1).to(device)
        out.append(model.embed(x).float().cpu())
    return torch.cat(out).numpy()


class Mahalanobis:
    def fit(self, emb, y, n_classes):
        self.classes = [c for c in range(n_classes) if (y == c).sum() >= 2]
        self.mu = np.stack([emb[y == c].mean(0) for c in self.classes])
        centered = np.concatenate([emb[y == c] - self.mu[i] for i, c in enumerate(self.classes)])
        self.prec = LedoitWolf().fit(centered).precision_
        return self

    def score(self, emb):
        d = []
        for m in self.mu:
            z = emb - m
            d.append(np.einsum("ij,jk,ik->i", z, self.prec, z))
        return np.sqrt(np.maximum(np.min(d, axis=0), 0))

    def to_dict(self):
        return {"classes": np.array(self.classes), "mu": self.mu, "prec": self.prec}

    @classmethod
    def from_dict(cls, d):
        o = cls(); o.classes = list(d["classes"]); o.mu = d["mu"]; o.prec = d["prec"]
        return o


class ScoreNormalizer:
    """z-score + flag threshold learned on the inner calibration set of a fold."""
    def fit(self, calib_scores, pct):
        self.mean, self.std = float(np.mean(calib_scores)), float(np.std(calib_scores) + 1e-9)
        self.thr = float(np.percentile(calib_scores, pct))
        return self

    def z(self, s):
        return (np.asarray(s) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean, "std": self.std, "thr": self.thr}

    @classmethod
    def from_dict(cls, d):
        o = cls(); o.mean, o.std, o.thr = d["mean"], d["std"], d["thr"]; return o


# ───────────────────────── synthetic corruptions (TorchIO) ─────────────────────────
def corrupt(vol, name, severity, seed):
    """vol: (D,H,W) float32 preprocessed volume. Returns corrupted copy. Raises on failure (never silently falls back)."""
    import torchio as tio
    torch.manual_seed(seed); np.random.seed(seed)
    t = torch.from_numpy(vol).unsqueeze(0)
    img = tio.ScalarImage(tensor=t)
    s = float(severity)
    if name == "noise":
        tr = tio.RandomNoise(mean=0, std=(0.05 + 0.6 * s, 0.05 + 0.6 * s))
    elif name == "blur":
        tr = tio.RandomBlur(std=(0.3 + 2.0 * s, 0.3 + 2.0 * s))
    elif name == "motion":
        tr = tio.RandomMotion(degrees=(2 + 10 * s, 2 + 10 * s), translation=(2 + 8 * s, 2 + 8 * s), num_transforms=2)
    elif name == "ghosting":
        tr = tio.RandomGhosting(num_ghosts=(4, 6), axes=(0, 1, 2), intensity=(0.2 + 0.8 * s, 0.2 + 0.8 * s))
    elif name == "bias_field":
        tr = tio.RandomBiasField(coefficients=(0.2 + 0.8 * s, 0.2 + 0.8 * s), order=3)
    elif name == "low_res":
        tr = tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.0 + 3 * s, 1.0 + 3 * s))
    else:
        raise ValueError(name)
    out = tr(img).data[0].numpy().astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError(f"{name} produced non-finite values")
    return out


# ───────────────────────── metrics ─────────────────────────
def ood_metrics(s_in, s_out):
    y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    s = np.r_[s_in, s_out]
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(tpr, 0.95)
    return {"AUROC": float(roc_auc_score(y, s)), "AUPR": float(average_precision_score(y, s)),
            "FPR@95TPR": float(fpr[min(i, len(fpr) - 1)])}


def scores_for_arrays(model, vols, norm_maha, device, T, mc_passes, maha, norms):
    """Return dict of raw scores for a list of preprocessed volumes."""
    emb = embeddings_from_arrays(model, vols, device)
    z = np.concatenate([_logits(model, vols[i:i + 4], device) for i in range(0, len(vols), 4)])
    sc = {"mahalanobis": maha.score(emb)}
    mcz = np.stack([_logits(model, vols, device, mc=True) for _ in range(mc_passes)])     # (P,N,C)
    mcp = softmax(mcz.reshape(-1, mcz.shape[-1]), T).reshape(mcz.shape)
    mean_p = mcp.mean(0)
    sc["mc_entropy"] = entropy(mean_p)
    sc["mc_mutual_info"] = entropy(mean_p) - entropy(mcp.reshape(-1, mcp.shape[-1])).reshape(mcp.shape[:2]).mean(0)
    sc["probs"] = softmax(z, T)
    return sc


@torch.no_grad()
def _logits(model, vols, device, mc=False):
    from .train import enable_mc_dropout
    if mc:
        enable_mc_dropout(model)
    else:
        model.eval()
    out = []
    for i in range(0, len(vols), 4):
        x = torch.from_numpy(np.stack(vols[i:i + 4])).unsqueeze(1).to(device)
        out.append(model(x).float().cpu())
    model.eval()
    return torch.cat(out).numpy()


# ───────────────────────── ensemble scores (disagreement between independently trained models) ─────────────────────────
def unseen_keys(test_fold, pid, R):
    """The R models (one per CV repeat) that never trained on participant `pid` = the model whose TEST fold contains them."""
    return tuple((r, test_fold[(r, pid)]) for r in range(R))


def ens_scores(vols, keys_list, get_model, get_T, device):
    """For each volume use its own tuple of model keys. Returns arrays: ens_disagreement (max over classes of the SD of
    calibrated probabilities across models), ens_mutual_info (epistemic part), ens_entropy."""
    n = len(vols)
    dis, mi, ent = np.zeros(n), np.zeros(n), np.zeros(n)
    groups = {}
    for i, k in enumerate(keys_list):
        groups.setdefault(tuple(k), []).append(i)
    for keys, idx in groups.items():
        vs = [vols[i] for i in idx]
        P = np.stack([softmax(_logits(get_model(k), vs, device), get_T(k)) for k in keys])      # (M, n_i, C)
        mean_p = P.mean(0)
        dis[idx] = P.std(0).max(1)
        ent[idx] = entropy(mean_p)
        mi[idx] = entropy(mean_p) - np.mean([entropy(p) for p in P], axis=0)
    return {"ens_disagreement": dis, "ens_mutual_info": mi, "ens_entropy": ent}


# ───────────────────────── full analysis driver ─────────────────────────
SCORE_NAMES = ["mahalanobis", "mc_entropy", "mc_mutual_info"]
ENS_SCORE_NAMES = ["ens_disagreement", "ens_mutual_info"]


def fit_fold_ood(model, store, labels, ids, cfg, device, T):
    """Fit Mahalanobis on the fold's FIT participants, normalisers on its CALIB participants."""
    n_cls = 2 if cfg.task.mode == "binary" else 3
    y_fit = np.array([labels[p] for p in ids["fit"]])
    maha = Mahalanobis().fit(embeddings(model, store, ids["fit"], device), y_fit, n_cls)
    calib_vols = [store.get(p) for p in ids["calib"]]
    sc = scores_for_arrays(model, calib_vols, None, device, T, cfg.train.mc_passes, maha, None)
    norms = {k: ScoreNormalizer().fit(sc[k], cfg.ood.thresh_percentile) for k in SCORE_NAMES}
    return maha, norms


def run_ood_analysis(model_name, cfg, paths, labeled, young_ids, splits, store, device, repeat=0, log=print):
    import os
    from .train import load_model_for_fold
    from .data.splits import get_ids
    from .utils import load_json, save_json
    labels = dict(zip(labeled["participant_id"], labeled["label"].astype(int)))
    out_dir = os.path.join(paths.ood, model_name); os.makedirs(out_dir, exist_ok=True)
    R = cfg.cv.n_repeats
    # Check if all models for the R repeats actually exist on disk (prevents crash if user trained fewer repeats)
    available_repeats = []
    for r_idx in range(R):
        if all(os.path.exists(os.path.join(paths.models(model_name), f"r{r_idx}_f{f_idx}", "best.pt")) for f_idx in range(min(int(cfg.ood.folds_used), cfg.cv.n_splits))):
            available_repeats.append(r_idx)
    use_ens = len(available_repeats) >= 2
    if not use_ens:
        log(f"ensemble-disagreement score skipped (need >=2 trained CV repeats, found {len(available_repeats)})")
    test_fold = {(int(r), p): int(f) for r, f, p in splits.loc[splits["role"] == "test", ["repeat", "fold", "participant_id"]].itertuples(index=False)}
    mcache, tcache = {}, {}

    def get_model(k):
        if k not in mcache:
            mcache[k] = load_model_for_fold(model_name, cfg, paths, k[0], k[1], device)
        return mcache[k]

    def get_T(k):
        if k not in tcache:
            tcache[k] = load_json(os.path.join(paths.models(model_name), f"r{k[0]}_f{k[1]}", "fold_done.json"))["temperature"]
        return tcache[k]

    names = SCORE_NAMES + (ENS_SCORE_NAMES if use_ens else [])
    rows = []
    n_folds = min(int(cfg.ood.folds_used), cfg.cv.n_splits)
    for f in range(n_folds):
        T = get_T((repeat, f))
        ids = {k: get_ids(splits, repeat, f, k) for k in ("fit", "val", "calib", "test")}
        model = get_model((repeat, f))
        maha, norms = fit_fold_ood(model, store, labels, ids, cfg, device, T)
        if use_ens:
            cv = [store.get(p) for p in ids["calib"]]
            es = ens_scores(cv, [unseen_keys(test_fold, p, R) for p in ids["calib"]], get_model, get_T, device)
            for k in ENS_SCORE_NAMES:
                norms[k] = ScoreNormalizer().fit(es[k], cfg.ood.thresh_percentile)
        np.savez(os.path.join(out_dir, f"r{repeat}_f{f}_maha.npz"), **maha.to_dict())
        save_json({k: v.to_dict() for k, v in norms.items()}, os.path.join(out_dir, f"r{repeat}_f{f}_norms.json"))

        te_vols = [store.get(p) for p in ids["test"]]
        te_keys = [unseen_keys(test_fold, p, R) for p in ids["test"]] if use_ens else None
        clean = scores_for_arrays(model, te_vols, None, device, T, cfg.train.mc_passes, maha, None)
        if use_ens:
            clean.update(ens_scores(te_vols, te_keys, get_model, get_T, device))
        fa = {k: float((clean[k] > norms[k].thr).mean()) for k in names}
        log(f"[OOD fold {f}] clean-test false-alarm rate: " + ", ".join(f"{k}={v:.2f}" for k, v in fa.items()))

        shifts = {}
        if len(young_ids):
            yv = [store.get(p) for p in young_ids]
            sc = scores_for_arrays(model, yv, None, device, T, cfg.train.mc_passes, maha, None)
            if use_ens:                                   # young adults were never in any training set: any R models are unseen
                sc.update(ens_scores(yv, [tuple((r, f) for r in range(R))] * len(yv), get_model, get_T, device))
            shifts[("younger_cohort", 0.0)] = sc
        rng = np.random.default_rng(cfg.cv.seed + f)
        sub = list(rng.choice(len(te_vols), size=min(int(cfg.ood.n_per_cell), len(te_vols)), replace=False))
        n_failed = 0
        for cname in cfg.ood.corruptions:
            for sev in cfg.ood.severities:
                vols, keys, ok_idx = [], [], []
                for j in sub:
                    try:
                        vols.append(corrupt(te_vols[j], cname, sev, seed=int(1000 * f + j)))
                        if use_ens: keys.append(te_keys[j])
                    except Exception as e:           # recorded, never silently replaced by the clean scan
                        n_failed += 1
                        if n_failed <= 3:
                            log(f"   corruption {cname} failed: {e}")
                if vols:
                    sc = scores_for_arrays(model, vols, None, device, T, cfg.train.mc_passes, maha, None)
                    if use_ens:
                        sc.update(ens_scores(vols, keys, get_model, get_T, device))
                    shifts[(cname, float(sev))] = sc
        for (name, sev), sc in shifts.items():
            for k in names:
                m = ood_metrics(norms[k].z(clean[k]), norms[k].z(sc[k]))
                rows.append({"fold": f, "shift": name, "severity": sev, "score": k, **m,
                             "detection_rate": float((sc[k] > norms[k].thr).mean()),
                             "false_alarm_clean": fa[k], "n_shifted": len(sc[k]), "n_clean": len(clean[k])})

        # familiar-but-difficult (misclassified in-distribution) vs correct vs truly unfamiliar
        y_te = np.array([labels[p] for p in ids["test"]])
        wrong = clean["probs"].argmax(1) != y_te
        if wrong.sum() >= 2 and (~wrong).sum() >= 2:
            for k in names:
                z_ok, z_bad = norms[k].z(clean[k][~wrong]), norms[k].z(clean[k][wrong])
                rows.append({"fold": f, "shift": "hard_familiar(misclassified)_vs_correct", "severity": 0.0, "score": k,
                             **ood_metrics(z_ok, z_bad), "detection_rate": float((clean[k][wrong] > norms[k].thr).mean()),
                             "false_alarm_clean": fa[k], "n_shifted": int(wrong.sum()), "n_clean": int((~wrong).sum())})
                if ("younger_cohort", 0.0) in shifts:
                    zy = norms[k].z(shifts[("younger_cohort", 0.0)][k])
                    rows.append({"fold": f, "shift": "younger_vs_hard_familiar", "severity": 0.0, "score": k,
                                 **ood_metrics(z_bad, zy), "detection_rate": float((shifts[("younger_cohort", 0.0)][k] > norms[k].thr).mean()),
                                 "false_alarm_clean": float((clean[k][wrong] > norms[k].thr).mean()),
                                 "n_shifted": len(zy), "n_clean": int(wrong.sum())})
        if n_failed:
            log(f"   WARNING: {n_failed} corruptions failed in fold {f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ood_results.csv"), index=False)
    return df


def plot_ood(df, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    syn = df[~df["shift"].isin(["younger_cohort", "hard_familiar(misclassified)_vs_correct", "younger_vs_hard_familiar"])]
    if syn.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in syn["score"].unique():
        g = syn[syn["score"] == k].groupby("severity")["AUROC"].mean()
        ax[0].plot(g.index, g.values, marker="o", label=k)
        d = syn[syn["score"] == k].groupby("severity")["detection_rate"].mean()
        ax[1].plot(d.index, d.values, marker="o", label=k)
    fa = syn.groupby("score")["false_alarm_clean"].mean()
    ax[0].axhline(0.5, color="k", ls=":"); ax[0].set_xlabel("corruption severity"); ax[0].set_ylabel("AUROC vs clean scans"); ax[0].set_title("Corruption detection (mean over corruption types)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("corruption severity"); ax[1].set_ylabel("share flagged"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title("Detection rate at calibration-set threshold\n(clean false alarms: " + ", ".join(f"{k[:8]}={v:.2f}" for k, v in fa.items()) + ")", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
