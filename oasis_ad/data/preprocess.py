"""Preprocessing, quality control and caching.

Input: the processed OASIS-1 volume (bias-field corrected, atlas-registered, brain-masked, T88_111 grid).
Because those steps were already done by the dataset authors we do NOT run N4 / skull-stripping again.

preprocess_volume() is the single function used both in training and in the web app, so there is
no train/inference mismatch.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import ndimage

from ..utils import log, save_json, load_json
from .oasis_io import load_volume


# ─────────────────────────────── core transform ───────────────────────────────
def preprocess_volume(vol, target_shape):
    """raw masked volume -> (z-scored resized volume float32, brain mask bool).
    Aspect ratio is preserved when target_shape is a uniform scaling of the source grid."""
    vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = vol > 0
    f = [t / s for t, s in zip(target_shape, vol.shape)]
    v = ndimage.zoom(vol, f, order=1)
    m = ndimage.zoom(mask.astype(np.float32), f, order=1) > 0.5
    out = np.zeros(target_shape, dtype=np.float32)
    # zoom can be off by one voxel on odd sizes -> crop/pad to exact target
    sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, target_shape))
    tmp = np.zeros(target_shape, dtype=np.float32); tmp[sl] = v[sl]
    mm = np.zeros(target_shape, dtype=bool); mm[sl] = m[sl]
    vox = tmp[mm]
    if vox.size > 10:
        out[mm] = (vox - vox.mean()) / (vox.std() + 1e-6)
    return out, mm


def resize_labels(seg, target_shape):
    f = [t / s for t, s in zip(target_shape, seg.shape)]
    z = ndimage.zoom(seg.astype(np.uint8), f, order=0)
    out = np.zeros(target_shape, dtype=np.uint8)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(z.shape, target_shape))
    out[sl] = z[sl]
    return out


# ─────────────────────────────── QC ───────────────────────────────
QC_METRICS = ["brain_frac", "mean_in", "cov_in", "p99_over_p50", "grad_energy"]


def raw_qc_metrics(vol, expected_shape):
    """Cheap, interpretable image-quality indicators computed on the raw masked volume."""
    d = {"shape": "x".join(map(str, vol.shape)),
         "shape_ok": bool(tuple(vol.shape) == tuple(expected_shape)),
         "finite_frac": float(np.isfinite(vol).mean())}
    v = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m = v > 0
    d["brain_frac"] = float(m.mean())
    if m.sum() < 100:
        d.update({k: np.nan for k in QC_METRICS if k != "brain_frac"})
        return d
    x = v[m]
    d["mean_in"] = float(x.mean())
    d["cov_in"] = float(x.std() / (x.mean() + 1e-6))
    d["p99_over_p50"] = float(np.percentile(x, 99) / (np.percentile(x, 50) + 1e-6))
    gx, gy, gz = np.gradient(v)
    g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    inner = ndimage.binary_erosion(m, iterations=2)
    d["grad_energy"] = float(g[inner].mean() / (x.mean() + 1e-6)) if inner.any() else np.nan   # blur / motion proxy
    return d


def qc_flags(qc, qc_cfg, stats=None):
    """Hard failures + soft outliers (robust z vs the cohort). Returns (flag_hard, flag_soft, reasons)."""
    reasons = []
    if not qc.get("shape_ok", False):
        reasons.append(f"unexpected shape {qc.get('shape')}")
    if qc.get("finite_frac", 1.0) < 1.0:
        reasons.append("non-finite voxels")
    lo, hi = qc_cfg.brain_frac_range
    bf = qc.get("brain_frac", np.nan)
    if not (lo <= bf <= hi):
        reasons.append(f"brain fraction {bf:.3f} outside [{lo}, {hi}]")
    hard = len(reasons) > 0
    soft = False
    if stats and not hard:
        for k in QC_METRICS:
            s = stats.get(k)
            if s and qc.get(k) is not None and np.isfinite(qc[k]):
                scale = s["mad"] * 1.4826
                if scale < 1e-3:
                    scale = 0.05
                z = (qc[k] - s["median"]) / scale
                if abs(z) > qc_cfg.robust_z_limit:
                    soft = True
                    reasons.append(f"{k} robust-z={z:.1f}")
    return hard, soft, "; ".join(reasons)


def compute_qc_stats(qc_df):
    ok = qc_df[qc_df["shape_ok"] & (qc_df["finite_frac"] >= 1.0)]
    stats = {}
    for k in QC_METRICS:
        x = ok[k].dropna().values
        if len(x) >= 5:
            med = float(np.median(x))
            stats[k] = {"median": med, "mad": float(np.median(np.abs(x - med)))}
    return stats


# ─────────────────────────────── cache building ───────────────────────────────
def build_cache(manifest, cfg, paths, force=False):
    """Process every labelled + young participant once and save to the Drive cache. Resumable."""
    target = tuple(cfg.preprocess.target_shape)
    exp = tuple(cfg.data.expected_shape)
    cd = paths.cache_drive
    qdir = os.path.join(cd, "qc_json"); os.makedirs(qdir, exist_ok=True)
    todo = manifest[manifest["role"].isin(["labeled", "ood_young"])]
    failed = []
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        pid = r["participant_id"]
        f_vol = os.path.join(cd, f"{pid}_vol.npy")
        f_qc = os.path.join(qdir, f"{pid}.json")
        if os.path.exists(f_vol) and os.path.exists(f_qc) and not force:
            continue
        try:
            raw = load_volume(r["t88_img"])
            qc = raw_qc_metrics(raw, exp)
            qc["participant_id"] = pid
            if not qc["shape_ok"]:
                # still cache a resized copy so that the flagged scan can be inspected, but QC marks it as failed
                pass
            vol, mask = preprocess_volume(raw, target)
            np.save(f_vol + ".tmp.npy", vol.astype(np.float16)); os.replace(f_vol + ".tmp.npy", f_vol)
            np.save(os.path.join(cd, f"{pid}_mask.npy"), mask.astype(np.uint8))
            if isinstance(r.get("seg_img"), str) and os.path.exists(r["seg_img"]):
                try:
                    seg = load_volume(r["seg_img"]).round().astype(np.uint8)
                    np.save(os.path.join(cd, f"{pid}_seg.npy"), resize_labels(seg, target))
                except Exception as e:
                    log(f"  segmentation skipped for {pid}: {e}")
            save_json(qc, f_qc)
        except Exception as e:
            failed.append((pid, str(e)))
            save_json({"participant_id": pid, "load_error": str(e), "shape_ok": False,
                       "finite_frac": 0.0, "brain_frac": 0.0}, f_qc)
            log(f"  FAILED {pid}: {e}")
        if i % 25 == 0:
            log(f"  processed {i}/{len(todo)}")
    return failed


def compile_qc(manifest, cfg, paths):
    """Collect per-participant QC json -> qc_report.csv, qc_stats.json and the QC-passed manifest."""
    cd = paths.cache_drive
    rows = []
    for pid in manifest.loc[manifest["role"].isin(["labeled", "ood_young"]), "participant_id"]:
        f = os.path.join(cd, "qc_json", f"{pid}.json")
        rows.append(load_json(f) if os.path.exists(f) else {"participant_id": pid, "load_error": "not processed",
                                                           "shape_ok": False, "finite_frac": 0.0, "brain_frac": 0.0})
    qc = pd.DataFrame(rows)
    for c in QC_METRICS + ["finite_frac"]:
        if c not in qc:
            qc[c] = np.nan
    if "shape_ok" not in qc:
        qc["shape_ok"] = False
    stats = compute_qc_stats(qc)
    hard, soft, why = [], [], []
    for _, r in qc.iterrows():
        h, s, w = qc_flags(r.to_dict(), cfg.qc, stats)
        if isinstance(r.get("load_error"), str):
            h, w = True, f"load error: {r['load_error']}"
        hard.append(h); soft.append(s); why.append(w)
    qc["qc_hard_fail"], qc["qc_soft_flag"], qc["qc_reasons"] = hard, soft, why
    qc.to_csv(os.path.join(cd, "qc_report.csv"), index=False)
    save_json(stats, os.path.join(cd, "qc_stats.json"))
    m = manifest.merge(qc[["participant_id", "qc_hard_fail", "qc_soft_flag", "qc_reasons"]], on="participant_id", how="left")
    passed = m[m["role"].isin(["labeled", "ood_young"]) & (~m["qc_hard_fail"].fillna(True).astype(bool))].copy()
    # soft-flagged scans are kept but marked (reviewed visually); hard failures are excluded and logged
    passed.to_csv(os.path.join(cd, "manifest_qc_passed.csv"), index=False)
    return qc, passed, stats


def sync_cache_to_local(paths):
    """Copy the Drive cache to the fast local disk (once per Colab session)."""
    import shutil
    src, dst = paths.cache_drive, paths.cache_local
    n = 0
    for f in os.listdir(src):
        if f.endswith(".npy") or f.endswith(".csv") or f.endswith(".json"):
            d = os.path.join(dst, f)
            if not os.path.exists(d):
                shutil.copy2(os.path.join(src, f), d); n += 1
    return n


def spot_check_figure(passed, paths, n=4, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    ids = passed.loc[passed["role"] == "labeled", "participant_id"].tolist()
    ids = list(rng.choice(ids, size=min(n, len(ids)), replace=False))
    fig, ax = plt.subplots(len(ids), 3, figsize=(9, 3 * len(ids)), squeeze=False)
    for r, pid in enumerate(ids):
        v = np.load(os.path.join(paths.cache_local, f"{pid}_vol.npy")).astype(np.float32)
        c = [s // 2 for s in v.shape]
        for k, (sl, t) in enumerate(((v[c[0]], "axis0"), (v[:, c[1]], "axis1"), (v[:, :, c[2]], "axis2"))):
            ax[r, k].imshow(sl.T, cmap="gray", origin="lower"); ax[r, k].set_title(f"{pid} {t}", fontsize=8); ax[r, k].axis("off")
    fig.tight_layout()
    p = os.path.join(paths.cache_drive, "spot_check.png"); fig.savefig(p, dpi=100); plt.close(fig)
    return p


def build_tissue_features(passed, paths):
    """Image-derived baseline features from the FSL tissue segmentation cache: CSF / GM / WM fractions of the brain mask.
    Assumes OASIS-1 fseg labels 0=background, 1=CSF, 2=grey matter, 3=white matter - CHECK this against the OASIS documentation."""
    rows, odd = [], set()
    for pid in passed.loc[passed["role"] == "labeled", "participant_id"]:
        fs, fm = os.path.join(paths.cache_drive, f"{pid}_seg.npy"), os.path.join(paths.cache_drive, f"{pid}_mask.npy")
        if not (os.path.exists(fs) and os.path.exists(fm)):
            continue
        seg, m = np.load(fs), np.load(fm).astype(bool)
        odd |= set(np.unique(seg).tolist()) - {0, 1, 2, 3}
        n = max(int(m.sum()), 1)
        rows.append({"participant_id": pid, "csf_frac": float(((seg == 1) & m).sum() / n),
                     "gm_frac": float(((seg == 2) & m).sum() / n), "wm_frac": float(((seg == 3) & m).sum() / n)})
    df = pd.DataFrame(rows)
    if odd:
        log(f"WARNING: unexpected segmentation labels {sorted(odd)} - check the tissue-label convention")
    df.to_csv(os.path.join(paths.cache_drive, "tissue_features.csv"), index=False)
    return df
