"""Qualitative saliency (Grad-CAM) WITH sanity checks. Maps are for inspection only - not a validated biomarker.

Checks implemented:
  1. cascading weight randomisation (Adebayo et al. 2018): re-initialise layers from output to input;
     a trustworthy map must change (correlation with the original map must drop)
  2. stability across independently trained models (different CV repeats) for the same participant
  3. descriptive overlap of the map with tissue classes (CSF / GM / WM) from the FSL segmentation
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def gradcam(model, vol, class_idx=None, device="cpu"):
    """vol: (D,H,W) float32 preprocessed. Returns (cam (D,H,W) in [0,1], predicted class, probs)."""
    if not getattr(model, "has_feature_map", False):
        raise ValueError("Grad-CAM needs a model with a 3-D feature map (cnn3d / resnet10).")
    model.eval()
    x = torch.from_numpy(vol).float()[None, None].to(device)
    with torch.enable_grad():
        fm = model.feature_map(x)
        fm.retain_grad()
        logits, _ = model.classify(fm)
        c = int(logits.argmax(1).item()) if class_idx is None else int(class_idx)
        model.zero_grad(set_to_none=True)
        logits[0, c].backward()
    w = fm.grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((w * fm).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=vol.shape, mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, c, torch.softmax(logits.detach(), 1)[0].cpu().numpy()


def _reinit(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _layer_groups(model):
    """Output -> input ordered list of (name, module)."""
    g = [("head", model.head)]
    if hasattr(model, "stages"):
        g += [(f"stage{i}", s) for i, s in reversed(list(enumerate(model.stages)))]
    else:
        g += [(n, getattr(model, n)) for n in ("layer4", "layer3", "layer2", "layer1")]
        g += [("conv1", model.conv1)]
    return g


def _corr(a, b, mask):
    a, b = a[mask], b[mask]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cascading_randomization(model, vol, mask, device="cpu"):
    base, c, _ = gradcam(model, vol, device=device)
    m = copy.deepcopy(model)
    rows = [{"randomised_up_to": "none (original)", "pearson_with_original": 1.0}]
    for name, mod in _layer_groups(model):
        for n2, mod2 in _layer_groups(m):
            if n2 == name:
                _reinit(mod2)
        cam, _, _ = gradcam(m, vol, class_idx=c, device=device)
        rows.append({"randomised_up_to": name, "pearson_with_original": _corr(base, cam, mask)})
    return rows


def tissue_overlap(cam, seg, thr=0.5):
    """Share of the high-activation region (cam >= thr) falling in each tissue class (descriptive only)."""
    if seg is None:
        return None
    hot = cam >= thr
    if hot.sum() == 0:
        return None
    tot = hot.sum()
    return {"CSF": float(((seg == 1) & hot).sum() / tot), "GM": float(((seg == 2) & hot).sum() / tot),
            "WM": float(((seg == 3) & hot).sum() / tot), "other": float(((seg == 0) & hot).sum() / tot)}


def overlay_png(vol, cam, path=None, title=""):
    """3-plane overlay figure. Returns matplotlib figure (caller saves/encodes)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # centre planes on the CAM maximum so that the highlighted region is visible
    c = np.unravel_index(np.argmax(cam), cam.shape) if cam.max() > 0 else tuple(s // 2 for s in cam.shape)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for a, (sl, cs, nm) in zip(ax, ((vol[c[0]], cam[c[0]], "plane A"), (vol[:, c[1]], cam[:, c[1]], "plane B"), (vol[:, :, c[2]], cam[:, :, c[2]], "plane C"))):
        a.imshow(sl.T, cmap="gray", origin="lower"); a.imshow(np.ma.masked_less(cs.T, 0.15), cmap="jet", alpha=0.45, origin="lower", vmin=0, vmax=1)
        a.set_title(nm, fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def pick_examples(pred_df, n_each=2, n_random=4, seed=0):
    """Balanced selection (TP/TN/FP/FN + random) so the gallery is NOT cherry-picked by confidence."""
    d = pred_df.copy()
    P = d[[c for c in d.columns if c.startswith("p_cal_")]].values
    d["pred"] = P.argmax(1); d["y"] = d["label"].astype(int)
    rng = np.random.default_rng(seed)
    chosen = []
    for name, m in (("TP", (d.pred > 0) & (d.y > 0)), ("TN", (d.pred == 0) & (d.y == 0)),
                    ("FP", (d.pred > 0) & (d.y == 0)), ("FN", (d.pred == 0) & (d.y > 0))):
        ids = d.loc[m, "participant_id"].tolist()
        for pid in rng.permutation(ids)[:n_each]:
            chosen.append((pid, name))
    rest = [p for p in d["participant_id"] if p not in {c[0] for c in chosen}]
    for pid in rng.permutation(rest)[:n_random]:
        chosen.append((pid, "random"))
    return chosen


def stability_across_models(models, vol, mask, device="cpu"):
    cams = [gradcam(m, vol, device=device)[0] for m in models]
    vals = [_corr(cams[i], cams[j], mask) for i in range(len(cams)) for j in range(i + 1, len(cams))]
    return {"pairwise_pearson": vals, "mean": float(np.mean(vals)) if vals else np.nan}


def perturbation_stability(model, vol, mask, device="cpu", seed=0):
    """Grad-CAM (for the originally predicted class) on small, plausible input perturbations vs the original map.
    Returns {perturbation: {"pearson": r, "pred_changed": bool}}. Shifted maps are shifted back before comparing."""
    from .data.dataset import _shift
    rng = np.random.default_rng(seed)
    base, c, _ = gradcam(model, vol, device=device)
    out = {}
    s = (2, 2, 2)
    perts = {
        "noise (sd 0.1)": (lambda v: v + rng.normal(0, 0.1, v.shape).astype(np.float32) * mask, None),
        "shift 2 voxels": (lambda v: _shift(v, s), s),
        "intensity x1.05": (lambda v: v * 1.05, None),
        "smooth bias field": (lambda v: v * (1 + 0.05 * np.linspace(-1, 1, v.shape[0], dtype=np.float32)[:, None, None]), None),
    }
    for name, (fn, sh) in perts.items():
        cam2, _, pr = gradcam(model, fn(vol).astype(np.float32), class_idx=c, device=device)
        if sh is not None:
            cam2 = _shift(cam2, tuple(-x for x in sh))
        out[name] = {"pearson": _corr(base, cam2, mask), "pred_changed": bool(pr.argmax() != c)}
    return out
