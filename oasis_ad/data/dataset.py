"""Volumes are small (88x104x88 float16 = 1.6 MB), so the whole cohort is kept in RAM: fast and simple."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VolumeStore:
    """variant: 'full' (normal input) | 'mask_only' (brain silhouette only - tests how much size/shape alone predicts)
    | 'eroded' (outer 3-voxel brain shell removed - tests reliance on the brain boundary / skull-stripping edge)."""
    def __init__(self, cache_dir, ids, with_mask=True, variant="full"):
        self.cache_dir = cache_dir
        self.variant = variant
        self._eroded = {}
        self.vols, self.masks = {}, {}
        for pid in ids:
            self.add(pid, with_mask)

    def add(self, pid, with_mask=True):
        if pid in self.vols:
            return
        self.vols[pid] = np.load(os.path.join(self.cache_dir, f"{pid}_vol.npy"))           # float16
        if with_mask:
            m = os.path.join(self.cache_dir, f"{pid}_mask.npy")
            self.masks[pid] = np.load(m).astype(bool) if os.path.exists(m) else (self.vols[pid] != 0)

    def get(self, pid):
        v = self.vols[pid].astype(np.float32)
        if self.variant == "full":
            return v
        m = self.masks[pid]
        if self.variant == "mask_only":
            return m.astype(np.float32) * 2.0 - 1.0
        if self.variant == "eroded":
            if pid not in self._eroded:
                from scipy import ndimage
                self._eroded[pid] = ndimage.binary_erosion(m, iterations=3)
            return v * self._eroded[pid]
        raise ValueError(f"unknown input variant {self.variant}")

    def get_mask(self, pid):
        return self.masks[pid]

    def seg(self, pid):
        p = os.path.join(self.cache_dir, f"{pid}_seg.npy")
        return np.load(p) if os.path.exists(p) else None


def _shift(v, s):
    """Integer translation with zero fill (no wrap-around)."""
    out = np.zeros_like(v)
    src, dst = [], []
    for k, d in enumerate(s):
        n = v.shape[k]
        if d >= 0:
            src.append(slice(0, n - d)); dst.append(slice(d, n))
        else:
            src.append(slice(-d, n)); dst.append(slice(0, n + d))
    out[tuple(dst)] = v[tuple(src)]
    return out


def augment(vol, mask, a, rng):
    """Training-only augmentation. No flips (axis/orientation of the Analyze files is not assumed),
    small translations, global intensity scaling and Gaussian noise INSIDE the brain mask only."""
    if rng.random() > a["p"]:
        return vol
    s = [int(rng.integers(-a["shift_vox"], a["shift_vox"] + 1)) for _ in range(3)]
    vol = _shift(vol, s)
    mask = _shift(mask.astype(np.uint8), s).astype(bool)
    vol = vol * float(rng.uniform(1 - a["intensity_scale"], 1 + a["intensity_scale"]))
    if a["noise_std"] > 0:
        vol = vol + (rng.normal(0, a["noise_std"], vol.shape).astype(np.float32) * mask)
    return vol


class VolDataset(Dataset):
    def __init__(self, store, ids, labels, aug_cfg=None, seed=0):
        self.store, self.ids, self.labels = store, list(ids), labels
        self.aug = dict(aug_cfg) if aug_cfg else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        v = self.store.get(pid)
        if self.aug:
            v = augment(v, self.store.get_mask(pid), self.aug, self.rng)
        y = int(self.labels[pid]) if (self.labels is not None and pid in self.labels) else -1
        return torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0), y
