"""Ablation study: which design choices help, and where do they fail?

Every variant is trained with the SAME outer splits (first `ablation.n_repeats` CV repeats) and compared with the main
model's results for the same repeats. Two variants are shortcut probes:
  * mask_only_input        - the network sees only the brain silhouette (no intensities). If it does about as well as the
                             full model, size/shape (or skull-stripping edges) carries most of the signal.
  * eroded_boundary_input  - the outer 3-voxel brain shell is removed. A big drop means reliance on the brain boundary.
"""
import os, copy
import numpy as np
import pandas as pd

from .config import Paths
from .utils import log, get_device, save_json

VARIANTS = {
    "no_augmentation":       {"model": "cnn3d",    "overrides": {"train.augment.p": 0.0}},
    "no_class_weighting":    {"model": "cnn3d",    "overrides": {"train.class_weighting": False}},
    "mask_only_input":       {"model": "cnn3d",    "overrides": {"train.input_variant": "mask_only"}},
    "eroded_boundary_input": {"model": "cnn3d",    "overrides": {"train.input_variant": "eroded"}},
    "cnn2d_slices":          {"model": "cnn2d",    "overrides": {}},
    "resnet10_scratch":      {"model": "resnet10", "overrides": {"models.resnet10.pretrained": False}},
    "resnet10_medicalnet":   {"model": "resnet10", "overrides": {"models.resnet10.pretrained": True}, "needs": "medicalnet_weights"},
}


class AblationPaths(Paths):
    """Own model/evaluation folders, but shares cache, splits and baselines with the main study."""
    def __init__(self, cfg_variant, main, variant):
        cfg_variant.paths.out_dir = os.path.join(main.out_dir, "step10_ablations", variant)
        super().__init__(cfg_variant)
        self._main = main

    @property
    def audit(self): return self._main.audit
    @property
    def cache_drive(self): return self._main.cache_drive
    @property
    def cache_local(self): return self._main.cache_local
    @property
    def splits(self): return self._main.splits
    @property
    def baselines(self): return self._main.baselines


def _set(cfg, dotted, value):
    d = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def run_ablations(cfg, paths, main_model="cnn3d", variants=None, repeats=None):
    from . import pipeline as PL
    from . import evaluation as E, uncertainty as U
    from .train import run_cv
    variants = list(variants or cfg.ablation.variants)
    n_rep = int(repeats or cfg.ablation.n_repeats)
    lab, sp = PL.load_labeled(paths), PL.load_splits(paths)
    dev = get_device()
    reps = list(range(min(n_rep, cfg.cv.n_repeats)))

    # reference: main model, same repeats
    ref_df = E.load_fold_outputs(paths.models(main_model))
    ref_df = ref_df[ref_df["repeat"].isin(reps)]
    sets = {"full model (reference)": E.to_predset(ref_df, "full model (reference)", lab)}
    cal = {"full model (reference)": E.calibration_report(ref_df)[0]}

    for v in variants:
        spec = VARIANTS[v]
        if spec.get("needs") and not cfg.paths.get(spec["needs"]):
            log(f"[ablation {v}] skipped: paths.{spec['needs']} not set"); continue
        cv = copy.deepcopy(cfg)
        for k, val in spec["overrides"].items():
            _set(cv, k, val)
        pv = AblationPaths(cv, paths, v)
        store = PL.make_store(cv, pv, include_young=False)
        log(f"[ablation {v}] model={spec['model']} overrides={spec['overrides']}")
        run_cv(spec["model"], cv, pv, lab, sp, store, dev, repeats=reps)
        df = E.load_fold_outputs(pv.models(spec["model"]))
        df = df[df["repeat"].isin(reps)]
        sets[v] = E.to_predset(df, v, lab)
        cal[v] = E.calibration_report(df)[0]

    ref = sets["full model (reference)"]
    rows = []
    for name, ps in sets.items():
        s = U.bootstrap_participants(ps.y, ps.P_list, U.metrics_from_probs, n_boot=cfg.eval.n_boot)
        c = cal[name]; c = c[c["variant"] == "temperature-scaled"].iloc[0]
        row = {"variant": name, "balanced_acc": round(s["balanced_acc"]["mean"], 3),
               "balanced_acc_CI": f"[{s['balanced_acc']['ci_lo']:.2f}, {s['balanced_acc']['ci_hi']:.2f}]",
               "auroc": round(s["auroc"]["mean"], 3), "auroc_CI": f"[{s['auroc']['ci_lo']:.2f}, {s['auroc']['ci_hi']:.2f}]",
               "ECE_after_T": round(float(c["ECE_equal_width"]), 3), "Brier_after_T": round(float(c["Brier"]), 3)}
        if name != "full model (reference)" and cfg.task.mode == "binary":
            d = U.paired_bootstrap_diff(ref.y, ps.P_list, ref.P_list, "auroc", cfg.eval.n_boot)
            row["AUROC_diff_vs_full"] = round(d["diff"], 3)
            row["AUROC_diff_CI"] = f"[{d['ci_lo']:.2f}, {d['ci_hi']:.2f}]"
        rows.append(row)
    tab = pd.DataFrame(rows)
    out = os.path.join(paths.out_dir, "step10_ablations"); os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "ablation_table.csv"), index=False)
    return tab
