"""One function per pipeline stage. The notebooks are thin wrappers around these; tests/smoke_test.py runs them all."""
import os, glob
import numpy as np
import pandas as pd

from .utils import log, save_json, load_json, set_seed, get_device
from .config import Paths
from .data import oasis_io, audit as A, preprocess as P, splits as SP
from .data.dataset import VolumeStore


# ───────────────────────── loaders ─────────────────────────
def load_manifest_full(paths):
    return pd.read_csv(os.path.join(paths.audit, "manifest.csv"))


def load_passed(paths):
    return pd.read_csv(os.path.join(paths.cache_drive, "manifest_qc_passed.csv"))


def load_labeled(paths):
    m = load_passed(paths)
    lab = m[m["role"] == "labeled"].copy()
    lab["label"] = lab["label"].astype(int)
    return lab.reset_index(drop=True)


def load_splits(paths):
    return pd.read_csv(os.path.join(paths.splits, "splits.csv"))


# ───────────────────────── step 1 ─────────────────────────
def step1_audit(cfg, paths):
    clin_path = oasis_io.find_clinical_file(paths.data_dir, cfg.paths.clinical_file)
    log(f"clinical file: {clin_path}")
    clin = oasis_io.read_clinical(clin_path)
    idx = oasis_io.get_file_index(paths, cfg, extract=True)
    log(f"found processed volumes for {int(idx['t88_img'].notna().sum())} sessions "
        f"({int(idx['seg_img'].notna().sum())} with tissue segmentation)")
    manifest = A.build_manifest(clin, idx, cfg)
    manifest.to_csv(os.path.join(paths.audit, "manifest.csv"), index=False)
    idx.to_csv(os.path.join(paths.audit, "file_index.csv"), index=False)
    rep = A.audit_report(clin, manifest, cfg, paths.audit)
    A.plot_age_confounding(manifest, paths.audit)
    return manifest, rep


# ───────────────────────── step 2 ─────────────────────────
def step2_preprocess(cfg, paths, force=False):
    manifest = load_manifest_full(paths)
    todo = manifest[manifest["role"].isin(["labeled", "ood_young"])]
    missing = [p for p in todo["participant_id"] if not os.path.exists(os.path.join(paths.cache_drive, f"{p}_vol.npy"))]
    if missing or force:
        log(f"{len(missing)} participants still need preprocessing -> making sure scans are extracted")
        idx = oasis_io.get_file_index(paths, cfg, extract=True)
        manifest = manifest.drop(columns=["t88_img", "seg_img"], errors="ignore").merge(idx, on="session_id", how="left")
        failed = P.build_cache(manifest, cfg, paths, force=force)
        if failed:
            log(f"{len(failed)} scans failed to load (logged in QC report)")
    qc, passed, stats = P.compile_qc(manifest, cfg, paths)
    n = P.sync_cache_to_local(paths)
    log(f"QC: {int(qc['qc_hard_fail'].sum())} hard failures, {int(qc['qc_soft_flag'].sum())} soft flags; "
        f"cohort after QC: {int((passed['role']=='labeled').sum())} labelled, {int((passed['role']=='ood_young').sum())} younger")
    P.spot_check_figure(passed, paths)
    tf = P.build_tissue_features(passed, paths)
    log(f"tissue features (CSF/GM/WM fractions) computed for {len(tf)} participants")
    return qc, passed


# ───────────────────────── step 3 ─────────────────────────
def step3_splits(cfg, paths):
    lab = load_labeled(paths)
    sp = SP.make_outer_splits(lab, cfg)
    sp.to_csv(os.path.join(paths.splits, "splits.csv"), index=False)
    summ = SP.split_summary(sp, lab)
    summ.to_csv(os.path.join(paths.splits, "split_summary.csv"), index=False)
    return sp, summ


# ───────────────────────── step 4 ─────────────────────────
def step4_baselines(cfg, paths):
    from . import baselines as B
    lab, sp = load_labeled(paths), load_splits(paths)
    tf_path = os.path.join(paths.cache_drive, "tissue_features.csv")
    tissue = pd.read_csv(tf_path) if os.path.exists(tf_path) else None
    names = B.run_baselines(lab, sp, cfg, paths, tissue)
    pairs = B.age_matched_ids(lab, cfg.eval.age_match_caliper, cfg.cv.seed)
    pairs.to_csv(os.path.join(paths.baselines, "age_matched_pairs.csv"), index=False)
    log(f"age-matched pairs: {len(pairs)} (caliper {cfg.eval.age_match_caliper} y)")
    return names, pairs


# ───────────────────────── step 5 ─────────────────────────
def make_store(cfg, paths, include_young=True):
    P.sync_cache_to_local(paths)
    m = load_passed(paths)
    roles = ["labeled", "ood_young"] if include_young else ["labeled"]
    return VolumeStore(paths.cache_local, m.loc[m["role"].isin(roles), "participant_id"].tolist(),
                       variant=cfg.train.get("input_variant", "full"))


def step5_train(cfg, paths, models=None, repeats=None, folds=None):
    from .train import run_cv
    lab, sp = load_labeled(paths), load_splits(paths)
    store = make_store(cfg, paths, include_young=False)
    dev = get_device()
    log(f"device: {dev}")
    for name in (models or cfg.train.models):
        run_cv(name, cfg, paths, lab, sp, store, dev, repeats, folds)


# ───────────────────────── step 6 ─────────────────────────
def step6_evaluate(cfg, paths, model_name, include_baselines=True):
    from . import evaluation as E
    lab = load_labeled(paths)
    ed = os.path.join(paths.evaluation, model_name); os.makedirs(ed, exist_ok=True)
    df = E.load_fold_outputs(paths.models(model_name))
    cnn = E.to_predset(df, model_name, lab)
    ens = E.ensemble_predset(cnn)
    base_sets = []
    if include_baselines:
        for d in sorted(glob.glob(os.path.join(paths.baselines, "*", "fold_outputs.csv"))):
            base_sets.append(E.to_predset(pd.read_csv(d), os.path.basename(os.path.dirname(d)), lab))
    sets = [cnn, ens] + base_sets
    comp = E.compare_models(sets, cfg.eval.n_boot, ed)
    res = {"model_comparison": comp}
    cal_mean, cal_rep = E.calibration_report(df)
    cal_mean.to_csv(os.path.join(ed, "calibration_summary.csv"), index=False); res["calibration"] = cal_mean
    res["uncertainty_vs_error"] = E.uncertainty_vs_error(df)
    E.plot_reliability(df, os.path.join(ed, "reliability.png"))
    conf, per_part = E.conformal_report(df, lab, cfg)
    per_part.to_csv(os.path.join(ed, "conformal_per_participant.csv"), index=False)
    save_json(conf, os.path.join(ed, "conformal_report.json")); res["conformal"] = conf
    # age control
    pairs_f = os.path.join(paths.baselines, "age_matched_pairs.csv")
    if os.path.exists(pairs_f):
        res["age_matched"] = E.age_matched_metrics(sets, pd.read_csv(pairs_f), cfg.eval.n_boot)
        res["age_matched"].to_csv(os.path.join(ed, "age_matched_metrics.csv"), index=False)
    age = cnn.meta["age"].values.astype(float)
    res["age_dependence"] = E.age_dependence(cnn, age)
    res["incremental_over_age"] = E.incremental_value_over_age(cnn, age)
    base = {s.name: s for s in base_sets}
    res["paired_vs_baselines"] = {}
    if cfg.task.mode == "binary":
        for b in ("majority__prior", "age_sex__logreg", "volume_age_sex__rf", "volume_age_sex__logreg", "tissue_age_sex__logreg"):
            if b in base:
                res["paired_vs_baselines"][b] = E.U.paired_bootstrap_diff(cnn.y, cnn.P_list, base[b].P_list, "auroc", cfg.eval.n_boot)
    # subgroup performance: CNN, deep ensemble and the strongest simple baselines
    pick = [cnn, ens] + [base[b] for b in ("age_sex__logreg", "volume_age_sex__logreg") if b in base]
    res["subgroups"] = E.subgroup_table(pick, [tuple(b) for b in cfg.eval.age_bands], n_boot=min(cfg.eval.n_boot, 300))
    res["subgroups"].to_csv(os.path.join(ed, "subgroup_performance.csv"), index=False)
    save_json({k: (v.to_dict("records") if hasattr(v, "to_dict") else v) for k, v in res.items()}, os.path.join(ed, "evaluation_summary.json"))
    return res


# ───────────────────────── step 7 ─────────────────────────
def step7_ood(cfg, paths, model_name):
    from . import ood
    lab, sp, passed = load_labeled(paths), load_splits(paths), load_passed(paths)
    young = passed.loc[passed["role"] == "ood_young", "participant_id"].tolist()
    store = make_store(cfg, paths, include_young=True)
    df = ood.run_ood_analysis(model_name, cfg, paths, lab, young, sp, store, get_device())
    ood.plot_ood(df, os.path.join(paths.ood, model_name, "ood_summary.png"))
    return df


# ───────────────────────── step 8 ─────────────────────────
def step8_selective_saliency(cfg, paths, model_name, n_examples=12, max_stability=3):
    import torch
    from . import evaluation as E, saliency as S
    from .train import load_model_for_fold
    from .data.splits import get_ids
    out = os.path.join(paths.selective, model_name); os.makedirs(out, exist_ok=True)
    df = E.load_fold_outputs(paths.models(model_name))
    sel, folds = E.selective_report(df, cfg)
    save_json(sel, os.path.join(out, "selective_report.json")); folds.to_csv(os.path.join(out, "selective_per_fold.csv"), index=False)
    E.plot_risk_coverage(df, os.path.join(out, "risk_coverage.png"))
    res = {"selective": sel}
    lab, sp = load_labeled(paths), load_splits(paths)
    dev = get_device()
    store = make_store(cfg, paths, include_young=False)
    probe = load_model_for_fold(model_name, cfg, paths, 0, 0, dev)
    if not getattr(probe, "has_feature_map", False):
        log("model has no 3-D feature map -> Grad-CAM skipped"); return res
    t0 = df[(df["role"] == "test") & (df["repeat"] == 0)]
    chosen = S.pick_examples(t0, n_each=2, n_random=max(0, n_examples - 8), seed=cfg.cv.seed)
    rows, rand_rows, stab_rows = [], [], []
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, (pid, kind) in enumerate(chosen):
        fold = int(sp[(sp.repeat == 0) & (sp.role == "test") & (sp.participant_id == pid)]["fold"].iloc[0])
        m = load_model_for_fold(model_name, cfg, paths, 0, fold, dev)
        vol, mask = store.get(pid), store.get_mask(pid)
        cam, c, pr = S.gradcam(m, vol, device=dev)
        fig = S.overlay_png(vol, cam, title=f"{pid} [{kind}] predicted class {c} (p={pr[c]:.2f})")
        fig.savefig(os.path.join(out, f"gradcam_{k:02d}_{pid}_{kind}.png"), dpi=100); plt.close(fig)
        ov = S.tissue_overlap(cam, store.seg(pid))
        if ov: rows.append({"participant_id": pid, "kind": kind, **ov})
        if k < 3:
            for r in S.cascading_randomization(m, vol, mask, dev):
                rand_rows.append({"participant_id": pid, **r})
        if k < max_stability:
            ms = []
            for r in range(cfg.cv.n_repeats):
                f_r = int(sp[(sp.repeat == r) & (sp.role == "test") & (sp.participant_id == pid)]["fold"].iloc[0])
                ms.append(load_model_for_fold(model_name, cfg, paths, r, f_r, dev))
            st = S.stability_across_models(ms, vol, mask, dev)
            stab_rows.append({"participant_id": pid, "mean_pairwise_pearson": st["mean"]})
    # ── explanation stability under small input perturbations, split by correct / wrong predictions ──
    P0 = t0[[c for c in t0.columns if c.startswith("p_cal_")]].values
    t0 = t0.assign(correct=(P0.argmax(1) == t0["label"].values.astype(int)))
    n_cases = int(cfg.get("stability", {}).get("n_cases", 40))
    rng = np.random.default_rng(cfg.cv.seed)
    wrong_ids = t0.loc[~t0.correct, "participant_id"].tolist()
    right_ids = t0.loc[t0.correct, "participant_id"].tolist()
    pick = list(rng.permutation(wrong_ids)[: n_cases // 2]) + list(rng.permutation(right_ids)[: n_cases - min(len(wrong_ids), n_cases // 2)])
    fold_models, pert_rows = {}, []
    for pid in pick:
        fold = int(sp[(sp.repeat == 0) & (sp.role == "test") & (sp.participant_id == pid)]["fold"].iloc[0])
        if fold not in fold_models:
            fold_models[fold] = load_model_for_fold(model_name, cfg, paths, 0, fold, dev)
        vol, mask = store.get(pid), store.get_mask(pid)
        ok = bool(t0.loc[t0.participant_id == pid, "correct"].iloc[0])
        for pn, v in S.perturbation_stability(fold_models[fold], vol, mask, dev).items():
            pert_rows.append({"participant_id": pid, "correct_prediction": ok, "perturbation": pn, **v})
    pert = pd.DataFrame(pert_rows)
    pert.to_csv(os.path.join(out, "perturbation_stability.csv"), index=False)
    res["perturbation_stability"] = (pert.groupby(["perturbation", "correct_prediction"]).agg(
        n=("pearson", "size"), mean_pearson=("pearson", "mean"), median_pearson=("pearson", "median"),
        prediction_changed=("pred_changed", "mean")).round(3).reset_index()) if len(pert) else pert
    if rows: pd.DataFrame(rows).to_csv(os.path.join(out, "tissue_overlap.csv"), index=False)
    rd = pd.DataFrame(rand_rows); rd.to_csv(os.path.join(out, "randomization_check.csv"), index=False)
    sd = pd.DataFrame(stab_rows); sd.to_csv(os.path.join(out, "seed_stability.csv"), index=False)
    res["randomization"], res["stability"] = rd, sd
    res["tissue_overlap"] = pd.DataFrame(rows)
    return res


# ───────────────────────── step 9 ─────────────────────────
def step9_export(cfg, paths, model_name):
    from .inference import export_bundle
    lab = load_labeled(paths)
    stats = load_json(os.path.join(paths.cache_drive, "qc_stats.json"))
    b = export_bundle(model_name, cfg, paths, lab, stats)
    log(f"bundle written to {paths.bundle}  (q_hat={b['qhat']:.3f}, rejection tau={b['conf_tau']:.3f})")
    return b


# ───────────────────────── step 10: ablations ─────────────────────────
def step10_ablations(cfg, paths, main_model="cnn3d", variants=None):
    from .ablation import run_ablations
    return run_ablations(cfg, paths, main_model, variants)


# ───────────────────────── step 11: external validation (optional) ─────────────────────────
def step11_external(cfg, paths, table_csv):
    from .external import evaluate_external
    res, df = evaluate_external(paths.bundle, table_csv, cfg.eval.n_boot)
    d = paths.out("step11_external")
    df.to_csv(os.path.join(d, "external_predictions.csv"), index=False); save_json(res, os.path.join(d, "external_report.json"))
    return res, df
