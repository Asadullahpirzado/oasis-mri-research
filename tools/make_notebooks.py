"""Generates notebooks/*.ipynb (plain JSON, no dependencies).  python tools/make_notebooks.py"""
import json, os, uuid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(OUT, exist_ok=True)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True), "id": uuid.uuid4().hex[:8]}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True), "id": uuid.uuid4().hex[:8]}


SETUP = '''
# ── 1. Environment Detection (Google Colab vs Local Machine) ──────────────
import sys, os, subprocess

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── 2. Configure Paths ───────────────────────────────────────────────────
if IN_COLAB:
    # Google Colab paths (on your Google Drive)
    PROJECT  = '/content/drive/MyDrive/oasis_project'      # this unzipped project folder on Drive
    DATA_DIR = '/content/drive/MyDrive/OASIS1'             # folder with the 12 OASIS-1 disc archives (+ clinical csv)
    OUT_DIR  = '/content/drive/MyDrive/OASIS1_outputs'     # results are saved here (survives Colab disconnects)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', f'{PROJECT}/requirements.txt'], check=False)
else:
    # Local machine paths (runs directly on your laptop)
    PROJECT  = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
    DATA_DIR = os.path.join(PROJECT, 'Dataset')
    OUT_DIR  = os.path.join(PROJECT, 'OASIS1_outputs')

# ── 3. install + load ─────────────────────────────────────────────────────
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
from oasis_ad.config import load_config, Paths
from oasis_ad import pipeline as PL, DISCLAIMER
cfg = load_config(os.path.join(PROJECT, 'configs', 'default.yaml'),
                  {'paths.data_dir': DATA_DIR, 'paths.out_dir': OUT_DIR})
paths = Paths(cfg)
import torch, pandas as pd
print(DISCLAIMER)
print('Environment:', 'Google Colab' if IN_COLAB else 'Local Laptop')
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (on Colab switch to Runtime > Change runtime type > T4 GPU)')
'''

NBS = {}

NBS["NB01_Data_Audit_and_Labels"] = [
    md("""# NB01 - Data audit, extraction and label definition
Reads the 12 OASIS-1 disc archives **once**, extracts only the processed brain-masked scans (+ tissue segmentation) to fast local disk,
joins them with the clinical table and **freezes the labels**.

* Primary task: CDR = 0 vs CDR > 0, age >= 60 &nbsp;|&nbsp; Secondary: CDR 0 / 0.5 / >= 1
* One scan per participant (MR1). MR2 repeat scans are excluded and logged.
* MMSE, education and SES are never model inputs.
* Younger participants (< 60) are kept aside for out-of-distribution tests only.

Run time: about 15-40 min the first time (reading 12 archives), seconds afterwards."""),
    code(SETUP),
    code("""
manifest, rep = PL.step1_audit(cfg, paths)
from oasis_ad.utils import _clean
import json; print(json.dumps(_clean({k: rep[k] for k in ['cohort_flow','role_counts','label_counts_binary','label_counts_3class','three_class_warning','young_ood_n']}), indent=2, default=str))
"""),
    md("### Age confounding - is age a shortcut?"),
    code("""
from oasis_ad.utils import _clean
print(json.dumps(_clean(rep['age_confounding']), indent=2, default=str))
from IPython.display import Image, display
display(Image(f"{paths.audit}/age_confounding.png"))
"""),
    md("### Check list before continuing\n* Cohort flow numbers look plausible (about 436 sessions, about 235 with CDR at age >= 60)\n* `three_class_warning` - if the CDR >= 1 group is small, report 3-class results descriptively only\n* Everything is saved in `OASIS1_outputs/step1_audit/`"),
    code("manifest[manifest.role=='labeled'][['participant_id','age','sex','cdr','label_binary','label_3class','nwbv','etiv']].head(10)"),
]

NBS["NB02_Preprocessing_Cache_QC"] = [
    md("""# NB02 - Preprocessing, caching and quality control
The OASIS-1 processed volumes are already bias-corrected, atlas-registered and brain-masked, so **no second N4 / skull stripping** is done.
Each volume is resized to 88x104x88 (aspect ratio preserved) and z-scored inside the brain mask. QC metrics are computed on the raw volume;
hard failures are excluded and logged, soft outliers are flagged. The cache is saved on Drive, so this runs once."""),
    code(SETUP),
    code("qc, passed = PL.step2_preprocess(cfg, paths)\nqc[['participant_id','brain_frac','cov_in','grad_energy','qc_hard_fail','qc_soft_flag','qc_reasons']].sort_values('qc_hard_fail', ascending=False).head(15)"),
    code("from IPython.display import Image, display\ndisplay(Image(f'{paths.cache_drive}/spot_check.png'))   # visually confirm the volumes look like brains"),
    code("print(qc[['qc_hard_fail','qc_soft_flag']].sum())\nqc[qc.qc_soft_flag | qc.qc_hard_fail][['participant_id','qc_reasons']]"),
]

NBS["NB03_Participant_Level_Splits"] = [
    md("""# NB03 - Participant-level repeated cross-validation splits
5-fold x 3 repeats, stratified by class (and age band), grouped by participant. Inside every training part we carve out
**val** (early stopping + temperature scaling) and **calib** (conformal quantile + rejection threshold). The test fold is used for scoring only.
A leakage assertion runs automatically."""),
    code(SETUP),
    code("sp, summ = PL.step3_splits(cfg, paths)\nsumm.head(15)"),
    code("print(sp.groupby(['repeat','role']).size().unstack())"),
]

NBS["NB04_Baselines_and_Age_Matching"] = [
    md("""# NB04 - Baselines (age, sex, brain volume) and age-matched subset
Your MRI model must beat these. They use the *same* outer folds:
* **majority-class** (the floor), **age**, **age + sex**, **brain volume (nWBV, eTIV)**, and **tissue fractions** (CSF / GM / WM from the segmentation files = image-derived features)
An age-matched CDR-0 / CDR>0 subset is prepared for the age-control analysis in NB06."""),
    code(SETUP),
    code("names, pairs = PL.step4_baselines(cfg, paths)\nprint(names); pairs.head()"),
    code("""
from oasis_ad import evaluation as E
lab = PL.load_labeled(paths)
import glob, os
sets = [E.to_predset(pd.read_csv(f), os.path.basename(os.path.dirname(f)), lab) for f in sorted(glob.glob(f'{paths.baselines}/*/fold_outputs.csv'))]
E.compare_models(sets, 300, paths.baselines)
"""),
]

NBS["NB05_Train_Models"] = [
    md("""# NB05 - Train the models (repeated CV, resumable)
`cnn3d` (3D CNN from scratch) is the default. Optional extra models via `cfg.train.models`: `cnn2d`, `resnet10` (set `paths.medicalnet_weights` for transfer learning), `foundation`.
* A checkpoint is written to Drive **after every epoch**. If Colab disconnects, simply re-run the cell: finished folds are skipped, the unfinished one resumes.
* Mixed precision (fp16) is used on GPU.
* Select **Runtime > Change runtime type > T4 GPU** first. Expect roughly 1-3 hours for 15 folds of `cnn3d`."""),
    code(SETUP),
    code("# OPTIONAL quick test: uncomment to train only repeat 0 / fold 0 first\n# PL.step5_train(cfg, paths, repeats=[0], folds=[0])"),
    code("PL.step5_train(cfg, paths)   # models listed in cfg.train.models"),
    code("""
import glob, pandas as pd
rows = [__import__('json').load(open(f)) for f in sorted(glob.glob(f'{paths.out_dir}/step5_models/*/r*_f*/fold_done.json'))]
pd.DataFrame(rows)[['model','repeat','fold','epochs_run','best_val_nll','temperature','n_params','train_seconds']]
"""),
]

NBS["NB06_Evaluation_Calibration_Conformal"] = [
    md("""# NB06 - Evaluation: MRI vs baselines, calibration, conformal prediction, age control
All numbers come from **pooled out-of-fold predictions** (each participant is scored by models that never saw them).
Means over CV repeats with participant-level bootstrap 95 % CIs."""),
    code(SETUP),
    code("MODEL = 'cnn3d'\nres = PL.step6_evaluate(cfg, paths, MODEL)\nres['model_comparison']"),
    md("### Does the image add information beyond age? (RQ2)"),
    code("""
print('Age-matched subset:'); display(res['age_matched'])
print('Score vs age (Spearman):', res['age_dependence'])
print('CV AUROC with age vs age + MRI score:', res['incremental_over_age'])
print('Paired bootstrap AUROC difference (MRI - baseline):'); import json; print(json.dumps(res['paired_vs_baselines'], indent=2))
"""),
    md("### Subgroup performance (age bands, sex)"),
    code("""
cols = [c for c in ['model','subgroup','n','n_positive','balanced_acc','balanced_acc_CI','auroc','auroc_CI','sensitivity','specificity','note'] if c in res['subgroups'].columns]
display(res['subgroups'][cols].fillna(''))
"""),
    md("### Calibration (RQ4): raw softmax vs temperature scaling vs MC dropout vs deep ensemble of the CV repeats"),
    code("display(res['calibration'])\nfrom IPython.display import Image\ndisplay(Image(f'{paths.evaluation}/{MODEL}/reliability.png'))\nprint(res['uncertainty_vs_error'])"),
    md("### Conformal prediction (RQ5) - coverage must be checked per subgroup"),
    code("c = res['conformal']; print('target coverage', c['target_coverage'], ' 95% CI of observed coverage', c['overall_coverage_CI95'])\npd.DataFrame(c['pooled_mean_over_repeats']).T"),
]

NBS["NB07_OOD_and_Robustness"] = [
    md("""# NB07 - Quality and out-of-distribution detection
* **Natural shift:** younger OASIS-1 participants (never used for training/scoring).
* **Synthetic shift:** TorchIO noise, blur, motion, ghosting, bias-field, low-resolution at 5 severities.
* Scores: class-conditional Mahalanobis (shrinkage covariance), MC-dropout entropy, mutual information. Thresholds come from each fold's calibration set, so the **false-alarm rate on clean scans is reported**.
* These tests show robustness only - they do not prove generalisation to other scanners/cohorts."""),
    code(SETUP),
    code("MODEL = 'cnn3d'\ndf = PL.step7_ood(cfg, paths, MODEL)\ndf.groupby(['shift','score'])[['AUROC','AUPR','FPR@95TPR','detection_rate','false_alarm_clean']].mean().round(3)"),
    code("from IPython.display import Image, display\ndisplay(Image(f'{paths.ood}/{MODEL}/ood_summary.png'))"),
    md("""### Familiar-but-difficult vs truly unfamiliar
* `hard_familiar(misclassified)_vs_correct` - can the score tell wrongly classified *familiar* scans from correct ones (error detection)?
* `younger_vs_hard_familiar` - can the score separate truly unfamiliar scans (younger cohort) from merely difficult familiar ones?
A good OOD detector scores high on the second row; a good uncertainty score scores high on the first."""),
    code("df[df['shift'].isin(['hard_familiar(misclassified)_vs_correct','younger_vs_hard_familiar'])].groupby(['shift','score'])[['AUROC','AUPR','detection_rate']].mean().round(3)"),
]

NBS["NB08_Selective_Prediction_and_Saliency"] = [
    md("""# NB08 - Selective prediction (abstention) and Grad-CAM with sanity checks
* Rejection threshold per fold from the calibration set; risk-coverage curves, AURC, selective accuracy at 90 % / 80 % coverage.
* Grad-CAM on a **balanced** set of cases (TP, TN, FP, FN + random) - not only the most confident ones.
* Sanity checks: cascading weight randomisation, stability across independently trained models, descriptive overlap with CSF/GM/WM."""),
    code(SETUP),
    code("MODEL = 'cnn3d'\nres = PL.step8_selective_saliency(cfg, paths, MODEL)\nimport json; print(json.dumps(res['selective']['pooled_mean'], indent=2))\nprint(res['selective']['rejection_rule'])"),
    code("from IPython.display import Image, display\ndisplay(Image(f'{paths.selective}/{MODEL}/risk_coverage.png'))"),
    code("print('Randomisation check (correlation with the original map should DROP):')\nres['randomization'].pivot_table(index='randomised_up_to', values='pearson_with_original').sort_values('pearson_with_original', ascending=False)"),
    code("print('Stability across models trained on different splits:'); display(res['stability'])\nprint('Share of high-activation voxels per tissue (descriptive):'); display(res['tissue_overlap'].groupby('kind').mean(numeric_only=True).round(3))"),
    md("### Explanation stability under small input perturbations (correct vs wrong predictions)\nPearson correlation between the original Grad-CAM and the map after mild noise / 2-voxel shift / intensity change / smooth bias field."),
    code("display(res['perturbation_stability'])"),
    code("import glob\nfor f in sorted(glob.glob(f'{paths.selective}/{MODEL}/gradcam_*.png'))[:8]: display(Image(f))"),
]

NBS["NB09_Export_and_Final_Report"] = [
    md("""# NB09 - Export the deployment bundle + collect the final tables
Packages the 5 fold-models of one CV repeat (an ensemble), their temperature scalers, OOD statistics, the conformal quantile and rejection threshold
into `OASIS1_outputs/step9_bundle/` - the only thing the web app needs."""),
    code(SETUP),
    code("MODEL = 'cnn3d'\nb = PL.step9_export(cfg, paths, MODEL)\n{k: b[k] for k in ['model_name','class_names','qhat','conf_tau','conformal_note']}"),
    code("""
import os, pandas as pd
for f in [f'{paths.evaluation}/{MODEL}/model_comparison.csv', f'{paths.evaluation}/{MODEL}/calibration_summary.csv', f'{paths.evaluation}/{MODEL}/age_matched_metrics.csv']:
    if os.path.exists(f): print(os.path.basename(f)); display(pd.read_csv(f))
print('All result files are under', paths.out_dir)
"""),
]

NBS["NB10_Web_App_Frontend_Backend"] = [
    md("""# NB10 - Web app (FastAPI backend + browser frontend)
Starts the backend inside Colab and shows the frontend in the notebook (or in a new tab). Upload an OASIS-1 `*_masked_gfc.hdr` + `.img` pair,
or run a cached participant. The page shows the prediction **or the abstention and its reasons**, probabilities, conformal set, quality and OOD checks, the age/sex-only comparison and Grad-CAM.
It is a research demonstration - not a medical device."""),
    code(SETUP),
    code("""
from oasis_ad.app.server import run_in_thread
from oasis_ad.data.preprocess import sync_cache_to_local
sync_cache_to_local(paths)
server = run_in_thread(paths.bundle, port=cfg.app.port, cache_dir=paths.cache_local,
                       manifest_csv=f'{paths.cache_drive}/manifest_qc_passed.csv')
print('backend running on port', cfg.app.port)
"""),
    code("""
try:
    from google.colab import output
    output.serve_kernel_port_as_iframe(cfg.app.port, height=950)
    print("If embedded view doesn't load, use: output.serve_kernel_port_as_window(cfg.app.port)")
except ImportError:
    print(f"\\n>>> App running! Open http://localhost:{cfg.app.port} in your browser to view the interface <<<\\n")
"""),
    code("# API check (backend side)\nimport requests; print(requests.get(f'http://localhost:{cfg.app.port}/api/model-info').json())"),
]


NBS["NB11_Ablations"] = [
    md("""# NB11 - Ablation study
Which design choices matter, and where does the model fail? Each variant uses the *same* outer splits (first `ablation.n_repeats` repeats) and is compared
with the main model by paired participant-level bootstrap.

| variant | question |
|---|---|
| no_augmentation / no_class_weighting | do these training choices help? |
| cnn2d_slices | 2D slices vs 3D volume |
| resnet10_scratch / resnet10_medicalnet | does pre-training help? (medicalnet needs `paths.medicalnet_weights`) |
| **mask_only_input** | shortcut probe: only the brain silhouette, no intensities. If this is about as good as the full model, size/shape (or skull-stripping edges) carries the signal |
| **eroded_boundary_input** | shortcut probe: outer brain shell removed. A big drop = reliance on the brain boundary |

Run time: roughly 1/3 of NB05 per variant."""),
    code(SETUP),
    code("tab = PL.step10_ablations(cfg, paths, 'cnn3d')   # variants from cfg.ablation.variants\ntab"),
    md("Interpretation: a variant whose AUROC difference interval includes 0 does **not** differ detectably from the full model - report that honestly."),
]

NBS["NB12_External_Validation_Optional"] = [
    md("""# NB12 - External validation (optional, e.g. OASIS-3)
Evaluates the **frozen** OASIS-1 bundle on another cohort. Nothing is re-fitted (temperature, conformal quantile, rejection and OOD thresholds all come from OASIS-1).
Read `docs/EXTERNAL_VALIDATION.md` first: access, label compatibility and image space must be checked. If you cannot obtain suitable data, report NB07 as *internal stress tests* only."""),
    code(SETUP),
    code("# CSV with columns: participant_id, path, label (0/1, same meaning as training), age (optional), sex (optional)\nTABLE = '/content/drive/MyDrive/external_cohort.csv'\nres, df = PL.step11_external(cfg, paths, TABLE)\nimport json; print(json.dumps(res, indent=2, default=str))"),
]


def main():
    for name, cells in NBS.items():
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"},
              "nbformat": 4, "nbformat_minor": 5}
        with open(os.path.join(OUT, name + ".ipynb"), "w") as f:
            json.dump(nb, f, indent=1)
    print("wrote", len(NBS), "notebooks to", OUT)


if __name__ == "__main__":
    main()
