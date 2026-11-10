# Supervisor's plan -> where it is in this package

| Supervisor asks for | Where |
|---|---|
| Data audit, label audit, leakage prevention | NB01 (audit_report.json), NB03 (leakage assertion) |
| Majority-class baseline | NB04 `majority__prior` |
| Conventional ML on clinical / extracted features | NB04: age, sex, nWBV/eTIV, **tissue fractions** (image-derived) |
| 3D CNN | NB05 `cnn3d` |
| Raw softmax vs temperature scaling vs deep ensemble (ECE, Brier, NLL) | NB06 calibration table (raw / temp-scaled / MC dropout / **deep ensemble of repeats**) |
| OOD: entropy, ensemble disagreement, feature distance | NB07 `mc_entropy`, `ens_disagreement`, `ens_mutual_info`, `mahalanobis` |
| Familiar-but-difficult vs unfamiliar scans | NB07 `hard_familiar(...)` and `younger_vs_hard_familiar` rows |
| Independent-cohort OOD, otherwise internal stress tests | NB07 (younger adults + corruptions); NB12 for a real external cohort |
| Selective prediction: coverage, selective risk, risk-coverage | NB08 |
| Age/data-imbalance effects, subgroup analysis | NB06 age-matched table, score-age correlation, incremental value, **subgroup table (age band, sex)** |
| Explanation stability + correct vs incorrect cases | NB08 randomisation, cross-model stability, **perturbation stability split by correct/wrong** |
| Ablations | NB11 (augmentation, class weighting, 2D vs 3D, pre-training, **mask-only and eroded-boundary shortcut probes**) |
| Sensitivity / specificity, AUROC, balanced acc, macro-F1, CIs | all comparison tables (participant-level bootstrap CIs) |
| Research prototype with probability, uncertainty, explanation, review warning | NB10 web app |
| Literature review | docs/LITERATURE_NOTES.md (starting point) |
