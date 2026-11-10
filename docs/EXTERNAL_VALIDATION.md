# External validation checklist (optional - e.g. OASIS-3)

Do this only if you actually obtain suitable data. Otherwise describe NB07 as **internal stress tests**; do not claim external validation.

1. **Access** - confirm you are allowed to use the dataset and how its data-use terms apply.
2. **Label compatibility** - the external label must mean the same as the training label (CDR 0 vs CDR > 0, age >= 60). Check how and when CDR was assessed relative to the scan date (longitudinal cohorts have many sessions per person; use one scan per person and keep the closest clinical assessment).
3. **Image space** - the model expects OASIS-1-style processed volumes (bias-corrected, atlas-registered, brain-masked, 176x208x176 grid). Other data must be put into the same space first; otherwise the QC check will flag it (this is expected behaviour, and worth reporting).
4. **One scan per participant, no overlap** - make sure no external participant also appears in OASIS-1.
5. Build a CSV: `participant_id, path, label, age, sex` and run NB12.
6. **Report** - AUROC / balanced accuracy vs the internal out-of-fold numbers, calibration (ECE, Brier), conformal coverage vs the 90 % target, abstention rate, QC and OOD flag rates. A drop is a result, not a failure.
Nothing is re-fitted on the external cohort.
