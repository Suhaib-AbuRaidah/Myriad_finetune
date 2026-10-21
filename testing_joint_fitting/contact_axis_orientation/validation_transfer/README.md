# Frozen fitting transfer to validation objects

This evaluates the previously recommended strength-128 fitter without changing
its code, thresholds or weights. The user limited inference to **50 scenes**,
with every initially eligible parent–child joint in each selected scene.
`subset50.json` fixes the selection across all 12 validation object identities.

Object IDs were read for all splits to check overlap; that inventory is not
model inference. The full-validation inference attempt failed before producing
predictions because features were missing. Its cache pass was stopped when the
user requested 50 scenes. The old setup is kept as a superseded protocol.

`inference.py` reuses the existing uncropped dataset and inference functions
with a process-local 50-scene dataset adapter. No production source is patched
on disk. `cache.py` fills missing original-image DINO features for this subset.
Inference artifacts live in
`training_from_scratch/runs/val50_contact_axis_transfer_20261021/` to respect
the existing artifact path policy.

Run with the myriad_finetune environment, from the repository root:

```bash
python -B -u -m testing_joint_fitting.contact_axis_orientation.validation_transfer.inference --prepare-only
python -B -u -m testing_joint_fitting.contact_axis_orientation.validation_transfer.cache
python -B -u -m testing_joint_fitting.contact_axis_orientation.validation_transfer.inference
python -B -u -m testing_joint_fitting.contact_axis_orientation.validation_transfer.evaluate --workers 4
MPLCONFIGDIR=/tmp/myriad_contact_val_matplotlib python -B -m testing_joint_fitting.contact_axis_orientation.validation_transfer.report
python -B -m testing_joint_fitting.contact_axis_orientation.validation_transfer.audit
```

Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for these runs.
GPU access is required for feature caching and inference; fitting/reporting
are CPU-only. The report includes per-object and per-joint comparisons,
object-clustered intervals, explicit joint exclusions, and separate GT-track
controls. The controls do not select a new candidate.
