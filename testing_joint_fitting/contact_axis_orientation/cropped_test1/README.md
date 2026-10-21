# Frozen candidate evaluation on cropped-model test1

This runner evaluates the unchanged public `contact_axis_orientation/candidate.py`
on archived predictions from the cropped checkpoint at step 86000. It performs
no model inference and no fitting-parameter search. Training code and the fitter
are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/aburaidah/miniconda3/envs/myriad_finetune/bin/python \
  -m testing_joint_fitting.contact_axis_orientation.cropped_test1.evaluate \
  --archive training_from_scratch_cropped/runs/full_test1_val80_20261021 \
  --workers 4
```

Outputs are saved in the inference archive's `candidate_test1/` directory. Read
`REPORT.md`, `summary.json`, and `audit.json` there. `pairs/` and `rows.json` retain
fitted world transforms, axes, pivots, angles, diagnostics, scoring references,
and links/hashes for the original prediction files. Original inference images,
masks, source scenes, five hypotheses and trajectories remain in the archive.

After evaluation, run the same environment with
`python -m testing_joint_fitting.contact_axis_orientation.cropped_test1.finalize`
to verify saved prediction/record hashes and add failure coverage and per-object
tables to the report. The completed run's verification passed for all 832 records.

All 320 test1 scenes are considered. The cohort contains 832 scoreable revolute
pairs, with 45 fixed pairs, four pairs without future targets, and 38 joints with
no usable initial observations reported separately. Joint ordinals preserve
the original joint index + 1. No joint-type classification is performed.

Only hypothesis 0 is fitted, in original source-image pixel coordinates, with
the original visibility masks. The estimator receives initial world anchors,
initial parent/child mesh points and normals, camera calibration, and predicted
2D tracks. Ground-truth articulation is used after fitting for scoring only.
Fitting states are 2, 3 and 4, matching the previous candidate experiment.

The protocol freezes the archive manifest, every inference record, the runner,
candidate configuration and fitting dependency hashes before execution.
Independent checks recompute axis/pivot/rotation errors and reprojection RMSE,
and verify transform reconstruction and positive scored depth. An exact
scene/joint/object match compares the overlapping pairs with the earlier
uncropped-model 80-scene development evaluation; different checkpoints mean
that comparison does not isolate cropping's causal effect.
