# Frozen candidate on cropped-model validation predictions

Evaluate the existing 80-scene validation inference archive with the unchanged
public `contact_axis_orientation/candidate.py`. No model inference, training or
fitting-parameter search is performed. The test1 runner and previous results
remain unchanged. This separate runner uses the same fitting and independent
scoring checks, reading the archive's `val/` records and predictions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/aburaidah/miniconda3/envs/myriad_finetune/bin/python \
  -m testing_joint_fitting.contact_axis_orientation.cropped_validation.evaluate \
  --archive training_from_scratch_cropped/runs/full_test1_val80_20261021 \
  --workers 4
```

Outputs are saved in the archive's `candidate_val/` directory:

- `REPORT.md`: overall, per-joint and per-object metrics, coverage, comparisons.
- `summary.json`: all aggregates, including per-scene metrics.
- `rows.json`, `pairs/`: fitted axes, pivots, angles and transforms; scoring
  references; solver diagnostics; original prediction paths and hashes.
- `outliers.json`: largest axis errors.
- `exclusions.json`: fixed joints and missing observations.
- `protocol.json`: frozen configuration, runner/dependency and input hashes.
- `audit.json`: independent error/reprojection/transform checks and saved-file
  verification.

Prediction/source paths in result rows are relative to the archive's `val/`
directory. Original images, masks, all five trajectory hypotheses and mesh data
remain available there for visualization. Primary evaluation uses hypothesis 0,
states 2–4, source-image pixels and initial mesh geometry only. Joint types are
assumed revolute; fixed joints are excluded. Unavailable fits remain in coverage
and receive 90 degrees in the separate failure-inclusive axis metric.

The exact fitting configuration/dependency hashes must match the preceding
test1 evaluation. A comparison with the older uncropped validation evaluation
matches original scene filenames, joint order and object identity, since local
scene indices differ between archived subsets. Different model checkpoints and
data distributions prevent attributing differences solely to cropping.
