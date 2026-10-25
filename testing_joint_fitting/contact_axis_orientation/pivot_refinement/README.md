# Soft contact-center pivot refinement

This second development loop keeps the frozen orientation estimator and its
previous experiments intact. It reuses saved uncropped hypothesis-zero tracks
on the original 80 scenes / 204 revolute pairs; no inference or training runs.
The axis orientation prior stays at strength 128 with 64 paired normals.

The selected prototype is `pivot_refinement.candidate.estimate_motion`, with the
same argument and result interface as the previous `candidate.estimate_motion`.
See [REPORT.md](REPORT.md) for measured improvements and tradeoffs.

Position strengths, patch sizes, arithmetic means and symmetric contact patches
are compared. An additional adaptive multiplier `sigma / hypot(sigma, 1 pixel)`
reduces contact-center bias on clean tracks while preserving a strong pull when
the original trajectory fit has large residuals. `sigma` is the frozen fitter's
seed reprojection scale; it uses no ground-truth parameters or future meshes.
The position residual remains soft-L1, and axis, pivot and angles fit jointly.

Selection minimizes development mean pivot error subject to complete-cohort
guards: zero failures, mean axis increase at most 0.25°, no decrease in the
fraction within 15°, mean rotation increase at most 1°, GT-track mean pivot at
most 0.002 world units, maximum GT-track axis at most 5°, and GT-track mean
rotation increase at most 1°. The adaptive sweep is evaluated with these guards
before any new validation fitting. All configurations and rejections are saved.

Use `/home/aburaidah/miniconda3/envs/myriad_finetune/bin/python`, from the repository
root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```bash
python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.run
python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.run --control
python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.select
python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.transfer
python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.audit
MPLCONFIGDIR=/tmp/contact_pivot_matplotlib python -B -m testing_joint_fitting.contact_axis_orientation.pivot_refinement.report
python -B -m unittest testing_joint_fitting.contact_axis_orientation.test_geometry testing_joint_fitting.contact_axis_orientation.pivot_refinement.test_refinement -v
```

The selected configuration is frozen before re-evaluating the existing 50-scene
validation archive. These validation objects differ from development objects,
but this archive has been evaluated previously. It is a transfer check, not a
newly untouched benchmark. Production fitting and model checkpoints stay intact.

`results/` contains immutable source snapshots, all pair parameters and metrics,
GT-track controls, the frozen selection, transfer protocol and independent
audits. Historical results are preserved if fitter source changes. The pivot
metric measures distance to an infinite line: moving a pivot along its axis
alone cannot improve it. The constraint must move the line toward the center.
