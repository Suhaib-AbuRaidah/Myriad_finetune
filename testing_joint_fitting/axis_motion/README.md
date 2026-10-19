# Axis orientation: trajectory length and localization shifts

This is an isolated continuation of the first joint-fitting experiments. It
uses the exact same frozen 80 development scenes and 204 scoreable revolute
joints. Fixed joints and classification are outside scope. No estimator is
evaluated on the remaining 240 scenes. No Myriad inference, GPU workload, model
change, or production-fitter replacement is performed.

Read [results/REPORT.md](results/REPORT.md), with figures, all 41 configurations,
per-joint breakdowns, clean-input controls, and 94 perturbation conditions.
Generated data under `results/` is local and ignored by Git.

## Findings

- Predicted paths of 2--5 model pixels have about 81 degrees mean image-motion
  direction error; paths over 40 pixels have about 22 degrees. However, retaining
  only the longest four/six tracks within a joint does not improve 3D axes.
- The selected development candidate, `frame_length_contact`, reduces mean
  axis error from 46.60 to 44.50 degrees on all 204 joints, and increases the
  fraction within 15 degrees from 9.31% to 14.71%. This small gain needs held-out
  evaluation; the remaining absolute error is high.
- Shift structure matters. On GT tracks with a shared 1-model-pixel horizontal
  future offset, original/candidate mean axis errors are 2.18/0.77 degrees.
  Independent persistent per-track offsets require another model: at 1 pixel,
  `bias_track_2px` gives 1.54 degrees versus 4.90 for the original, but it handles
  jitter less well and does not improve unperturbed real predictions overall.
- No joints are dropped from headline metrics. Length-based joint filtering is
  reported separately with retained counts; it changes the evaluation cohort.

## Candidate interface

`candidate.estimate_revolute_motion` accepts initial world points, one predicted
trajectory array `[5,N,2]` in original source-camera pixels, validity `[5,N]`,
camera intrinsic/extrinsic matrices, initial depth/segmentation and pair labels.
It assumes a revolute joint, with the parent frozen at its initial pose.
It returns the world axis line, rotation transforms, and diagnostics. Normally
the returned states are 1--4; if initialization fallback is needed, the explicit
`states` field identifies the available states. A point on the axis is a gauge
choice and is not a uniquely identifiable 3D pivot.

The method uses saturating length weights, common per-frame pixel-offset
nuisance variables, and a soft contact-region line constraint. The offsets
are eliminated by regularized quadratic profiling; the augmented residual is
then minimized with a robust loss. Their prior standard deviation is 2 model
pixels **per coordinate**. This is a tolerance model, not an assertion that the
estimated corrections equal real tracking bias. Individual points cannot move
freely at each frame. See `candidate_config.json` and `refinement.py`.

Pixel errors and perturbation magnitudes are reported at 384x384; the source
images are 512x512. Fitting works in source-camera pixels, converting one model
pixel to 512/384 source pixels. The observed initial state is exact and remains
fixed in every perturbation. Temporal reanchoring alone is algebraically the
same as absolute fitting when that state is already exact.

Estimation receives no true axis, pivot, future transform or true trajectory
length. True kind defines the deliberately revolute-only cohort; scoring truth
is separate from `FitInput`. Explicit GT-input controls and sensitivity runs
replace the input trajectories with GT only to measure estimator behavior.
The inherited benchmark GT geometric-validity mask is kept unchanged; deployment
must obtain its own correspondence validity.

## Reproduction

Use the `myriad_finetune` environment from the repository root. Set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` before running. CPU
processes are parallelized; BLAS remains single-threaded per process.

```bash
python -B -m testing_joint_fitting.axis_motion.analysis
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --variants \
  original contact length_soft length_square extent_square longest4 longest6 spatial4 length_seed
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --variants \
  full_absolute full_length increments hybrid_increments bias_shared \
  bias_track_1px bias_track_2px bias_track_4px bias_track_relative bias_shared_relative
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --variants \
  length_bias frame_bias two_way_bias spatial_differences contact_tight \
  contact_fixed_noise contact_fixed flow_first flow_long flow_quadratic flow_refined flow_fixed_pivot
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --variants \
  frame_length frame_contact frame_length_contact frame_covariance \
  frame_covariance_length frame_covariance_consistent
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --variants \
  geometry_weight frame_geometry frame_length_geometry two_way_conditioned
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --control --variants \
  original contact length_soft full_absolute full_length frame_bias frame_length_contact \
  bias_track_2px increments two_way_conditioned
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --sensitivity --variants original
python -B -u -m testing_joint_fitting.axis_motion.run --workers 4 --sensitivity \
  --variants frame_length_contact bias_track_2px \
  --patterns common_x per_track jitter --amplitudes .5 1 2 4
MPLCONFIGDIR=/tmp/myriad_axis_motion_mpl python -B -m testing_joint_fitting.axis_motion.report
python -B -m testing_joint_fitting.axis_motion.audit
python -B -m unittest testing_joint_fitting.axis_motion.test_axis_motion -v
```

The original sensitivity grid uses 0.25/0.5/1/2/4 model pixels and shared x/y,
persistent per-track and independent frame jitter patterns. The two comparison
estimators use the explicitly listed subset of that grid, always on all 204
joints and both predicted and controlled GT inputs. One fixed random direction
realization is used per pair/pattern, reused across magnitudes and algorithms.
These simulations do not cover every possible localization error distribution.

`summary.json` and per-configuration JSON retain full parameters, counts,
failures, mean/median/tails, and timings. `sensitivity_by_length.json` groups the
1-pixel experiment by true/predicted motion length. `worst_shift_cases.json`
identifies large axis changes. `length_analysis.json` additionally retains
true-axis-radius and common/differential-motion diagnostics (scoring only).
Exact Python snapshots are stored by hash in `results/code/`. `audit.json`
checks baseline parity, axis/rotation/pivot metrics, cohort coverage, pixel
units, positive candidate depth, and unchanged original input hashes.
