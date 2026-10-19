# Isolated joint-fitting experiments

This folder contains the timing benchmark, an isolated candidate, and the
80-scene development experiments. The existing training code, evaluation
archives, and production fitter are unchanged. Candidate results have not been
evaluated on the other 240 scenes.

## Development results

Read [results/REPORT.md](results/REPORT.md) for every ablation, common-cohort
metrics, per-joint comparisons, controls, uncertainty diagnostics, and figures.
The frozen selection is [results/subset.json](results/subset.json): 80 scenes,
219 initially eligible pairs, 216 scoreable pairs (204 revolute, 12 fixed).

The simplest useful candidate is `contact_restart`: it keeps the original type
decisions and adds a soft axis-line preference near a connection region derived
from the initial parent/child depth surfaces. Mean pivot distance falls from
0.6762 to 0.0792 world units, with improvements on 83.8% of revolute pairs.
Median falls from 0.0921 to 0.0620. Mean axis error remains 46.60 degrees.
This is a development finding, not a held-out performance claim. Joint-type
recognition remains poor, particularly for fixed joints.

`candidate.estimate_motion` exposes this candidate without integrating it into
the original fitter. Inputs are initial world points, source-pixel trajectories,
correspondence validity, initial depth/segmentation, pair labels and camera
calibration. It returns world-space transforms at states 2--4 and the complete
fit/diagnostics. Axis direction uncertainty is a local stability diagnostic,
not a calibrated error probability. The wrapper still depends on the unchanged
original initializer/type selector; it should not be installed by overwriting
that initializer in place.

Reproduce the experiments in the `myriad_finetune` environment, from the repo
root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```bash
python -B -u -m testing_joint_fitting.experiment --variants \
  baseline all_states track_filter moving_tracks bic refined track_consensus \
  contact_prior contact_restart contact_tracks contact_all_states contact_type \
  shape_mdl shape_only early_contact ensemble_weighted track_stability pnp_global
python -B -u -m testing_joint_fitting.experiment --variants ensemble_physics ensemble_medoid
python -B -u -m testing_joint_fitting.experiment --control --variants baseline contact_restart shape_mdl
python -B -u -m testing_joint_fitting.experiment --mode with_poke --variants baseline contact_restart
MPLCONFIGDIR=/tmp/myriad_fitting_mpl python -B -m testing_joint_fitting.report
python -B -m testing_joint_fitting.audit
python -B -m unittest testing_joint_fitting.test_conditioning -v
```

`experiment` reuses the existing frozen scene selection; it never runs Myriad.
The second ensemble rule reuses the first rule's cached five-hypothesis fits
when run in the same invocation. The report includes that pool-building cost.
`inputs_sha256.json` fingerprints the 945 development input files used by both
modes. `audit.json` checks original-baseline parity, common cohorts, unit axes,
summary consistency, frozen type decisions, and unchanged inputs on rerun.
Source snapshots for later experiments are retained under `results/code/`.
Generated local artifacts under `results/` are ignored by this folder's Git
rules; no changes were made to the repository's root `.gitignore`.

## Measured baseline timing

On 2026-10-18, `benchmark.py` applied the unchanged fitter to 64 joint pairs
from 56 scenes, sampling across object and joint-ordinal groups without using
prediction errors. It ran image-only predictions, first-motion-hint predictions,
and GT-trajectory controls: 192 fits in 10.37 seconds, excluding Python import
and a separate 0.06-second warm-up. BLAS/OpenMP threads were set to one.

| Input | Mean fitting time / pair | Extrapolated 877 pairs, with measured loading |
| --- | ---: | ---: |
| Image-only | 0.0522 s | 88.1 s |
| First-motion hint | 0.0404 s | 36.2 s |
| GT control | 0.0196 s | 17.8 s |

The first mode pays source-file loading costs, and subsequent modes reuse the
camera/source cache. These are sequential, warm-filesystem estimates on this
machine, not guaranteed timings. There are 320 scenes, 880 eligible joint
pairs, and 877 image-only scoreable pairs. The with-hint scoreable count is 876;
the common 877-pair extrapolation is an approximation for that mode.

An optimizer with 8--16 starts could require roughly 7--14 minutes for a full
image-only pass if each start costs as much as today's complete fitter. That is
an extrapolation, not a measurement of the proposed optimizer. Source loading
can be reused across trials; no Myriad inference, checkpoint, or GPU is needed.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/aburaidah/miniconda3/envs/myriad_finetune/bin/python -B -u \
  -m testing_joint_fitting.benchmark
```

Full measurements and sampled indices are in `benchmark_results.json`.

## Proposed geometric experiments

Keep predictions and hypothesis 0 fixed when comparing estimators. The fitter
receives initial RGB-D points, camera calibration, predicted trajectories, and
the existing benchmark correspondence masks. True future transforms, joint
type, axis, and pivot are available only to the scorer, never to fitting or
per-example hypothesis selection. Do not average the five stochastic rollouts:
they can represent distinct motions. Image-only is primary; hint and GT inputs
are separate diagnostics.

1. **A well-conditioned shared-axis fit.** Fit all times jointly under
   X(t) = R(u, theta(t)) (X(0)-q) + q, with a unit axis u and pivot q constrained
   to the plane through the initial point centroid perpendicular to u. Scale and
   center world coordinates. Use minimal direction/pivot coordinates to remove
   redundant degrees of freedom, and require physically valid camera depth.
   This targets scale conditioning and the unobservable pivot coordinate.
2. **Initialization and convergence.** Compare robust rigid-pose initializations
   and a deterministic small set of alternative joint-axis starts, followed by
   direct shared-axis reprojection refinement. Compare converged robust costs
   without any GT-axis or GT-pivot selection. This targets PnP/local-minimum
   failures and the current short, finite-difference optimization budget.
3. **Trajectory-aware robustness.** Test weighting an entire point track when
   it remains inconsistent across times, alongside point-state robust losses.
   Test adding predicted state 1 to the fit, keeping geometry scoring at states
   2--4. Do not impose constant angular speed simply because there are few
   frames. This targets nonrigid/outlier tracks and limited geometric evidence.
4. **Joint-type selection.** Compare fixed, prismatic, and revolute models with
   a complexity penalty or held-out-point predictive residuals, rather than
   choosing unpenalized training residuals. A revolute model has extra freedom
   to explain noise; improvement should justify that freedom. Do not exploit
   the test set's absence of true prismatic joints by removing that candidate.
5. **Identifiability.** Examine reprojection Jacobian conditioning and compare
   estimated axes under small input perturbations. Use any regularization
   conservatively and with initial geometry only. A lower 2D residual need not
   imply a more accurate 3D joint for small or geometrically ambiguous motions.

Each round should state its geometric motivation, change one mechanism, save
its configuration, record runtime, and compare axis/pivot errors, rotation
errors, balanced joint-type accuracy, failures, and per-joint breakdowns. Keep
forced-revolute metrics and a fixed cohort so apparent improvement cannot come
from excluding difficult joints. Use the transform-consistent pivot reference
already established in the full report; keep raw annotation distances too.

Choose a fixed scene-level development portion (for example 80 scenes) before
inspecting candidate scores; reserve the other 240 scenes for the final frozen
comparison. Repeated development feedback makes that portion development data,
not an unbiased test. Report development and held-out results separately plus
the all-320 summary. GT-input controls and synthetic known-joint perturbations
help distinguish optimization failure from information lost in predictions.

Stop when successive geometrically motivated variants no longer improve the
development trade-off and failure analysis indicates noise/ambiguity rather
than an unresolved numerical issue. This is a practical stopping criterion,
not proof of globally optimal recovery. Keep the winning candidate here for
review before replacing the production fitter.
