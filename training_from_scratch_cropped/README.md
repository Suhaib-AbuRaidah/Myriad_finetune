# Cropped, all-pair Myriad training

Run from the repository root in the `myriad_finetune` conda environment. This
package writes only inside `training_from_scratch_cropped/`. Existing trainers,
data, checkpoints, feature caches, and experiment results are read-only inputs.

The complete randomly initialized Myriad Large motion transformer and flow head
run in FP32 on `cuda:1`. The existing `DeviceAdamW` transfers FP32 gradients to
`cuda:0`, clips them there, performs FP32 AdamW, and transfers updated weights
back. Defaults are microbatch 1, accumulation 4, eight child slots, four
transitions. There is one process, no AMP/TF32, and no distributed or layer
splitting. The unchanged teacher-forced flow-matching objective trains the
model. Joint fitting supplies validation metrics only.

## Commands

```bash
conda activate myriad_finetune
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Precompute the training images and the selected 100 validation images. This
command is resumable: repeat it after interruption; valid entries are preserved.
The frozen encoder is loaded from the same existing `myriad_openset.pt` archive.

```bash
python -m training_from_scratch_cropped.precompute \
  --output-dir training_from_scratch_cropped/runs/cropped \
  --feature-cache training_from_scratch_cropped/feature_cache \
  --scene-cache training_from_scratch_cropped/scene_cache \
  --dino-checkpoint myriad_openset.pt --device cuda:1
```

`--precompute-limit 100` computes at most 100 missing entries per invocation
while still enumerating and reporting the entire required image inventory.
Omit the limit to complete coverage. Initial inventory/validation selection may
take several minutes because compressed scenes must be inspected. The saved manifest avoids repeating selection on subsequent runs. Precomputation never allocates
the motion model or optimizer.

Fresh training (after precomputation):

```bash
python -m training_from_scratch_cropped.train \
  --output-dir training_from_scratch_cropped/runs/cropped \
  --feature-cache training_from_scratch_cropped/feature_cache \
  --scene-cache training_from_scratch_cropped/scene_cache \
  --no-precompute --gradient-checkpointing
```

Training enables precomputation by default, so omitting `--no-precompute` fills
missing/invalid entries before allocating AdamW. Once coverage is verified, the
encoder is released and the motion model has an encoder specification with no
weights. Every training/validation access thereafter requires valid disk
features; a miss fails instead of allocating DINO alongside optimizer states.

Resume a new-pipeline checkpoint:

```bash
python -m training_from_scratch_cropped.train \
  --resume training_from_scratch_cropped/runs/cropped/latest.pt \
  --no-precompute
```

The saved learning rate, schedule, crop settings, precision, and other training
defaults are inherited. Supply custom cache locations again on resume. Changed
source inventories, crop/preprocessing/cache configurations, validation cohort,
or scoring definitions are rejected. Old-pipeline checkpoints are rejected.

Validate all 100 selected **revolute pairs**, with one initial-only hypothesis, 16
diffusion steps, calibrated fitting, and original-coordinate prediction arrays:

```bash
python -m training_from_scratch_cropped.evaluate \
  --resume training_from_scratch_cropped/runs/cropped/best.pt \
  --no-precompute
```

Infer one enumerated pair and retain its contact-prior fit and original pixel
predictions. The pair index refers to the split's sorted pair inventory:

```bash
python -m training_from_scratch_cropped.inference \
  --resume training_from_scratch_cropped/runs/cropped/best.pt \
  --split val --pair-index 0
```

Inference supports the dataset's train/val scene archives; it does not yet
provide an arbitrary RGB-D upload interface. No evaluation command generates
routine PNGs or GIFs. `evaluate` saves all 100 source-coordinate predictions;
training validation saves JSON metrics and fitted parameters.

Configure locations using `--output-dir`, `--feature-cache`, `--scene-cache`,
`--data-root`, and `--dino-checkpoint`. Artifact paths must remain under this
package. Configure crops using `--crop-padding 0.1`, `--image-size 384`, and
`--overlay-alpha 0.85`, consistently for precomputation and training. FP32 DINO
computation is mandatory; `--feature-cache-dtype fp16` (default) or `fp32`
controls storage. Loaded features are widened to FP32.

## Traversal, FPS, and geometry

`pair_inventory.json` lists every initially visible, positive finite valid-depth
child edge, its source file signature, parent/child labels, and joint index.
Dataset length counts pairs. Each epoch shuffles scenes using `seed + epoch`,
then visits every eligible joint within each scene in sorted joint-index order.
The resumable cursor records epoch, pair offset, seed and inventory signature.
Batches stop at epoch boundaries; accumulation may span boundaries. Every pair
is visited once in each epoch. Scenes with more eligible joints contribute more
examples; updates per epoch average approximately pairs / (batch × accumulation),
so epoch length differs from the previous scene-sampling trainer.

The existing unique visible-depth FPS algorithm is imported unchanged: random
start, successive farthest-from-nearest-selected points, no repeated candidate
indices, eight slots with false padding masks, and training point-order shuffle.
Training FPS uses the saved Torch RNG stream; validation FPS is deterministic
per pair. Point identities, local 3D points, and ordering remain consistent
across every state. FPS choices never enter the image-feature identity.

Crops use only state-zero parent/child segmentation. The square union box has
10% padding on each side by default, a minimum side of eight source pixels,
and boundary shifts. Rectangular images can require a square extending beyond
the short dimension: RGB receives edge padding and masks receive false padding.
Context and original texture remain inside the crop; roles use the existing
blue-parent/orange-child overlays. No future target or GT joint parameter
chooses or enlarges the crop.

The pixel-center transform is
`u_crop = (u_source - left + 0.5) * image_size / side - 0.5` (likewise for y).
Normalized model positions divide these pixels by `image_size - 1`. The saved
3×3 affine matrix and inverse also transform calibration: `K_crop = A @ K_source`.
Images, masks, initial points, relative/actual tracks and flow targets share
this transform. Source trajectories, original intrinsics/extrinsics, source
pixel references, local 3D points, and original depth-file references remain
available. Physical geometry and time values are unchanged.

**Off-crop policy:** training supervises every positive-depth, finite target
inside the original source frame, even if outside the crop. Coordinates remain
unclamped. `source_position_valid` and `crop_position_valid` are separate;
`position_valid` uses original-frame validity. Evaluation retains exactly that
originally valid cohort, inverse-transforms predictions, and reports counts and
fractions of future targets leaving the crop. Unobservable actual-scene future
occlusion is not treated as frozen-parent visibility.

## Cache identity and coverage

A separate `cropped_dino_v1` namespace includes pretrained encoder identity,
source/pair inventory signatures, crop settings, role convention, output size,
preprocessing and compute/storage precision. Within it, SHA-256 hashes of the
actual conditioned float image identify unique inputs. Byte-identical images
can share features; different FPS points reuse the same image. Source/pair
provenance and all required image paths are retained in inventories and
`cache_coverage.json`. Encoder identity uses the repository's existing archive
metadata signature, rather than hashing all eight gigabytes of checkpoint data.

Writes use fsync plus atomic rename. Reads validate namespace, feature/position
shapes, token count, dtypes and finiteness. Precomputation replaces invalid
entries, preserving valid ones. Reports distinguish valid/remaining required
entries, hits, misses and invalid entries. Coverage must be complete before
training and is checked again whenever an epoch completes, including epoch one.
No DINO forward path exists after completion.

## Validation and checkpoint selection

`validation_manifest.json` contains exactly **100 unique scoreable GT-revolute
pairs** from the validation split. Fixed and prismatic joints are excluded.
Eligibility also requires four initial valid FPS points, at least eight
originally valid fitting observations at states 2–4, a scoreable future target,
and a finite nonzero GT axis. All 100 pairs belong to the axis-scoring cohort.
Training continues to enumerate every eligible joint pair.

Selection represents the distribution of **eligible scoreable revolute pairs
in the validation split**. Object counts use proportional largest-remainder
quotas, with at least one example per object when 100 slots permit. It then
prefers distinct scenes and minimizes marginal count discrepancies for
object/category, joint ordinal, chain depth, motion quartile, apparent child-area
quartile, crop-magnification quartile and viewpoint. Seeded priorities resolve
ties. It does not give common and rare objects equal quotas.

The manifest saves quotas, available/selected counts and fractions, continuous
quantiles, total variation distance for each categorical marginal, uncovered
strata, exclusions, source signatures and coverage constraints. This is a
coverage heuristic; 100 examples cannot preserve every joint cross-stratum
combination. No model predictions or prediction errors enter selection.

Build or inspect the reviewed cohort without a model or checkpoints:

```bash
python -m training_from_scratch_cropped.selection
```

This writes `manifests/validation_100_revolute.json`. New runs reuse it when the
source signatures, seed, FPS and crop settings match exactly; otherwise they
build their own deterministic selection. The per-run manifest is reused on
resume. Older mixed-type manifests and checkpoints using that cohort are
incompatible and require a fresh run with the new selection.

All selected pairs get one initial-only rollout. The existing calibrated
`contact_restart` estimator sees predicted source-image tracks, initial world
points, initial masks/depth and original calibration. It receives no GT type,
axis, pivot, future transform or future GT validity mask. Contact surface
uncertainty supplies a soft prior on the axis line; missing/insufficient contact
evidence falls back explicitly to trajectory-only fitting. The pivot gauge is
the plane through initial point centroid perpendicular to the estimated axis;
moving a pivot along its axis is unidentifiable. No fitter tuning or trajectory
offset experiments are performed.

For all 100 validation pairs, use the fitted revolute candidate regardless of
predicted classification:

```text
axis_error_deg = degrees(acos(clip(abs(dot(unit(pred_axis_world),
                                         unit(gt_axis_world))), 0, 1)))
checkpoint_score = arithmetic mean over the fixed GT-revolute cohort
```

The range is 0–90 degrees. Missing, invalid or failed axis estimates receive
90 degrees and stay in the denominator. `best.pt` changes only on a strict
decrease; exact ties retain the old best. Pivot, trajectory, rotation, type
accuracy and all other diagnostics never select or break a tie. The metric
definition, cohort, history and best state are saved with checkpoints.

Other reports include type confusion (with a failure column), coverage,
rotation/angle errors, trajectory ADE, source-pixel ADE, GT-path-normalized
error and overall/per-ordinal results. Joint-type accuracy here measures
revolute recall; this cohort cannot measure fixed/prismatic recall. Pivot
distance uses the perpendicular distance from the
transform-consistent frozen-parent GT reference pivot to the predicted revolute
axis line, in source world units. When relative rotation is unobservable this
reference is unavailable and coverage records that fact. Stored-pivot distance
is retained separately as a diagnostic in `forced_revolute_pivot_distance`.

`latest.pt` contains model, FP32 AdamW, scheduler, Python/NumPy/Torch/CUDA RNG,
pair cursor, validation cohort and best history. `best.pt` contains inference
weights and selection metadata. Both reference frozen DINO weights compactly.
Validation restores training mode and process RNG; latest is saved after
validation/best-state updates.

## Verification

```bash
python -m unittest training_from_scratch_cropped.tests.test_pipeline -v
```

Bounded real smoke run (two updates; one training scene, every eligible pair;
the same 100 revolute validation pairs are used). It never writes checkpoints;
temporary outputs and default smoke caches are deleted automatically:

```bash
python -m training_from_scratch_cropped.train \
  --smoke-test \
  --max-train-scenes 1 --gradient-checkpointing \
  --max-steps 100 --warmup-steps 0 --run-steps 2 \
  --evaluate-at-end --log-every-steps 1
```

Console diagnostics record finite losses/gradients, allocated/reserved VRAM on
both GPUs, update duration, optimizer transfer/update duration and examples per
second separately. Startup/cache/validation time is excluded from update
throughput. `--run-steps N` also disables checkpoint writes by default; explicit
`--save-checkpoints` is available for bounded production runs, and is rejected
with `--smoke-test`. Regular unbounded training saves `best.pt` and `latest.pt`.
Fresh smoke runs default to one training scene and two updates. Their full
100-pair validation runs when `--evaluate-at-end` is supplied.
See `VERIFICATION.md` for measurements and limitations. These checks do not
establish long-run convergence.
