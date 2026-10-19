# Cropped-input test1 evaluation

This isolated evaluation compares one fixed crop policy with original-image
inference on the same checkpoint, points, seeds and all original-valid targets.
It uses the existing joint fitter unchanged. It does not tune fitting algorithms.

The square crop encloses the initial visible parent/child union with 10% margin
per side and at least two source pixels of padding, capped/shifted inside the
original square view. The crop retains context inside its square. No future
trajectories determine its bounds. Masks, overlays, coordinates, flows and camera
intrinsics follow the same pixel-center affine transform. World geometry stays
unchanged; predictions return to original coordinates before fitting/scoring.
Future targets outside the crop still count; predictions are not clipped.

Run in `myriad_finetune`, from the repository root (requires both host GPUs):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_COMPILE_THREADS=1 \
python -B -u -m testing_crop_inference.run \
  --devices cuda:0 cuda:1 --output-dir testing_crop_inference/results
```

Default: entire test1 split, current `best.pt`, five hypotheses, 16 steps, no
motion hint. Each GPU processes disjoint pairs with its own frozen FP32 DINO and
FP32 motion model. Omitting `--devices` places DINO on cuda:0 and motion on cuda:1.
Feature rounding matches training (FP16 storage,
widened to FP32). No new features are saved to the disk feature cache. Both views
receive fresh DINO inference. A hard link freezes the checkpoint without copying
its tensor data. Per-pair records resume only with identical provenance/source
code. Inputs, training code and the production fitter are not edited.

Optional smoke run: add `--limit-pairs 2 --output-dir
testing_crop_inference/results_smoke`. Optional existing scene subset:
`--subset testing_joint_fitting/results/subset.json`.

```bash
python -B -m unittest testing_crop_inference.test_geometry -v
python -B -m testing_crop_inference.run --report-only \
  --output-dir testing_crop_inference/results
```

Results contain paired overall/per-joint metrics, joint-type confusion, forced
revolute and correctly classified axis/pivot scores, rotation errors, original-
coordinate trajectory errors and GT-path percentages. The audit checks point
and scoring-mask identity, coordinate round trips and camera reprojection.
