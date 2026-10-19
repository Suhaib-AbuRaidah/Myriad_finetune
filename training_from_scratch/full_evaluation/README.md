# Full test-split analysis of the cached FP32 model

From the repository root, in `myriad_finetune`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -B -u -m training_from_scratch.full_evaluation.run \
  --checkpoint training_from_scratch/runs/robot_arm_fp32_dual_gpu/best.pt \
  --output-dir training_from_scratch/runs/test1_best_full_20261017 \
  --devices cuda:0 cuda:1
```

All eligible pairs from all scenes are evaluated, using five hypotheses and
16 flow steps in FP32. The main `without_poke` mode receives only initial
positions; `with_poke` additionally receives GT state-1 relative positions.
Each GPU loads its own frozen motion model and processes disjoint pair indices.
DINO is represented by the existing read-only FP16 feature cache; embeddings
are widened to FP32 when read. No optimizer or DINO weights are allocated.

The source checkpoint is hard-linked into the output folder before inference,
so an atomic replacement of the training run's `best.pt` cannot change this
evaluation. No additional checkpoint tensor data is copied. Provenance,
SHA-256, all scenes/pairs, seeds and exclusions are saved in `run.json`.
Existing records resume only with the same manifest. A changed checkpoint
requires a new output folder.

The primary percentage metric is **100 × track ADE / complete GT track path
length**, averaged over eligible tracks within a pair and then equally over
pairs. Its denominator is the GT path across states 0–4 in model pixels. Only
complete, non-fixed tracks with path at least one model pixel are included;
the report explicitly counts fixed, shorter and incomplete exclusions. All
tracks retain absolute pixel errors. Alternative forecast-path and ratio-of-sums
normalizations are saved too. Percentages may exceed 100%.
The stationary baseline is also scored on exactly the same normalized-track
cohort. `normalized_stationary_comparison.json` and
`normalization_sensitivity.csv` show its comparison and sensitivity to 0.1, 1,
5 and 10 pixel minimum paths. Improving absolute ADE can coexist with worse
mean relative error when short motions dominate the ratio average.

The existing calibrated RGB-D joint fitter estimates type, axis, pivot and
rotations from hypothesis 0 at predicted states 2–4, with a stationary parent.
Ground-truth joint parameters do not enter fitting or type selection. Initial
depth/calibration and the future GT geometric-validity mask are used. Geometry
scores evaluate the combined prediction-and-fitting pipeline. The same fitter
on GT trajectories is a control. Both stored and transform-consistent pivot
references are reported; distances are in dataset world units.

`REPORT.md` describes definitions, coverage, overall and per-joint tables,
confidence intervals and limitations. `summary.json` and CSVs include per-object,
per-depth, per-scene, per-joint and per-GT-type distributions. Every prediction,
source RGB-D/geometry, fitted parameter vector and track denominator is saved.
Plots are exported as PNG/PDF, with representative trajectory images.

Rebuild reports without loading a model or using CUDA:

```bash
python -B -m training_from_scratch.full_evaluation.run \
  --output-dir training_from_scratch/runs/test1_best_full_20261017 --report-only
```

Metric/unit and synthetic-geometry tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest training_from_scratch.full_evaluation.test_metrics -v
```
