# Full cropped-model trajectory archive

This module runs the requested frozen cropped checkpoint on all 320 test1
scenes and 80 validation scenes, including every initially eligible joint.
It reuses the cropped dataset, crop conventions, cached FP32 model and sampler.
It does **not** evaluate or change a joint fitter.

The validation selection uses all 12 objects, seeded round-robin scene
coverage, with no error-based selection. It extends the previous 50-scene
validation subset. There are 881 test1 pairs and 222 validation pairs; source
joints lacking initial visible valid-depth child pixels are explicitly listed.

Default output:
`training_from_scratch_cropped/runs/full_test1_val80_20261021/`.
The checkpoint is snapshotted before execution, so later training changes to
`runs/cropped/best.pt` cannot replace the evaluation weights.

Run in the myriad_finetune conda environment from the repository root:

```bash
python -B -u -m training_from_scratch_cropped.full_inference.cache --device cuda:1
python -B -m unittest training_from_scratch_cropped.full_inference.test_archive -v
python -B -u -m training_from_scratch_cropped.full_inference.run
```

Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for each command.
The first command fills only missing cropped DINO features. The second GPU
command runs two independent inference workers. A record is committed after
its prediction array is saved, allowing interrupted runs to resume.

Each pair's NPZ contains all five hypotheses in three coordinate systems,
relative and actual GT tracks, original/crop/actual validity masks, original
point identities, world anchors and local 3D points, masks, exact model input,
clean cropped RGB, crop bounds/transforms, both camera calibrations and times.
All five-frame source scene arrays are preserved through a checksummed local
hard link or copy. Their mesh points/normals and GT link transforms support
later 3D axis overlays and animations.

Use `visualization_catalog.json` to find examples. It records one entry per
pair with split, object, original scene and joint identifiers, paths and primary
metrics. `summary.json` includes per-object and per-joint trajectory summaries.
No model rerun is needed to visualize these results or subsequently call
`candidate.py` with the saved `reference_world`, `predictions_source_pixels[0]`,
`valid`, camera matrices and initial mesh samples from the scene archive.

**Coordinate warning:** `predictions_crop_normalized` uses the pair's crop.
It must be converted with the saved crop inverse, not the full-image transform.
`predictions_source_pixels` is already restored to original image pixels.
Arrays use `[hypothesis, state, point, xy]`; images use CHW channel order.
`input_image` is the exact float32 model input in [-1,1]; `cropped_rgb` is
uint8 RGB. The `source_archive` array string is relative to the split directory.
Frozen-parent relative tracks belong on the initial RGB background. The
rendered future RGB frames contain actual parent motion; use the separately
saved actual GT trajectories for them. Do not clip future tracks at crop borders.

`audit.json` verifies checkpoint/source hashes, every output, crop transforms,
known initial positions, anchor projections and independently recalculated ADE.

To render a saved example without loading weights or rerunning the model:

```bash
MPLCONFIGDIR=/tmp/myriad_cropped_archive_matplotlib \
python -B -m training_from_scratch_cropped.full_inference.preview --split test1 --pair 0 --hypothesis 0
```
