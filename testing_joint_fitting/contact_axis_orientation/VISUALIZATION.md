# Visualize one saved cropped-model scene

From the repository root, in the existing `myriad_finetune` environment:

```bash
conda activate myriad_finetune
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m testing_joint_fitting.contact_axis_orientation.visualize_saved \
  --scene-index 0 --show-gt
```

The default archive is
`training_from_scratch_cropped/runs/full_test1_val80_20261021` (cropped checkpoint
at step 86000), with split `test1` and hypothesis 0. Change `--scene-index` to
select another saved scene, or use `--sample-id <scene-filename-stem>` instead.
Scene indices are local to the inference archive. `--split val` also supports
the 80 archived validation examples.

The script performs these steps:

1. Load all available parent-child pair predictions for the selected scene,
   ordered by source joint index. Include fixed pairs in the trajectory montage.
2. Save `cropped_pairs.png`: square crop tiles in one compact grid, using the
   exact role-tinted model inputs by default. Predicted tracks and observed
   anchor circles are drawn in the corresponding joint's axis color. Future
   predictions remain visible even where GT correspondence masks are invalid.
   Tracks leaving the crop are clipped to the image; coordinates are not clamped.
3. Recompute motion parameters by calling the unchanged public
   `contact_axis_orientation/candidate.py` on each eligible revolute pair. Use
   the selected hypothesis in source-image pixels, original validity masks,
   source camera calibration and initial mesh points/normals only. Fitting
   states are 2, 3, 4, matching the reported evaluations. GT articulation is used
   after fitting for comparison only. Fixed/prismatic pairs and pairs with fewer
   than eight valid fitting observations have explicit skipped statuses.
4. Save `joint_parameters.json` with fit results, metrics, transforms, source
   references, checkpoint identity and candidate configuration.
5. Open a dedicated montage image window and an interactive Open3D geometry
   window. Close both windows to exit. **No Open3D image or screenshot is saved.**

Default output directory:

```text
training_from_scratch_cropped/runs/full_test1_val80_20261021/
  presentation_visualizations/test1_scene_000_hyp_0/
    cropped_pairs.png
    joint_parameters.json
```

The Open3D view shows the initial sampled mesh cloud, including surfaces not
visible in the RGB-D frame. Its saturated, fully opaque part colors provide context; the
axis colors match the montage tile headers and tracks. GT axes are green,
slightly wider, and transparent. Toggle GT/predictions or individual joints in
the side panel, adjust GT opacity, and use the reset button to restore the
source-camera viewpoint. Drag to rotate and scroll to zoom. A small montage
preview also appears in the side panel. A separate resizable image window shows
the whole montage, preserving its aspect ratio; the saved PNG retains full resolution.

Axis meshes are undirected cylinders. Their segment centers are chosen on the
infinite axis line near each pair's initial anchors. This changes neither the
estimated axis line nor its error and avoids presenting an arbitrary pivot
representative as a unique physical joint point. GT uses the transform-consistent
pivot reference, as in the evaluation, rather than the potentially inconsistent
stored joint-origin point.

Dataset-labeled fixed joints are shown in the montage and point cloud, but have
no motion axis to fit or display. Their status reads "fixed joint: no motion axis
to fit (expected)" and their axis toggle is disabled. This is an expected skip.
Unsupported joint types, insufficient observations, and actual fitting failures
retain separate explanatory messages. Joint-type labels come from the dataset;
the candidate does not predict joint type.

Useful options:

```bash
# Visible RGB-D surface with natural image colors, raw untinted crop backgrounds,
# and GT trajectories as well as transparent GT axes:
python -m testing_joint_fitting.contact_axis_orientation.visualize_saved \
  --scene-index 0 --show-gt --gt-opacity 0.3 \
  --cloud rgbd --image raw --gt-trajectories

# Another hypothesis, a single montage row, larger square tiles:
python -m testing_joint_fitting.contact_axis_orientation.visualize_saved \
  --scene-index 10 --hypothesis 1 --columns 4 --tile-size 640 --show-gt
```

Additional options include `--run-dir`, `--output-dir`, `--axis-length`
(fraction of object diagonal, default 0.65), `--axis-radius` (fraction of object
diagonal, default 0.006), and `--point-size` (default 4). Initially unobservable
joints without saved crops are listed explicitly; the script does not invent
missing predictions or rerun neural inference.

Run from a graphical desktop session so Open3D can create its window. The
existing optional visualization dependency is documented in
`finetuning_pair/requirements-visualization.txt` (`open3d==0.19.0`). The script
does not load the checkpoint or DINO, train anything, modify the fitter, or alter
the saved inference/evaluation files.

Implementation follows the existing saved-crop preview and point-cloud viewer
in `training_from_scratch_cropped/full_inference/preview.py` and
`finetuning_pair/pointcloud_viewer.py`. The new script was syntax-checked without
running it, fitting joints or opening any window, as requested.

The axes use `defaultLit` and `defaultLitTransparency`. Open3D 0.19's
`defaultUnlitTransparency` takes an updater path that requests a missing
`srgbColor` uniform and can abort the process, even for hidden GT geometry.
The lit transparent material avoids that mismatch and retains adjustable alpha.
The point cloud still uses the supported opaque `defaultUnlit` material.
See the [Open3D material updater](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1017)
and the [unlit transparent shader](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/visualization/gui/Materials/defaultUnlitTransparency.mat).
