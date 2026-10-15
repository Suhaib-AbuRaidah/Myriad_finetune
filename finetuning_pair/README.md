# MYRIAD child-only relative fine-tuning

This folder contains an independent copy of the existing fine-tuning pipeline, with pair-specific data and inference. Existing `finetuning/`, the pretrained checkpoint, the MYRIAD architecture, dataset files, and root files are read only. Run commands from the repository root in the existing `myriad_finetune` environment. All generated artifacts must be inside `finetuning_pair/`.

## Training

```bash
conda activate myriad_finetune
python -m finetuning_pair.train \
  --data-root data/Robotic_arm \
  --base-checkpoint myriad_openset.pt \
  --output-dir finetuning_pair/runs/relative \
  --max-steps 10000 --overlay-alpha 0.85
```

The defaults preserve the original architecture, block LoRA (rank 8), optimizer groups (LoRA LR `1e-4`, other LR `3e-5`), AdamW betas `(0.9, 0.99)`, weight decay, clipping, schedule, and flow-matching loss. The frozen DINO encoder is initialized from the original checkpoint without downloading weights. Resume with `--resume finetuning_pair/runs/relative/latest.pt` and the same training settings. Checkpoint selection uses only validation `minADE_1`, with no best-of-five validation metric. Existing whole-object adapters cannot be resumed or used as relative-target adapters; initialize a new run from `myriad_openset.pt`.

Training visits scenes in the original seeded scene order. Each visit chooses uniformly among the ground-truth tree's directly connected edges whose child has an initially visible, finite, valid-depth point inside the model crop. Fixed joints are included and yield stationary relative targets. A parent with no visible pixels contributes an empty visible mask. Scenes with no eligible child are excluded. Pair selection, FPS and point shuffling use the checkpointed torch RNG. Validation/testing enumerate every eligible edge deterministically by scene filename and joint index; bounded periodic validation samples from this pair index. `--val-num-scenes` and `--test-num-scenes` count pair examples in this version; full evaluation covers all pairs.

Every example has eight child slots by default (`--points-per-link`), rather than eight slots per object link. FPS samples unique child pixels, falls back from mask erosion when necessary, and pads missing slots with false masks. The selected parent's points are never motion queries. Batch tensors retain the original shapes and normalized pixel-center coordinates. `pair_labels=[parent,child]` contains dataset/Bullet labels; `point_link_nodes=1` denotes the child in the pair's two-node list.

Default validation and periodic testing use only state-zero points. Optional `--val-poke-mode with/both` or `--test-poke-mode with/both` supplies the **relative** first motion as an observed hint for diagnostic evaluation; that hint is ground truth and is not required by the inference CLI.

## Two-GPU batched training

The trainer also accepts `torchrun`. Each GPU keeps one model replica, receives a different shard of the shared scene batch, and sums gradients before clipping and its optimizer update. Loss normalization uses the total valid child transitions across both GPUs. A short final batch uses a zero-weight dummy on an otherwise empty rank. The original flow-matching objective and optimizer learning rates are unchanged.

For the two RTX 2080 SUPER 8 GB cards, start with one example per GPU (global batch two) and float32:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m finetuning_pair.train \
  --device cuda \
  --data-root data/Robotic_arm \
  --base-checkpoint myriad_openset.pt \
  --output-dir finetuning_pair/runs/pair_2gpu \
  --batch-size 1 --max-steps 10000 \
  --amp none --val-amp-dtype none --test-amp-dtype none \
  --val-every-steps 500 --save-every-steps 500 --test-every-steps 1000
```

`--batch-size` is **per GPU**. Setting it to two gives a global batch of four. `--device cuda` lets each rank choose `cuda:LOCAL_RANK` within `CUDA_VISIBLE_DEVICES`; do not supply `cuda:0` to a two-process launch. The `auto` AMP setting selects BF16 only when native support is available, excluding allocation-only emulation on Turing cards.

Validation, checkpoint writes and test exports run on rank zero while the other rank waits. Intervals count synchronized optimizer steps. Validation reports only `minADE_1`, samples one rollout per pair by default, and runs every 500 steps (16 pair examples), testing every 1,000 steps (four test2 pair examples), and metric logging every 20 steps. `best.pt` saves when validation `minADE_1` improves; `latest.pt` now also saves every `--save-every-steps` (default 500) and at the end of the invocation, before scheduled validation starts so validation errors cannot discard those completed updates. Tests retain their prediction NPZ files, metrics, PNGs and GIFs under the run's `tests/` directory. Both validation and testing also run at the final `--max-steps` step.

Resume with the same launch and training settings, adding `--resume finetuning_pair/runs/pair_2gpu/latest.pt`. Distributed checkpoints include each rank's RNG stream and the shared scene cursor; resuming requires the same GPU count and per-GPU batch size. CPU two-process regression tests verify globally weighted gradients, missing/unused gradients, uneven batches, per-rank RNG restoration, and validation/checkpoint/testing intervals across resume. Small float32 sampler-attention checks and two-GPU NCCL gradient/RNG checks also passed on both physical GPUs; a full pretrained training run has not been launched. The complete regression suite passes 16 tests.

## Conditioning and geometry

Parent RGB color is `(0,120,255)`; child RGB color is `(255,145,0)`. Blending uses exactly:

```text
output[mask] = (1-alpha) * original_rgb[mask] + alpha * role_color
```

`alpha=0.85` means 85% role color and 15% original RGB. Lower `--overlay-alpha`, for example `0.2`, retains more texture. Blend in float before resizing, inside only the selected visible segmentation masks. Other parts and background are unchanged. The shared center crop and bilinear antialiased resize apply to both training and inference. Resize filtering can mix adjacent source pixels at mask boundaries, as it does for any RGB boundary; there is no background or other-part dimming operation.

Schema-v1 arrays provide `T_world_link(t)` and `E=T_camera_world`, a fixed +Z-depth camera. Set `G_link(t)=E @ T_world_link(t)`. Depth backprojection attaches each initially visible point to its fixed child-local coordinates. The target is:

```text
X_camera(t) = G_child(t) @ X_child_local
X_relative(t) = G_parent(0) @ inverse(G_parent(t)) @ X_camera(t)
u_relative(t) = project(K, X_relative(t))
```

All transform math uses float64 homogeneous coordinates before perspective projection. No 2D flow subtraction occurs. Positions use the existing pixel-center center-crop/resize convention:

```text
normalized = (((source_pixel - crop_offset + 0.5) * image_size / crop_side) - 0.5) / (image_size - 1)
```

The endpoint retains the parent's initial world pose while the child-parent transform reaches its zero-joint rest configuration. It can differ substantially from the actual whole-object rest endpoint. All motion positions, origins, teacher-forced histories, poke displacements and query displacement targets are rebuilt from these relative trajectories. Actual trajectories are diagnostics only.

The five source states span progress `[0,.25,.5,.75,1]`. Fewer requested transitions select states across the full sequence, retaining the true rest endpoint rather than truncating the source states. Times are selected progress fractions times the original time scale, not seconds.

`position_valid` excludes padding, nonfinite, behind-camera and out-of-image relative points. Future visibility is unknown: source future depth/masks describe a different moving-parent scene. `visibility` and `visibility_known` therefore mark only initial observations. Occlusion-only loss/metrics options are rejected until counterfactual renders exist; geometric supervision includes occluded surfaces. Relative visualizations consistently use the conditioned initial image, including the endpoint. No actual whole-object rest image is presented as a frozen-parent endpoint.

## Preprocessing, checks and visualizations

```bash
python -m finetuning_pair.preprocess \
  --split train --max-scenes 3 --visualize-examples 12

python -m unittest discover -s finetuning_pair/tests -v
```

Omit `--max-scenes` to prepare and verify an entire split. Preprocessing caches deterministic evaluation examples only; training FPS and random pair selection remain fresh. Cache keys include source path/size/mtime, parent/child/joint, target-definition version, point settings, seed, crop/resize, time settings and overlay alpha. Atomic cache writes live under `finetuning_pair/cache/`. No old whole-object preprocessing cache is consulted. Checkpoints and evaluation manifests record the target definition; saved predictions include the selected pair and version. Evaluation manifests also record the deterministic pair index and source fingerprint.

Saved verification examples are in `finetuning_pair/verification/`, including three actual stochastic training samples and a real fixed-joint case. [Example comparison](verification/pair_0001/comparison.png) shows an approximately 89-pixel difference between relative and actual endpoints; [the fixed-joint comparison](verification/fixed_joint/comparison.png) shows a stationary relative child despite approximately 70 pixels of inherited endpoint motion. Each includes:

- `model_input.png`: the exact conditioned tensor converted to display RGB, without annotations.
- `relative_trajectories.png`: initial query rings and relative targets over that input.
- `comparison.png`: original RGB, conditioned input, relative targets, actual trajectories for comparison.
- `relative.gif`, `targets.npz`, and `verification.json`.

Verification independently evaluates `E @ T_parent(0) @ inverse(T_parent(t)) @ T_child(t) @ local_point`, projects it, checks the initial depth/projection round trip, verifies unchanged non-pair pixels, and checks stationary fixed joints. Tests additionally cover nonidentity cameras, rotation/translation cancellation, unique point padding, batching, teacher forcing, attention/loss masks, deterministic caches, validation/test persistence, raw-input parity, and the original MYRIAD flow-matching head's forward/backward pass. These checks validate target construction and integration, not the quality of an untrained relative adapter.

## Full validation/testing

```bash
python -m finetuning_pair.evaluate \
  --checkpoint finetuning_pair/runs/relative/best.pt \
  --split test1 --output-dir finetuning_pair/runs/test1_relative
```

This enumerates all eligible child pairs and saves both with-poke and without-poke diagnostic metrics, per-pair predictions, PNG/GIF presentation exports, and summaries. Metrics retain the original paper-style `minADE_N` definition (scaled mean squared Euclidean error), evaluated only on the selected child's relative targets. Pixel ADE/FDE/RMSE are also available. Each pair gets equal aggregate weight. Repeated runs resume completed pairs only when the run manifest matches. `--render-only` rebuilds exports from saved relative results without weights or a GPU.

## Inference on a current image

```bash
python -m finetuning_pair.inference \
  --checkpoint finetuning_pair/runs/relative/best.pt \
  --image /path/current_rgb.png \
  --parent-mask /path/visible_parent.png \
  --child-mask /path/visible_child.png \
  --points /path/child_points.npy \
  --output-dir finetuning_pair/runs/inference
```

Masks must be disjoint visible masks in the original RGB resolution; use nonzero PNG or `.npy [H,W]` pixels. Points are unique initially visible child coordinates in original-image pixels (`.npy [N,2]`, x/y). The same crop/resize converts them into model coordinates. Alternatively supply `--point-space normalized-model` with coordinates normalized over the resized crop. Image size, alpha and default progress times come from the pair checkpoint. Autoregressive K/V cache buffers match the active sampling precision (float32 when AMP is disabled). This CLI requires no depth, kinematic tree, future image or ground-truth transforms, and always supplies only the initial child points. `predictions.npz` contains normalized child-relative trajectories in the resized image frame.

The implementation uses the existing dependencies in `environment.yml`. Full pretrained-model training/inference requires sufficient device memory. The saved target visualizations can be reproduced on CPU without loading the multi-gigabyte pretrained model.

## First-scene joint analysis

```bash
conda activate myriad_finetune
python -m finetuning_pair.analyze_joints \
  --checkpoint finetuning_pair/runs/pair/best.pt \
  --first-scenes 50 --device cuda:1 --amp-dtype none
```

This uses the first 50 scene filenames in ascending order and every eligible
child pair in those scenes. It saves five hypotheses per pair for both poke
modes, trajectory metrics, calibrated 3D joint fits, axis orientation error,
pivot-to-axis distance, rotation errors, and overall/per-joint/per-depth/per-scene
statistics under `runs/test1_first50_analysis/`. Axes and pivots are fit from
predicted states 2–4 using a stationary parent, matching the checkpoint's
relative targets. The same fitter on true tracks provides a geometry control.
Main axis/pivot metrics are conditional on correctly typed revolute joints;
forced-revolute metrics, type recall and exclusions are saved alongside them.

The prediction NPZ files include initial RGB-D, segmentation, calibration,
selected points, link transforms, and joint metadata for later visualization.
To render an example without loading weights or using CUDA:

```bash
python -m finetuning_pair.visualize_joint_results \
  --scene-index 0 --joint-index 0 --mode without_poke
```

Joint IDs are zero-based. `ARTIFACTS.md` in the output folder describes all
saved files. Reports can be rebuilt with
`python -m finetuning_pair.analyze_joints --report-only`.

## Live Open3D: all joints together

Install the optional viewer dependency in `myriad_finetune`:

```bash
python -m pip install -r finetuning_pair/requirements-visualization.txt
```

From a graphical desktop terminal, browse the ten best **whole scenes** from
the saved with-poke evaluation:

```bash
conda activate myriad_finetune
python -m finetuning_pair.inference \
  --saved-run finetuning_pair/runs/test1_first50_analysis \
  --mode with_poke --top-k 10 --show-pointcloud
```

This opens an interactive Open3D window with the initial object point cloud
and **every available predicted joint axis simultaneously**. A second window
shows **every joint's predicted point trajectories together** on the original
initial RGB image, using the same joint colors. Hypothesis 0 is displayed so
the axes and tracks match the reported evaluation errors. GT overlays are
green; the optional stored GT axes are cyan. Fixed predictions have no axis;
prismatic directions are anchored at the observed part centroid because their
axis location is unidentifiable. The cloud includes only initially visible
object surfaces with finite positive depth. Trajectories are the independently
parent-frozen relative predictions for each joint.

Use **Next / Previous** in the control window, or **N / P** in the Open3D
window. Drag to rotate, scroll to zoom, **R** resets the camera, and **S** saves
the current point-cloud screenshot plus the combined trajectory image. Closing
either window ends the process. The windows stay open until closed; no PLY
export or model rerun is needed.

Ranking averages axis orientation error across a scene's eligible GT revolute
joints, requiring a valid, correctly typed axis estimate for **all** of them.
Fixed joints do not contribute an undefined orientation score. This prevents
missing estimates from improving a scene's rank. It covers the saved first
50 test1 scenes, not the entire test1 split. The top ten scene indices are
47, 35, 41, 25, 21, 28, 10, 23, 32, 6. CSV/JSON rankings and all-joint trajectory
PNGs are saved in `runs/test1_first50_analysis/open3d_visualizations/`.

To show a specific whole scene:

```bash
python -m finetuning_pair.inference \
  --saved-run finetuning_pair/runs/test1_first50_analysis \
  --mode with_poke --scene-index 6 --show-pointcloud
```

To regenerate only the rankings and all-joint images, use `--list-top` instead
of `--show-pointcloud`. The viewer uses native Open3D visualization and Tk for
the companion image window; it does not initialize a browser or reload CUDA
model weights. [Open3D visualization API](https://www.open3d.org/docs/release/python_api/open3d.visualization.VisualizerWithKeyCallback.html).

For a **new raw RGB-D pair inference**, append `--show-pointcloud --depth
current_depth.npy --intrinsics camera_K.npy` to the existing RGB/mask/point
inference command. Optionally provide `--extrinsics T_camera_world.npy` and
`--object-mask whole_object.png`. Depth values use world units; the extrinsic
defaults to identity (camera frame). This fits the selected pair's hypothesis-0
motion with predicted in-frame validity only, saves `joint_estimate.json` and
`pointcloud_inputs.npz`, and opens the same live viewer. The saved-scene command
above aggregates all of a scene's pair predictions.
