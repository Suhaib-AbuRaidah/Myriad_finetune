# Train MYRIAD Large from scratch

Run commands from the repository root in the existing `myriad_finetune`
environment. This package trains the original `MyriadStepByStep_Large`
architecture: DINOv3-L/16, the 24-layer motion transformer, image/motion
projections, and flow-matching head. The default initializes and trains all
learned weights from scratch, including DINO. `--freeze-dino` instead loads
only DINO from the local `myriad_openset.pt` and trains the complete motion
model and flow head from scratch. No mode injects LoRA or downloads weights.
The architecture's fixed RoPE frequencies,
Fourier buffers, and flow-input scale constants retain their original behavior.

## FP16 on the two RTX 2080 SUPER cards

```bash
conda activate myriad_finetune
python -m training_from_scratch.train \
  --device cuda:0 --precision fp16 --freeze-dino --dino-device cuda:1 \
  --data-root data/Robotic_arm \
  --output-dir training_from_scratch/runs/fp16_robot_arm \
  --max-steps 10000 --gradient-checkpointing \
  --batch-size 1 --gradient-accumulation-steps 8
```

All **trainable** weights (the 362-million-parameter motion model and flow
head) are stored in FP16. Dynamic loss scaling protects small gradients.
FP32 master weights accumulate updates, and Adam's moments use FP32; pure
FP16 Adam would lose small updates and can underflow its epsilon. FP32 master
gradients are unscaled, synchronized and clipped before the update.
The motion transformer's residual stream accumulates in FP32: a real replay
of the step-6500 failure overflowed the last block's FP16 skip addition before
output normalization. Linear projections and attention still use FP16, and
learned weights stay FP16. This adds activation memory rather than converting
the entire model to FP32, and does not change checkpoint parameter names.
`--optimizer-device cpu` optionally offloads master weights, gradients and
Adam state to host RAM; this requires a single training process and incurs
CPU/transfer cost. Default optimizer placement is the training GPU.

Frozen pretrained DINO uses **FP32 on GPU 1**. A real check found nonfinite
values in its first layer's `layer_scale2` when converting that checkpoint to
FP16. The fixed flow-input scales also stay FP32 because they extend to 1e5,
which exceeds FP16 range. These do not receive gradients. The motion model
and flow head still start fresh and train in full, with no adapters.

Training zeros masked motion inputs before embedding and evaluates the flow
loss only on valid target tokens. A zero loss mask cannot remove NaNs after
they have already been computed (`NaN * 0` is still NaN).

An isolated nonfinite FP16 forward loss discards the whole accumulated update
and records a `nonfinite_forward` event in `metrics.jsonl`, including the
failing stage and sample IDs. Neither Adam nor the learning-rate scheduler
advances. The loss scale stays unchanged: lowering it only helps backward
gradient overflow. Training stops after three consecutive failed updates or
more than 100 such skips in one invocation. Nonfinite valid inputs, image
features or learned weights stop immediately with diagnostics. FP32 runs
continue to reject nonfinite losses immediately. Existing checkpoints remain
loadable; valid-token filtering changes noise draws for batches with padding.

Use `python`, not a two-process `torchrun` launch, for this device split.
Both cards are used by one process, with one training replica on GPU 0.

### Measure actual training memory

```bash
python -m training_from_scratch.train \
  --device cuda:0 --precision fp16 --freeze-dino --dino-device cuda:1 \
  --gradient-checkpointing --max-steps 10 --warmup-steps 0 \
  --max-train-scenes 2 --memory-check --memory-check-steps 3 \
  --output-dir training_from_scratch/runs/memory_check
```

This runs three real forward/backward/Adam updates on your data, reports
PyTorch's allocated and reserved GPU peaks, and writes `memory.json`.
It skips validation, test exports and checkpoint saves. Use a fresh output
directory when repeating it. These are short memory checks, not convergence
tests; validation ensembles and larger microbatches can need more memory.

The initial measured split-device check used 384-pixel inputs, eight child
points, four transitions, batch size one and activation checkpointing:

| Device | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| GPU 0: FP16 motion model + FP32 master Adam | 6,226 MiB | 6,566 MiB |
| GPU 1: frozen FP32 DINO | 1,198 MiB | 1,212 MiB |
| GPU 0: cached features, DINO on CPU | 6,226 MiB | 6,566 MiB |

Allocator statistics exclude the CUDA context and other processes. The run
completed all three updates with finite loss and gradients, without an overflow.
The cached run also completed three updates, with three cache hits, no misses,
and the same losses and gradient norms as the split-device run. GPU 1 is not
used during cached training. Machine-readable results are in
`runs/memory_fp16_split/memory.json` and `runs/memory_fp16_cached/memory.json`.

Repeating the same three-update checks **without activation checkpointing**
(omit `--gradient-checkpointing`) also completed with no overflows:

| Configuration | GPU 0 peak allocated | GPU 0 peak reserved | GPU 1 peak allocated |
| --- | ---: | ---: | ---: |
| DINO on GPU 1, checkpointing enabled | 6,226 MiB | 6,566 MiB | 1,198 MiB |
| DINO on GPU 1, checkpointing disabled | 6,223 MiB | 6,780 MiB | 1,198 MiB |
| Cached DINO, checkpointing enabled | 6,226 MiB | 6,566 MiB | Unused |
| Cached DINO, checkpointing disabled | 6,225 MiB | 6,766 MiB | Unused |

The overall allocated peak changes by only a few MiB at batch size one;
model/optimizer storage appears to dominate the overall peak in this setup.
Reserved memory increases by about 200–214 MiB. Final losses and gradient
norms match the checkpointed runs exactly. Larger microbatches or validation
ensembles may still require more memory. Reports are under
`runs/memory_fp16_split_no_checkpointing/` and
`runs/memory_fp16_cached_no_checkpointing/`, with the comparison in
`runs/checkpointing_comparison.json`. Checkpointing is off by default.

After the FP32 residual fix, a check resumed the real `robot_arm` checkpoint
at step 6000 and completed five updates with accumulation four and no
activation checkpointing. GPU 1 peaked at **6,369 MiB allocated / 7,160 MiB
reserved**; frozen DINO on GPU 0 used 1,157 MiB. All 20 feature lookups hit
the cache, all learned motion/head weights stayed FP16, and neither forward
nor gradient overflow occurred. This resume check used
`PYTORCH_ALLOC_CONF=expandable_segments:True` to avoid allocator fragmentation.
Its report is `runs/numerical_check_20261015/memory.json`; the earlier tables
use different accumulation settings and initial model weights.

### Cache DINO features and train using only GPU 0

Precompute all pair-specific conditioned images for the splits you use:

```bash
python -m training_from_scratch.cache_features \
  --device cuda:1 --splits train val test2 \
  --feature-cache training_from_scratch/feature_cache
```

Then train with DINO resident on CPU, requiring cache hits:

```bash
python -m training_from_scratch.train \
  --device cuda:0 --precision fp16 --freeze-dino --dino-device cpu \
  --dino-feature-cache training_from_scratch/feature_cache \
  --require-cached-features \
  --output-dir training_from_scratch/runs/fp16_cached \
  --max-steps 10000 --gradient-checkpointing \
  --batch-size 1 --gradient-accumulation-steps 8
```

DINO occupies no GPU memory in this mode. Its weights remain on CPU; compact
checkpoints reference the original DINO archive. Cache keys include encoder identity,
encoder/feature precision, encoder image size and the exact processed RGB
image, including the selected pair's color overlays. FPS point changes reuse
the same embedding. Features are FP16 and coordinates are FP32; each 384-pixel
feature map is approximately 1.13 MiB. Use the same `--image-size` and
`--overlay-alpha` in precomputation and training. `--max-scenes N` limits
precomputation for a quick check. Omitting `--require-cached-features` enables
lazy caching: missing images are encoded on `--dino-device` and saved.

## Check the setup first

```bash
conda activate myriad_finetune
python -m training_from_scratch.train \
  --data-root data/Robotic_arm \
  --max-steps 10000 --max-train-scenes 1 --dry-run
```

This loads one training example and constructs the full architecture on the
PyTorch `meta` device, which allocates no model weight storage. It reports the
parameter counts, batch shape, valid targets, and an estimate of FP32 Adam
memory for weights, gradients and optimizer moments. It performs no optimizer
updates. Remove `--max-train-scenes 1` for actual training on all scenes.

## Train

```bash
python -m training_from_scratch.train \
  --data-root data/Robotic_arm \
  --output-dir training_from_scratch/runs/robot_arm \
  --max-steps 10000 --warmup-steps 500 \
  --lr 1e-4 --weight-decay 0.01 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --gradient-checkpointing --amp auto
```

The learning rate and step count are starting settings for experiments, not
validated convergence settings for this dataset. Training from scratch can
need substantially more data and updates than fine-tuning and does not
guarantee better accuracy.

End-to-end FP32 training requires much more GPU memory, host RAM, and checkpoint disk
space than LoRA. The model has approximately 665 million parameters; FP32
weights, gradients and Adam moments alone require about 9.9 GiB per GPU,
before activations. This default end-to-end FP32 mode exceeds 8 GB. Use the
frozen-DINO FP16 commands above for the measured two-card configuration.
Activation checkpointing recomputes both DINO and motion-block activations;
accumulation reduces the microbatch size required for a desired effective
batch. Neither reduces parameter or Adam-state storage. `torchrun` replicates
the entire model and optimizer on each GPU; it does not pool VRAM or shard
the model. BF16 autocast is selected only with native CUDA BF16 support;
otherwise `auto` uses FP32. The image encoder still uses the original model's
384-pixel resolution even if dataset `--image-size` is reduced.

For multiple GPUs with sufficient memory **per GPU**:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m training_from_scratch.train \
  --device cuda --data-root data/Robotic_arm \
  --output-dir training_from_scratch/runs/robot_arm_2gpu \
  --max-steps 10000 --gradient-checkpointing \
  --batch-size 1 --gradient-accumulation-steps 8
```

`--batch-size` counts scenes per GPU per microbatch. The nominal effective
batch size is `batch-size × GPU count × gradient-accumulation-steps` (epoch-end
microbatches can be smaller). Loss normalization uses the total number of valid
child transitions over all microbatches and ranks. AdamW uses betas `(0.9, 0.99)`,
gradient clipping, and warmup followed by linear decay.

## Data and evaluation

`dataset.py` copies the `finetuning_pair/dataset.py` pipeline, with its cache
root redirected into this folder. It reads the existing schema-v1 NPZ files:

```text
data/Robotic_arm/
  train/scenes/*.npz
  val/scenes/*.npz
  test2/scenes/*.npz
  test1/scenes/*.npz
```

Each training scene visit chooses one eligible parent/child edge. The input
contains the same parent/child color overlays, center crop, resizing and
normalization as `finetuning_pair`. FPS selects eight initially visible child
points by default; missing points are padded and masked. Targets express
child motion relative to the parent frozen at its initial pose, with four
transitions toward the rest state. Teacher forcing, point IDs, time scaling,
attention masking and the flow-matching objective match the pair trainer.
Validation and testing enumerate pairs deterministically. This remains a
pair-relative model, with the same target semantics as the existing pipeline.

Validation runs every 500 optimizer updates on 16 pairs. `best.pt` is selected
by first-rollout validation `minADE_1`. Periodic test previews run every 1,000
updates on four `test2` pairs and save metrics, predictions, PNGs and GIFs.
Both also run at the final `--max-steps` update. Defaults evaluate without a
ground-truth motion hint. `--val-poke-mode both` / `--test-poke-mode both` add
the optional ground-truth first-motion diagnostic.

Use `--val-num-scenes 0 --test-num-scenes 0` to disable evaluation and avoid
requiring those splits, for example for a short training smoke test.
`--max-train-scenes N` limits training scenes for a small overfitting experiment.
Use `--warmup-steps 0` when `--max-steps` is below the default warmup of 500.

Validation, test exports and inference use PyTorch scaled-dot-product attention
for sampler prefill and incremental decoding. The original causal predicate is
evaluated as a boolean token mask, including fully masked image-prefix rows.
This avoids compiling the FlexAttention/Triton kernel that failed during
validation in the RTX 2080 SUPER environment. Attention scaling, KV-cache
behavior and model weights are preserved; existing checkpoints can be resumed
with the same training settings. Only the sampler's mask construction is
patched for the duration of a rollout; original repository files are unchanged.
The saved step-1000 model passed two real validation pairs and a five-hypothesis
test example with compiled FlexAttention blocked. The check report and rollout
artifacts are under `runs/attention_check_step1000/`; these are a bounded
execution check, not a full evaluation of model accuracy.

## Outputs and resume

All generated files stay under `training_from_scratch/`:

- `cache/`: deterministic scene indices and validation/test examples.
- `feature_cache/`: optional frozen DINO features for pair-conditioned RGB images.
- `runs/<run>/config.json`: CLI settings.
- `runs/<run>/metrics.jsonl`: training and evaluation history.
- `runs/<run>/latest.pt`: trained weights, optimizer, scheduler, scaler, RNG and scene cursor for exact resume.
- `runs/<run>/best.pt`: inference weights and metadata for the best validation score, without optimizer state.
- `runs/<run>/tests/`: periodic test artifacts.

`--checkpoint-storage compact` is the default. With `--freeze-dino`, both files
omit the unchanged encoder tensors and record the original DINO archive's path,
size, modification time, metadata identity and feature-cache identity. **Keep
`myriad_openset.pt`: compact checkpoints need it for resume and inference.**
The archive identity is an inexpensive size/metadata check, not a checksum of
every weight byte. Learned motion-model/head tensors and persistent constants
are saved without precision reduction.

For frozen FP32 DINO with an FP16 motion model/head:

| File | Contents | Approximate size |
| --- | --- | ---: |
| `latest.pt` | Motion weights, FP32 master weights, Adam state, scheduler, scaler, RNG and scene cursor | 4.7 GiB |
| `best.pt` | Motion weights, fixed constants, encoder reference and metadata | 0.67 GiB |

Use **`latest.pt` for resume**. New `best.pt` files support inference and
evaluation; exact training resume is deliberately rejected because they omit
Adam, master weights, scheduler, scaler, RNG and cursor state. They are never
copied over `latest.pt`.

`--checkpoint-storage full` embeds DINO as well, making the files independent
of the original archive. In this mode frozen-DINO FP16 `latest.pt` is about
5.8 GiB and inference-only `best.pt` about 1.8 GiB. When DINO itself is trained,
its learned tensors are always included regardless of the storage option.

Old version-1 full checkpoints remain readable, including for exact resume.
Existing files are not converted or rewritten automatically; subsequent saves
use the selected storage mode. Switching checkpoint storage does not change the
model, objective, sampling stream or optimizer settings. LoRA checkpoint
loaders remain incompatible.

Saving streams tensor storages through CPU and uses atomic replacement,
preserving the previous file if a save fails. Budget disk space for the existing
files **plus a temporary new file**. Compact frozen-DINO FP16 saves can require
about 10.1 GiB total for `latest.pt`, `best.pt` and a temporary new `latest.pt`.
Existing larger legacy files can temporarily increase this requirement.
Generated outputs and caches are ignored by Git within this folder.

Resume with the original training options and append:

```bash
--resume training_from_scratch/runs/robot_arm/latest.pt
```

Keep the dataset, model/optimizer settings, maximum steps, accumulation count,
GPU count and per-GPU batch size the same. Scene loading runs in the main
process so RNG state and scene order can be restored. `--run-steps N` stops
an invocation after N additional optimizer updates and saves a checkpoint,
while preserving the original `--max-steps` schedule. Use a new output directory
for a new experiment. Existing fine-tuning files are unchanged; shared geometry,
conditioning, validation, distributed and checkpoint utility functions are
imported from `finetuning_pair`.

If the original encoder archive is moved, pass `--dino-checkpoint /new/path/myriad_openset.pt`
on resume or inference. Its size and tensor metadata must match the reference;
feature-cache identity is retained. A version-1 full checkpoint can still be
resumed without its original DINO archive using `--checkpoint-storage full`.

## Inference

```bash
python -m training_from_scratch.inference \
  --checkpoint training_from_scratch/runs/robot_arm/best.pt \
  --image /path/to/rgb.png \
  --parent-mask /path/to/parent.png --child-mask /path/to/child.png \
  --points /path/to/child_points.npy \
  --output-dir training_from_scratch/runs/prediction
```

Points are unique `[N,2]` source-pixel `(x,y)` coordinates inside the visible
child mask and center crop. The loader reconstructs the full model from the
saved motion weights and the referenced DINO archive for compact checkpoints.
Full checkpoints include DINO and need no original archive. Outputs contain
relative predictions and rollout visualizations.

## Regression checks

```bash
python -m unittest discover -s training_from_scratch/tests -v
```

Tests use the real MYRIAD blocks with a small DINO configuration on CPU to check
dataset parity, encoder/transformer/head updates, activation checkpointing,
valid-token loss accumulation, full checkpoint restoration and exact resumed
training. They also check that test visualizations write into this folder.
With CUDA available, additional tests verify FP16 optimizer/scaler updates,
exact resumed master weights, CPU optimizer offload and overflow skipping.
They also force finite FP16 block contributions to overflow their residual
sum, verify FP32 accumulation prevents it with finite gradients, and check
that failed forward passes discard all accumulated gradients. Invalid data,
nonfinite weights and repeated failures remain fatal.
Checkpoint tests also cover compact training/inference restoration, legacy
full files, relocated/missing encoder archives, best/latest separation, and
preservation of the previous file after a simulated out-of-space save failure.
Sampler tests run the actual MYRIAD autoregressive loop on CPU and FP16 CUDA,
check causal masking and zero outputs for masked prefix rows, verify repeated
seeded rollouts, and assert that compiled FlexAttention is never called.
