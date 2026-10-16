# FP32 training with AdamW on a second GPU

This separate trainer leaves the original `training_from_scratch/train.py` and
its checkpoints untouched. Run commands from the repository root in the
`myriad_finetune` environment.

The complete motion transformer and flow head, gradients, forward pass, and
loss use FP32 on `cuda:1`. AdamW's weights, gradients, and moments use FP32 on
`cuda:0`. After gradient accumulation, gradients are copied to `cuda:0`, clipped
there, and used by ordinary AdamW (`betas=(0.9, 0.99)`). Updated weights are
copied back to `cuda:1` before the next forward pass. Device synchronization
makes those dependencies explicit. This is one training process; use `python`,
not `torchrun`. There is no AMP, GradScaler, TF32, or splitting of model layers.

DINO weights are never allocated. The existing DINO cache is required and read
using its original namespace: FP32 encoder computation, FP16 saved features.
Features are widened to FP32 when loaded; their original FP16 quantization
remains. A cache miss stops the run with the missing filename. This trainer
does not compute or write feature entries. Keep the original encoder archive
available because compact checkpoints still validate and reference its identity.

## Continue the saved FP16 run

```bash
conda activate myriad_finetune
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m training_from_scratch.fp32_dual_gpu.train \
  --resume training_from_scratch/runs/robot_arm/latest.pt \
  --output-dir training_from_scratch/runs/robot_arm_fp32_dual_gpu
```

Training settings and the schedule are inherited from the source checkpoint.
The FP32 master weights overwrite the rounded FP16 model values. Adam moments,
learning-rate schedule, completed step, shuffled-scene cursor, and random states
are restored. The obsolete FP16 loss scaler is discarded. Starting again from
random weights is unnecessary. FP32 changes the arithmetic, so the future
trajectory is not expected to match the old FP16 run.

The defaults place the model on `cuda:1` and the optimizer on `cuda:0`.
`--gradient-checkpointing` is available but disabled unless inherited from the
checkpoint or explicitly enabled. `--no-gradient-checkpointing` disables it.
The CLI rejects changes to the resumed dataset, loss, effective batch, or
learning-rate schedule. For a fresh run, omit `--resume`; the motion model and
head are randomly initialized while DINO remains represented by cached features.

## Memory and bounded checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m training_from_scratch.fp32_dual_gpu.train \
  --resume training_from_scratch/runs/robot_arm/latest.pt \
  --output-dir training_from_scratch/runs/fp32_memory_check \
  --memory-check --memory-check-steps 5 --log-every-steps 1 \
  --evaluate-at-end --val-num-scenes 2 --test-num-scenes 1
```

`--memory-check` performs real training updates but saves no checkpoints.
`--run-steps N` instead stops after N updates and saves a resumable `latest.pt`.
`--evaluate-at-end` also runs the configured validation and test subsets at that
bounded stopping point. Without this flag, the normal validation/test intervals
apply. Evaluation also uses FP32. `memory.json` records allocated and reserved
CUDA peaks on both GPUs, cache hits/misses, and the last metrics.

For the 361,785,728 learned parameters, persistent FP32 model weights need about
1.35 GiB on `cuda:1`; model gradients need another 1.35 GiB before transfer.
The optimizer device needs about 5.39 GiB for its FP32 weights, gradients, and
two Adam moments once initialized. Activations, temporary tensors, allocator
reservation, CUDA context, and desktop VRAM are additional. Measure these on
the actual batch configuration with the command above.

On the two RTX 2080 SUPER 8 GiB cards, the five-update check from step 10,000
passed without activation checkpointing, including two validation examples and
one five-hypothesis test example. PyTorch peaks were:

| Device | Role | Allocated | Reserved |
| --- | --- | ---: | ---: |
| `cuda:1` | FP32 model | 4,053 MiB | 4,246 MiB |
| `cuda:0` | FP32 AdamW | 5,516 MiB | 5,562 MiB |

It used microbatch 1, accumulation 4, image size 384, 8 point slots, and four
transitions. All 23 requested embeddings were cache hits. After the first
update, the measured update times were approximately 1.5–1.85 seconds including
data loading and device transfers. See
`training_from_scratch/runs/fp32_dual_gpu_memory_20261015/memory.json`.

All eight numerical/migration/resume tests passed, including a real MYRIAD
two-GPU comparison with ordinary AdamW. A longer full-size run was stopped on
request after the last logged step **10,200**, with finite losses and gradients.
Its logs are in
`training_from_scratch/runs/fp32_dual_gpu_stability_20261015_stopped`.
It stopped before its checkpoint interval, so it saved no converted training
checkpoint. The original step-10,000 checkpoint is still the resume source;
the default FP32 output directory is available for a fresh invocation. No
training process is left running. Stability beyond this tested range has not
been established.

## Checkpoints and subsequent resumes

All new files go to the separate output directory. A legacy run cannot be
converted in its original directory. `latest.pt` saves FP32 model weights,
Adam moments, scheduler, RNG, and scene cursor. Optimizer weights are identical
to the saved FP32 model weights and are not duplicated. Expected tensor sizes
are about **4.04 GiB for latest.pt** and **1.35 GiB for best.pt**, which contains
only inference weights and the frozen encoder reference. `best.pt` is selected
by the same validation metric as the existing trainer and cannot resume Adam.

After the first converted checkpoint is saved, resume that new `latest.pt`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m training_from_scratch.fp32_dual_gpu.train \
  --resume training_from_scratch/runs/robot_arm_fp32_dual_gpu/latest.pt \
  --output-dir training_from_scratch/runs/robot_arm_fp32_dual_gpu
```

Unit tests (including the real two-GPU AdamW comparison when CUDA is available):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest training_from_scratch.fp32_dual_gpu.test_training -v
```
