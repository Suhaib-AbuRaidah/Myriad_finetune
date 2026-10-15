"""Train all learned MYRIAD Large weights from scratch on pair-relative data.

Run ``python -m training_from_scratch.train --help`` from the repository root.
The motion model and head start from scratch. DINO can train from scratch or
be frozen, initialized from the local checkpoint and moved to another GPU.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from training_from_scratch.checkpoint_io import (
    inspect_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
    make_encoder_reference,
    resolve_encoder_reference,
)
from finetuning_pair.geometry import TARGET_DEFINITION
from training_from_scratch.paths import artifact_path
from training_from_scratch.dataset import ArticulatedDataset, collate_articulated
from training_from_scratch.model import ScratchModel, build_myriad
from training_from_scratch.precision import MasterAdamW
from training_from_scratch.features import FeatureCache, encoder_identity, load_frozen_encoder
from finetuning_pair.parallel import ParallelContext
from training_from_scratch.testing import TestConfig, run_test_subset
from training_from_scratch.validation import ValidationConfig, evaluate_validation


def make_attention_mask(token_valid: torch.Tensor, image_tokens: int) -> torch.Tensor:
    """MYRIAD causal mask with invalid motion tokens removed on both axes.

    Layout is [image prefix, observed pokes, query tokens], with L pokes and
    L queries. Query j sees image tokens, itself, and pokes 0..j-1. Poke j
    sees image tokens and valid pokes 0..j. The prefix has no attention rows,
    as in MYRIAD's original ``query_causal_mask_mod``. A dense boolean mask
    selects PyTorch's scaled-dot-product attention path in the original block.
    """
    if token_valid.ndim != 2 or token_valid.dtype != torch.bool:
        raise ValueError("token_valid must be bool [batch, motion_tokens]")
    if image_tokens < 1 or token_valid.shape[1] < 1:
        raise ValueError("image and motion token counts must be positive")
    batch, length = token_valid.shape
    total = image_tokens + 2 * length
    allowed = torch.zeros((batch, total, total), device=token_valid.device, dtype=torch.bool)
    poke = slice(image_tokens, image_tokens + length)
    query = slice(image_tokens + length, total)
    allowed[:, poke, :image_tokens] = token_valid[:, :, None]
    allowed[:, query, :image_tokens] = token_valid[:, :, None]
    causal = torch.ones(length, length, device=token_valid.device, dtype=torch.bool).tril()
    prior = torch.ones(length, length, device=token_valid.device, dtype=torch.bool).tril(-1)
    pairs = token_valid[:, :, None] & token_valid[:, None, :]
    allowed[:, poke, poke] = pairs & causal
    allowed[:, query, poke] = pairs & prior
    diagonal = torch.eye(length, device=token_valid.device, dtype=torch.bool)
    allowed[:, query, query] = token_valid[:, :, None] & diagonal
    return allowed[:, None]


class NonfiniteFlowLoss(FloatingPointError):
    """Report the failing stage without retaining the autograd graph."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__("Nonfinite flow-matching loss: " + json.dumps(self.details, sort_keys=True))


def training_loss(model: nn.Module, batch: Mapping[str, Any], *, amp_dtype: str,
                  loss_denominator: torch.Tensor | None = None) -> torch.Tensor:
    """One teacher-forced flow-matching objective with a batch-specific mask."""
    definitions = batch.get("target_definition")
    if not definitions or any(definition != TARGET_DEFINITION for definition in definitions):
        raise ValueError("Training requires child-only frozen-parent relative targets")
    device = batch["x"].device
    valid = batch["flow_loss_mask"]
    token_valid = batch["motion_token_mask"]
    if valid.dtype != torch.bool or token_valid.dtype != torch.bool or valid.shape != token_valid.shape:
        raise ValueError("Expected matching boolean attention and flow-loss masks")
    if loss_denominator is None and not bool(valid.any()):
        raise ValueError("The training batch has no valid flow targets")
    length = valid.shape[1]
    amp = (
        torch.autocast(device_type=device.type, dtype=torch.float16 if amp_dtype == "fp16" else torch.bfloat16)
        if amp_dtype in ("bf16", "fp16") else nullcontext()
    )
    with amp:
        image = model.embed_image(batch["x"])
        mask = make_attention_mask(token_valid, int(image["L"]))
        is_query = torch.cat((torch.zeros_like(token_valid), torch.ones_like(token_valid)), dim=1)
        target = batch["flow_query"]

        def motion_input(name):
            value = batch[name]
            keep = token_valid[..., None] if value.ndim == 3 else token_valid
            # Masked coordinates can lie far outside the image or be undefined.
            # Attention masks do not prevent their embeddings from being computed.
            return torch.where(keep, value, torch.zeros_like(value))

        distribution = model(
            flow=torch.cat((motion_input("flow_poke"), torch.zeros_like(target)), dim=1),
            pos=torch.cat((motion_input("pos_poke"), motion_input("pos_query")), dim=1),
            pos_orig=torch.cat((motion_input("pos_orig_poke"), motion_input("pos_orig_query")), dim=1),
            t=torch.cat((motion_input("t_poke"), motion_input("t_query")), dim=1),
            track_id=torch.cat((motion_input("id_poke"), motion_input("id_query")), dim=1),
            is_query=is_query,
            camera_static=batch["camera_static"],
            mask=mask,
            d_img=image,
            track_id_emb_table=None,
        )
        # Select before evaluating the head: multiplying a NaN by a zero mask
        # leaves NaN, and masking only after loss can also poison its backward.
        selected_target = target[valid]
        selected_distribution = distribution[:, length:][valid]
        per_token = selected_distribution.loss(selected_target)
        expected_shape = selected_target.shape[:-1]
        if per_token.shape != expected_shape:
            raise ValueError(f"Flow loss returned {tuple(per_token.shape)}, expected {tuple(expected_shape)}")
        denominator = valid.sum() if loss_denominator is None else loss_denominator
        if denominator.item() <= 0:
            raise ValueError("The global training batch has no valid flow targets")
        loss = per_token.float().sum() / denominator
    if not torch.isfinite(loss):
        # Diagnose only on failure so successful steps need no extra scans.
        bad_inputs = []
        for key in ("flow_query", "flow_poke", "pos_poke", "pos_query",
                    "pos_orig_poke", "pos_orig_query", "t_poke", "t_query"):
            selected = valid if key == "flow_query" else token_valid
            if not bool(torch.isfinite(batch[key][selected]).all()):
                bad_inputs.append(key)
        bad_image = [key for key, value in image.items()
                     if torch.is_tensor(value) and not bool(torch.isfinite(value).all())]
        cond = getattr(selected_distribution, "cond", None)
        source = ("inputs" if bad_inputs else "image_features" if bad_image else
                  "transformer" if cond is not None and not bool(torch.isfinite(cond).all()) else "flow_head")
        raise NonfiniteFlowLoss({
            "source": source, "sample_ids": batch.get("sample_id", []),
            "nonfinite_inputs": bad_inputs, "nonfinite_image_tensors": bad_image,
            "nonfinite_loss_tokens": int((~torch.isfinite(per_token)).sum()),
        })
    return loss


class SceneCursor:
    """Seeded epoch shuffle with a compact, exact resume position."""

    def __init__(self, size: int, seed: int, *, epoch: int = 0, offset: int = 0):
        if size < 1 or epoch < 0 or not 0 <= offset <= size:
            raise ValueError("Invalid scene cursor position")
        self.size, self.seed = size, seed
        self.epoch, self.offset = epoch, offset
        self._order: list[int] | None = None

    def next_indices(self, batch_size: int) -> list[int]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
            self._order = None
        if self._order is None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self._order = torch.randperm(self.size, generator=generator).tolist()
        end = min(self.offset + batch_size, self.size)
        indices = self._order[self.offset:end]
        self.offset = end
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset, "dataset_size": self.size}

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, size: int, seed: int) -> SceneCursor:
        if state.get("dataset_size") != size:
            raise ValueError("Training dataset size changed since the checkpoint")
        return cls(size, seed, epoch=state["epoch"], offset=state["offset"])


def make_scheduler(optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int):
    """Warm up then decay the learning rate to a small positive floor."""
    if max_steps < 1 or not 0 <= warmup_steps < max_steps:
        raise ValueError("Need 0 <= warmup_steps < max_steps")
    functions = []
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        floor = min(1.0, 1e-8 / base_lr)

        def multiplier(completed: int, floor: float = floor) -> float:
            if warmup_steps and completed < warmup_steps:
                return floor + (1 - floor) * (completed + 1) / warmup_steps
            progress = (completed - warmup_steps) / (max_steps - warmup_steps)
            return max(floor, 1 - (1 - floor) * progress)

        functions.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/Robotic_arm"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_from_scratch/runs/articulated"))
    parser.add_argument("--resume", type=Path, help="Resumable latest.pt or a legacy full training checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:index")
    parser.add_argument("--max-steps", type=_positive_int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Check one dataset batch and report architecture size without allocating model weights")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                        help="Storage dtype of learned model weights; FP16 uses loss scaling and FP32 master AdamW")
    parser.add_argument("--freeze-dino", action="store_true",
                        help="Train the whole motion model/head from scratch with a frozen pretrained DINO encoder")
    parser.add_argument("--dino-checkpoint", type=Path, default=Path("myriad_openset.pt"),
                        help="Local checkpoint used ONLY for frozen DINO weights")
    parser.add_argument("--dino-device", help="Device for frozen DINO, e.g. cuda:1 or cpu; default is training device")
    parser.add_argument("--dino-precision", choices=("fp32", "fp16"), default="fp32",
                        help="Frozen pretrained DINO needs FP32 range for this checkpoint")
    parser.add_argument("--dino-feature-cache", type=Path, help="Cache frozen DINO embeddings inside training_from_scratch")
    parser.add_argument("--require-cached-features", action="store_true", help="Fail on a cache miss instead of running DINO")
    parser.add_argument("--optimizer-device", choices=("model", "cpu"), default="model",
                        help="Keep FP32 master weights/Adam state on GPU or offload them to CPU (FP16 only)")
    parser.add_argument("--memory-check", action="store_true",
                        help="Run real forward/backward/Adam updates and report CUDA peaks without saving checkpoints")
    parser.add_argument("--memory-check-steps", type=_positive_int, default=3)
    parser.add_argument("--run-steps", type=_positive_int,
                        help="Stop this invocation after N updates, save, and resume later")
    parser.add_argument("--batch-size", type=_positive_int, default=1,
                        help="Scenes per GPU per microbatch; effective batch also includes world size and accumulation")
    parser.add_argument("--save-every-steps", type=_positive_int, default=1000,
                        help="Save resumable latest.pt at this optimizer-step interval")
    parser.add_argument("--checkpoint-storage", choices=("compact", "full"), default="compact",
                        help="Compact frozen-DINO checkpoints reference the original encoder file; full embeds its weights")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1,
                        help="Scene microbatches per optimizer update on each GPU")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute DINO and motion-block activations during backward")
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--points-per-link", type=_positive_int, default=8)
    parser.add_argument("--overlay-alpha", type=float, default=0.85,
                        help="Role-color weight in (1-alpha)*RGB + alpha*role_color")
    parser.add_argument("--cache-dir", type=Path, default=Path("training_from_scratch/cache"),
                        help="Deterministic pair-target cache for validation/testing")
    parser.add_argument("--image-size", type=_positive_int, default=384)
    parser.add_argument("--num-transitions", type=_positive_int, default=4)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--max-train-scenes", type=_positive_int)
    parser.add_argument("--exclude-occluded-loss", action="store_true")
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--amp", choices=("auto", "none", "bf16"), default="auto")
    parser.add_argument("--log-every-steps", type=_positive_int, default=100)
    parser.add_argument("--val-every-steps", type=_positive_int, default=1000)
    parser.add_argument("--val-num-scenes", type=_nonnegative_int, default=16, help="Pair examples; 0 disables validation")
    parser.add_argument("--val-ensemble-size", type=_positive_int, default=1,
                        help="Validation hypotheses; only the first is scored (default one)")
    parser.add_argument("--val-diffusion-steps", type=_positive_int, default=16)
    parser.add_argument(
        "--val-poke-mode", choices=("with", "without", "both"), default="without",
        help="Evaluate with the first observed motion, without it, or both",
    )
    parser.add_argument("--val-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--val-visible-only", action="store_true")
    parser.add_argument("--val-amp-dtype", choices=("auto", "none", "bf16"), default="auto")
    parser.add_argument("--test-every-steps", type=_positive_int, default=1000)
    parser.add_argument("--test-num-scenes", type=_nonnegative_int, default=12, help="Pair examples; 0 disables testing")
    parser.add_argument("--test-split", choices=("test1", "test2"), default="test1")
    parser.add_argument("--test-ensemble-size", type=_positive_int, default=5)
    parser.add_argument("--test-diffusion-steps", type=_positive_int, default=16)
    parser.add_argument("--test-poke-mode", choices=("with", "without", "both"), default="without")
    parser.add_argument("--test-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--test-visible-only", action="store_true")
    parser.add_argument("--test-amp-dtype", choices=("auto", "none", "bf16"), default="auto")
    args = parser.parse_args(argv)
    args.world_size = ParallelContext.from_environment().world_size
    args.output_dir = artifact_path(args.output_dir)
    args.cache_dir = artifact_path(args.cache_dir)
    if args.dino_feature_cache is not None:
        args.dino_feature_cache = artifact_path(args.dino_feature_cache)
    if (args.dino_device or args.dino_feature_cache) and not args.freeze_dino:
        parser.error("--dino-device and --dino-feature-cache require --freeze-dino")
    if args.require_cached_features and not args.dino_feature_cache:
        parser.error("--require-cached-features requires --dino-feature-cache")
    if args.precision == "fp16" and args.amp not in ("auto", "none"):
        parser.error("--precision fp16 uses FP16 autocast; omit --amp bf16")
    if args.optimizer_device == "cpu" and (args.precision != "fp16" or args.world_size > 1):
        parser.error("CPU optimizer offload requires FP16 and a single training process")
    if args.dino_device and args.world_size > 1:
        parser.error("A separate DINO device uses one training process; launch with python, not torchrun")
    if not math.isfinite(args.overlay_alpha) or not 0 <= args.overlay_alpha <= 1:
        parser.error("--overlay-alpha must be in [0,1]")
    if args.exclude_occluded_loss or args.val_visible_only or args.test_visible_only:
        parser.error("Relative occlusion requires counterfactual renders; use geometric validity")
    if args.num_transitions > 4:
        parser.error("Schema-v1 scenes provide four transitions to rest")
    if args.num_transitions == 1 and (args.val_poke_mode != "without" or args.test_poke_mode != "without"):
        parser.error("One-transition targets support only without-poke evaluation")
    if args.warmup_steps >= args.max_steps:
        parser.error("--warmup-steps must be below --max-steps")
    for name in ("lr", "grad_clip", "time_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("weight_decay",):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.test_ensemble_size < 5:
        parser.error("testing needs at least five hypotheses for BEST-5")
    return args


def _resolve_device(setting: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if setting == "auto" else torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("This trainer supports CPU or CUDA devices")
    return device


def _amp_setting(setting: str, device: torch.device) -> str:
    if setting == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else "none"
    if setting == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("CUDA bf16 was requested but is unsupported")
    if setting == "bf16" and device.type != "cuda":
        raise ValueError("Training bf16 autocast requires CUDA")
    return setting


def _resume_config(args: argparse.Namespace) -> dict[str, Any]:
    """Fields that define weights, data stream, or optimizer/schedule state."""
    keys = (
        "batch_size", "lr", "gradient_accumulation_steps", "gradient_checkpointing",
        "weight_decay", "warmup_steps", "max_steps",
        "points_per_link", "image_size", "num_transitions", "time_scale",
        "max_train_scenes", "exclude_occluded_loss", "seed", "amp", "overlay_alpha",
    )
    config = {"initialization": "random_all_learned_weights", "architecture": "myriad_large_dinov3_large",
              "target_definition": TARGET_DEFINITION, "pair_selection": "random_train_enumerated_eval_v1",
              "data_root": str(args.data_root.expanduser().resolve()), **{key: getattr(args, key) for key in keys}}
    if args.precision != "fp32" or args.freeze_dino:
        config.update(precision=args.precision, freeze_dino=args.freeze_dino, optimizer_device=args.optimizer_device)
        if args.freeze_dino:
            config.update(initialization="random_motion_pretrained_frozen_dino", dino_identity=args.dino_identity,
                          dino_precision=args.dino_precision)
    if args.world_size > 1:
        config.update(world_size=args.world_size, parallel_training="global_valid_token_sum_v1")
    return config


def _append_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), allow_nan=False, sort_keys=True) + "\n")


def _save(path: Path, model: ScratchModel, optimizer, scheduler, step: int,
          args: argparse.Namespace, cursor: SceneCursor,
          metrics: Mapping[str, Any], parallel: ParallelContext | None = None, scaler=None,
          weights_only=False) -> None:
    parallel = parallel or ParallelContext()
    data_state = parallel.checkpoint_data_state(cursor.state_dict())
    if not parallel.primary:
        return
    save_training_checkpoint(
        path, model, optimizer, scheduler, step=step,
        train_config=_resume_config(args),
        metrics=metrics, data_state=data_state, scaler=scaler,
        checkpoint_storage=args.checkpoint_storage, weights_only=weights_only,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    parallel = ParallelContext.from_environment()
    device = parallel.initialize(args.device, _resolve_device)
    try:
        _run_training(args, device, parallel)
    finally:
        parallel.close()


def _run_training(args: argparse.Namespace, device: torch.device, parallel: ParallelContext) -> None:
    amp_dtype = "fp16" if args.precision == "fp16" else _amp_setting(args.amp, device)
    if args.precision == "fp16" and device.type != "cuda" and not args.dry_run:
        raise ValueError("FP16 training requires CUDA; use FP32 for CPU regression checks")
    if args.memory_check and (device.type != "cuda" or parallel.enabled):
        raise ValueError("--memory-check requires one training process on CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    checkpoint_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"

    if not args.resume and (checkpoint_path.exists() or best_path.exists() or log_path.exists()):
        raise FileExistsError("Output directory already contains a run; use --resume or a new --output-dir")
    saved_info = inspect_checkpoint(args.resume) if args.resume else None
    if saved_info is not None and saved_info["kind"] != "training":
        raise ValueError("Resume requires latest.pt; best.pt contains inference weights only")
    encoder_reference = None
    if args.freeze_dino:
        args.dino_identity = (saved_info["train_config"]["dino_identity"] if saved_info
                              else encoder_identity(args.dino_checkpoint))
        if saved_info and saved_info["frozen_encoder"] is not None:
            encoder_reference = resolve_encoder_reference(saved_info["frozen_encoder"], args.dino_checkpoint)
        elif args.checkpoint_storage == "compact":
            encoder_reference = make_encoder_reference(args.dino_checkpoint)
            if encoder_reference["identity"] != args.dino_identity:
                raise ValueError("Compact saves need the original DINO file; use --checkpoint-storage full for a self-contained legacy resume")
    if saved_info is not None and saved_info["train_config"] != _resume_config(args):
        raise ValueError("Resume data/model/optimizer/schedule arguments differ from the saved run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_data = ArticulatedDataset(
        args.data_root, split="train", points_per_link=args.points_per_link,
        image_size=args.image_size, num_transitions=args.num_transitions,
        time_scale=args.time_scale, seed=args.seed, max_scenes=args.max_train_scenes,
        supervise_occluded=True, overlay_alpha=args.overlay_alpha, cache_dir=args.cache_dir,
    )
    if args.dry_run:
        batch = collate_articulated([train_data[0]])
        with torch.device("meta"):
            preview = ScratchModel(build_myriad(), freeze_dino=args.freeze_dino)
            preview.place(torch.device("meta"), precision=args.precision)
        print(json.dumps({"event": "dry_run", "training_scenes": len(train_data),
                          "image_shape": list(batch["x"].shape),
                          "valid_flow_targets": int(batch["flow_loss_mask"].sum()),
                          "target_definition": TARGET_DEFINITION,
                          **preview.parameter_report()}, sort_keys=True), flush=True)
        return
    if parallel.primary:
        (args.output_dir / "config.json").write_text(json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2, sort_keys=True) + "\n")
    val_data = ArticulatedDataset(
        args.data_root, split="val", points_per_link=args.points_per_link,
        image_size=args.image_size, num_transitions=args.num_transitions,
        time_scale=args.time_scale, seed=args.val_seed, training=False,
        overlay_alpha=args.overlay_alpha, cache_dir=args.cache_dir,
    ) if args.val_num_scenes and not args.memory_check else None
    test_data = ArticulatedDataset(
        args.data_root, split=args.test_split, points_per_link=args.points_per_link,
        image_size=args.image_size, num_transitions=args.num_transitions,
        time_scale=args.time_scale, seed=args.test_seed, training=False,
        include_final_image=True, overlay_alpha=args.overlay_alpha, cache_dir=args.cache_dir,
    ) if args.test_num_scenes and not args.memory_check else None
    # Construction uses local architecture configs and never opens a checkpoint
    # or downloads encoder weights. Stagger large CPU allocations across ranks.
    for loading_rank in range(parallel.world_size):
        if parallel.rank == loading_rank:
            model = ScratchModel(build_myriad(), gradient_checkpointing=args.gradient_checkpointing,
                                 freeze_dino=args.freeze_dino)
            if encoder_reference is not None:
                model.frozen_encoder_reference = encoder_reference
            if args.freeze_dino and not args.resume:
                load_frozen_encoder(model.image_embedder, args.dino_checkpoint)
            model.place(device, precision=args.precision,
                        dino_device=torch.device(args.dino_device) if args.dino_device else None,
                        dino_precision=args.dino_precision if args.freeze_dino else None)
            if args.dino_feature_cache:
                model.feature_cache = FeatureCache(args.dino_feature_cache, identity=args.dino_identity,
                    image_size=model.image_embedder.image_size,
                    dtype=next(model.image_embedder.parameters()).dtype,
                    feature_dtype=next(model.transformer.parameters()).dtype,
                    require_cached=args.require_cached_features)
        parallel.barrier()
    parallel.synchronize_initial_parameters(model.trainable_parameters())
    if parallel.enabled:
        random.seed(args.seed + parallel.rank)
        np.random.seed(args.seed + parallel.rank)
        torch.manual_seed(args.seed + parallel.rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + parallel.rank)
    model.train()
    if args.precision == "fp16":
        optimizer = MasterAdamW(model.trainable_parameters(),
            device="cpu" if args.optimizer_device == "cpu" else device,
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay, betas=(0.9, 0.99), foreach=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0) if args.precision == "fp16" else None
    scheduler = make_scheduler(optimizer, args.max_steps, args.warmup_steps)
    report = model.parameter_report()
    print(json.dumps({"event": "model_ready", "device": str(device), "rank": parallel.rank,
                      "world_size": parallel.world_size, "batch_per_gpu": args.batch_size,
                      "global_batch_size": args.batch_size * parallel.world_size * args.gradient_accumulation_steps, **report}, sort_keys=True), flush=True)

    step = 0
    cursor = SceneCursor(len(train_data), args.seed)
    if args.resume:
        step = resume_training_checkpoint(
            args.resume, model, optimizer, scheduler,
            restore_rng=not parallel.enabled, scaler=scaler,
            dino_checkpoint=args.dino_checkpoint if args.freeze_dino else None,
        )
        if saved_info is None or saved_info["data_state"] is None:
            raise ValueError("Resume checkpoint has no training-scene position")
        cursor = SceneCursor.from_state(saved_info["data_state"], size=len(train_data), seed=args.seed)
        parallel.restore_rank_rng(saved_info["data_state"])
        print(json.dumps({"event": "resumed", "step": step, **cursor.state_dict()}), flush=True)
    if step >= args.max_steps:
        print(json.dumps({"event": "already_complete", "step": step}), flush=True)
        return

    val_modes = ("with", "without") if args.val_poke_mode == "both" else (args.val_poke_mode,)
    val_configs = tuple(ValidationConfig(
        num_scenes=args.val_num_scenes, ensemble_size=args.val_ensemble_size,
        diffusion_steps=args.val_diffusion_steps, seed=args.val_seed,
        visible_only=args.val_visible_only, amp_dtype=args.val_amp_dtype,
        initial_poke=mode == "with",
    ) for mode in val_modes) if val_data is not None else ()
    selection_mode = "without" if "without" in val_modes else "with"
    selection_settings = {
        "metric": "minADE_1", "mode": selection_mode, "num_scenes": args.val_num_scenes,
        "seed": args.val_seed, "diffusion_steps": args.val_diffusion_steps,
        "ensemble_size": args.val_ensemble_size, "visible_only": args.val_visible_only,
        "amp_dtype": args.val_amp_dtype,
    }
    best_score = math.inf
    best_step = None
    if args.resume and best_path.exists():
        best_info = inspect_checkpoint(best_path)
        saved_selection = best_info["metrics"].get("best_selection")
        if best_info["train_config"] != _resume_config(args):
            raise ValueError("Existing best.pt belongs to a different run")
        if best_info["step"] > step:
            raise ValueError("best.pt is newer than the checkpoint selected for resume")
        if saved_selection == selection_settings:
            best_score = float(best_info["metrics"]["best_val_minADE_1"])
            best_step = best_info["step"]
        else:
            raise ValueError("Existing best.pt belongs to a different validation setup")
    test_config = TestConfig(
        num_scenes=max(1, args.test_num_scenes), ensemble_size=args.test_ensemble_size,
        diffusion_steps=args.test_diffusion_steps, seed=args.test_seed,
        poke_mode=args.test_poke_mode, visible_only=args.test_visible_only,
        amp_dtype=args.test_amp_dtype,
    )
    latest_metrics: dict[str, Any] = dict(saved_info["metrics"]) if saved_info else {}
    stop_step = min(args.max_steps, step + args.run_steps) if args.run_steps else args.max_steps
    if args.memory_check:
        stop_step = min(args.max_steps, step + args.memory_check_steps)
    empty_batches = 0
    overflow_batches = 0
    forward_overflow_batches = 0
    consecutive_forward_overflows = 0
    while step < stop_step:
        model.train()
        # Normalize all microbatches by the same global valid-token count.
        # Keep pending data on CPU; each GPU holds one microbatch at a time.
        batches = []
        for _ in range(args.gradient_accumulation_steps):
            indices = cursor.next_indices(args.batch_size * parallel.world_size)
            local_indices, dummy = parallel.local_indices(indices)
            batch = collate_articulated([train_data[index] for index in local_indices])
            if dummy:
                batch["flow_loss_mask"].zero_()
                batch["motion_token_mask"].zero_()
            batches.append(batch)
        local_valid = sum(int(batch["flow_loss_mask"].sum()) for batch in batches)
        global_valid = parallel.sum(torch.tensor(local_valid, device=device))
        if not bool(global_valid.item()):
            empty_batches += 1
            if empty_batches >= 10 * len(train_data):
                raise ValueError("Repeated training batches have no valid relative displacement targets")
            continue
        empty_batches = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        forward_error = None
        loss = None
        for batch in batches:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            try:
                loss = training_loss(model, batch, amp_dtype=amp_dtype, loss_denominator=global_valid)
            except NonfiniteFlowLoss as error:
                if scaler is None:
                    raise
                forward_error = error.details
                break
            (scaler.scale(loss) if scaler is not None else loss).backward()
            loss_sum += loss.detach()
        del loss, batch, batches
        # Every replica discards the entire accumulation if any rank failed.
        # Loss scaling cannot repair a forward overflow; leave its scale alone.
        if scaler is not None and bool(parallel.sum(torch.tensor(int(forward_error is not None), device=device))):
            optimizer.zero_grad(set_to_none=True)
            if forward_error is not None:
                forward_error = dict(forward_error, nonfinite_parameters=[
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not bool(torch.isfinite(parameter).all())
                ])
            errors = [forward_error]
            if parallel.enabled:
                errors = [None] * parallel.world_size
                torch.distributed.all_gather_object(errors, forward_error)
            errors = [error for error in errors if error is not None]
            forward_overflow_batches += 1
            consecutive_forward_overflows += 1
            record = {"event": "nonfinite_forward", "step": step, "errors": errors,
                      "skipped_accumulations": forward_overflow_batches,
                      "consecutive_skips": consecutive_forward_overflows,
                      "loss_scale": scaler.get_scale()}
            if parallel.primary:
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if any(error["source"] in ("inputs", "image_features") or error["nonfinite_parameters"]
                   for error in errors):
                raise FloatingPointError("Nonfinite training inputs, image features or model parameters; see nonfinite_forward diagnostics")
            if consecutive_forward_overflows >= 3 or forward_overflow_batches > 100:
                raise FloatingPointError("Repeated nonfinite FP16 forward losses; see nonfinite_forward diagnostics")
            continue
        if isinstance(optimizer, MasterAdamW):
            optimizer.capture_model_gradients()
            scaler.unscale_(optimizer)
            gradient_parameters = optimizer.optimizer_parameters()
        else:
            gradient_parameters = list(model.trainable_parameters())
        parallel.synchronize_gradients(gradient_parameters)
        reported_loss = parallel.sum(loss_sum)
        norm = torch.nn.utils.clip_grad_norm_(gradient_parameters, args.grad_clip)
        if not torch.isfinite(norm):
            if scaler is None:
                raise FloatingPointError("Nonfinite gradient norm")
            scaler.update(new_scale=scaler.get_scale() / 2)
            overflow_batches += 1
            if overflow_batches > 100:
                raise FloatingPointError("Repeated FP16 gradient overflow")
            continue
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        step += 1
        consecutive_forward_overflows = 0
        latest_metrics = {
            "train_loss": float(reported_loss),
            "grad_norm": float(norm.detach()),
            "lr": optimizer.param_groups[0]["lr"],
            "step_seconds": time.perf_counter() - started,
            "loss_scale": scaler.get_scale() if scaler else 1.0,
        }
        if parallel.primary and (step % args.log_every_steps == 0 or step == 1):
            record = {"event": "train", "step": step, **latest_metrics}
            _append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        # Persist completed updates before validation/inference can fail.
        save_due = not args.memory_check and (step % args.save_every_steps == 0 or step == stop_step)
        if save_due:
            _save(checkpoint_path, model, optimizer, scheduler, step,
                  args, cursor, latest_metrics, parallel, scaler=scaler)
            if parallel.primary:
                record = {"event": "checkpoint", "step": step, "path": str(checkpoint_path)}
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
            parallel.barrier()

        improved = False
        if val_data is not None and (step % args.val_every_steps == 0 or step == args.max_steps):
            if parallel.primary:
                validation_results = {}
                for mode, val_config in zip(val_modes, val_configs):
                    val_metrics = evaluate_validation(model, val_data, val_config, device=device)
                    validation_results[mode] = val_metrics
                    record = {"event": "validation", "step": step, **val_metrics}
                    _append_log(log_path, record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    prefix = "val_" if mode == "with" else "val_no_poke_"
                    latest_metrics.update({f"{prefix}{key}": value for key, value in val_metrics.items()})
                candidate = float(validation_results[selection_mode]["minADE_1"])
                if candidate < best_score:
                    best_score, best_step, improved = candidate, step, True
                    latest_metrics.update({
                        "best_val_minADE_1": best_score, "best_val_step": best_step,
                        "best_selection": selection_settings,
                    })
            improved = parallel.primary_flag(improved, device)

        if improved:
            _save(best_path, model, optimizer, scheduler, step,
                  args, cursor, latest_metrics, parallel, scaler=scaler, weights_only=True)
            if parallel.primary:
                record = {"event": "best_checkpoint", "step": step, "path": str(best_path),
                          "selection_mode": selection_mode, "val_minADE_1": best_score}
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

        if test_data is not None and (step % args.test_every_steps == 0 or step == args.max_steps):
            if parallel.primary:
                test_result = run_test_subset(
                    model, test_data, test_config, args.output_dir / "tests", step=step,
                )
                record = {"event": "test", **test_result}
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
            parallel.barrier()


    if args.memory_check:
        devices = {device}
        if args.dino_device and torch.device(args.dino_device).type == "cuda":
            devices.add(torch.device(args.dino_device))
        memory = []
        for gpu in sorted(devices, key=str):
            torch.cuda.synchronize(gpu)
            memory.append({"device": str(gpu), "name": torch.cuda.get_device_name(gpu),
                "allocated_MiB": torch.cuda.memory_allocated(gpu) / 1024**2,
                "reserved_MiB": torch.cuda.memory_reserved(gpu) / 1024**2,
                "peak_allocated_MiB": torch.cuda.max_memory_allocated(gpu) / 1024**2,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved(gpu) / 1024**2})
        result = {"event": "memory_check", "completed_steps": step, "precision": args.precision,
                  "freeze_dino": args.freeze_dino, "optimizer_device": args.optimizer_device,
                  "dino_device": str(next(model.image_embedder.parameters()).device),
                  "dino_precision": str(next(model.image_embedder.parameters()).dtype),
                  "trainable_weights_dtypes": sorted({str(p.dtype) for p in model.trainable_parameters()}),
                  "batch_size": args.batch_size, "image_size": args.image_size,
                  "points_per_link": args.points_per_link, "num_transitions": args.num_transitions,
                  "gradient_checkpointing": args.gradient_checkpointing,
                  "gradient_accumulation_steps": args.gradient_accumulation_steps,
                  "feature_cache_hits": model.feature_cache.hits if model.feature_cache else 0,
                  "feature_cache_misses": model.feature_cache.misses if model.feature_cache else 0,
                  "overflow_batches": overflow_batches, "gpus": memory, "metrics": latest_metrics}
        result["forward_overflow_batches"] = forward_overflow_batches
        (args.output_dir / "memory.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)



if __name__ == "__main__":
    main()
