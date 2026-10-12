"""Single-device MYRIAD LoRA fine-tuning on rendered articulated scenes.

Run from the repository root with ``python -m finetuning.train --help``. The
original repository is left intact. This trainer needs PyTorch, NumPy, einops,
jaxtyping, torchvision and transformers. DINOv3-L is built from its architecture
locally; the supplied MYRIAD checkpoint initializes every model tensor before
LoRA injection, without fetching gated Hugging Face weights.

The DINOv3 image encoder and original transformer blocks stay frozen. All
mid_level in_proj/out_proj blocks receive LoRA. Motion/image projections,
output MLP and flow-matching head train normally. Attention and loss ignore
invalid/padded articulated point transitions. Validation scores a fixed subset
at a configurable interval. Scene loading uses the main process so the saved
torch RNG and scene cursor reproduce FPS sampling on resume.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from finetuning.checkpoint_io import (
    inspect_checkpoint,
    load_pretrained_weights,
    resume_training_checkpoint,
    save_training_checkpoint,
)
from finetuning.dataset import ArticulatedDataset, collate_articulated
from finetuning.model_adapter import FineTuningModel, LoRAConfig
from finetuning.testing import TestConfig, run_test_subset
from finetuning.validation import ValidationConfig, evaluate_validation


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


def training_loss(model: nn.Module, batch: Mapping[str, Any], *, amp_dtype: str) -> torch.Tensor:
    """One teacher-forced flow-matching objective with a batch-specific mask."""
    device = batch["x"].device
    valid = batch["flow_loss_mask"]
    token_valid = batch["motion_token_mask"]
    if valid.dtype != torch.bool or token_valid.dtype != torch.bool or valid.shape != token_valid.shape:
        raise ValueError("Expected matching boolean attention and flow-loss masks")
    if not bool(valid.any()):
        raise ValueError("The training batch has no valid flow targets")
    length = valid.shape[1]
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp_dtype == "bf16" and device.type == "cuda" else nullcontext()
    )
    with amp:
        image = model.embed_image(batch["x"])
        mask = make_attention_mask(token_valid, int(image["L"]))
        is_query = torch.cat((torch.zeros_like(token_valid), torch.ones_like(token_valid)), dim=1)
        target = batch["flow_query"]
        distribution = model(
            flow=torch.cat((batch["flow_poke"], torch.zeros_like(target)), dim=1),
            pos=torch.cat((batch["pos_poke"], batch["pos_query"]), dim=1),
            pos_orig=torch.cat((batch["pos_orig_poke"], batch["pos_orig_query"]), dim=1),
            t=torch.cat((batch["t_poke"], batch["t_query"]), dim=1),
            track_id=torch.cat((batch["id_poke"], batch["id_query"]), dim=1),
            is_query=is_query,
            camera_static=batch["camera_static"],
            mask=mask,
            d_img=image,
            track_id_emb_table=None,
        )
        per_token = distribution[:, length:].loss(target)
        if per_token.shape != valid.shape:
            raise ValueError(f"Flow loss returned {tuple(per_token.shape)}, expected {tuple(valid.shape)}")
        loss = (per_token.float() * valid).sum() / valid.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite flow-matching loss")
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
    """Warm up then decay both optimizer groups to a small positive floor."""
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
    parser.add_argument("--base-checkpoint", type=Path, default=Path("myriad_openset.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning/runs/articulated"))
    parser.add_argument("--resume", type=Path, help="Fine-tuning checkpoint to resume")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:index")
    parser.add_argument("--max-steps", type=_positive_int, required=True)
    parser.add_argument("--run-steps", type=_positive_int,
                        help="Stop this invocation after N updates, save, and resume later")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--lora-rank", type=_positive_int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--other-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--points-per-link", type=_positive_int, default=8)
    parser.add_argument("--image-size", type=_positive_int, default=384)
    parser.add_argument("--num-transitions", type=_positive_int, default=4)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--max-train-scenes", type=_positive_int)
    parser.add_argument("--exclude-occluded-loss", action="store_true")
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--amp", choices=("auto", "none", "bf16"), default="auto")
    parser.add_argument("--log-every-steps", type=_positive_int, default=20)
    parser.add_argument("--val-every-steps", type=_positive_int, default=500)
    parser.add_argument("--val-num-scenes", type=_positive_int, default=16)
    parser.add_argument("--val-ensemble-size", type=_positive_int, default=5)
    parser.add_argument("--val-diffusion-steps", type=_positive_int, default=16)
    parser.add_argument(
        "--val-poke-mode", choices=("with", "without", "both"), default="with",
        help="Evaluate with the first observed motion, without it, or both",
    )
    parser.add_argument("--val-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--val-visible-only", action="store_true")
    parser.add_argument("--val-amp-dtype", choices=("auto", "none", "bf16"), default="auto")
    parser.add_argument("--test-every-steps", type=_positive_int, default=1000)
    parser.add_argument("--test-num-scenes", type=_positive_int, default=4)
    parser.add_argument("--test-split", choices=("test1", "test2"), default="test2")
    parser.add_argument("--test-ensemble-size", type=_positive_int, default=5)
    parser.add_argument("--test-diffusion-steps", type=_positive_int, default=16)
    parser.add_argument("--test-poke-mode", choices=("with", "without", "both"), default="both")
    parser.add_argument("--test-seed", type=_nonnegative_int, default=42)
    parser.add_argument("--test-visible-only", action="store_true")
    parser.add_argument("--test-amp-dtype", choices=("auto", "none", "bf16"), default="auto")
    args = parser.parse_args(argv)
    if args.warmup_steps >= args.max_steps:
        parser.error("--warmup-steps must be below --max-steps")
    for name in ("lora_lr", "other_lr", "grad_clip", "time_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("weight_decay", "lora_weight_decay"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.val_ensemble_size < 5 or args.test_ensemble_size < 5:
        parser.error("validation and testing need at least five hypotheses for BEST-5")
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
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "none"
    if setting == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA bf16 was requested but is unsupported")
    if setting == "bf16" and device.type != "cuda":
        raise ValueError("Training bf16 autocast requires CUDA")
    return setting


def _resume_config(args: argparse.Namespace) -> dict[str, Any]:
    """Fields that define weights, data stream, or optimizer/schedule state."""
    keys = (
        "batch_size", "lora_rank", "lora_alpha", "lora_dropout", "lora_lr", "other_lr",
        "weight_decay", "lora_weight_decay", "warmup_steps", "max_steps",
        "points_per_link", "image_size", "num_transitions", "time_scale",
        "max_train_scenes", "exclude_occluded_loss", "seed", "amp",
    )
    return {"data_root": str(args.data_root.expanduser().resolve()), **{key: getattr(args, key) for key in keys}}


def _append_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), allow_nan=False, sort_keys=True) + "\n")


def _save(path: Path, model: FineTuningModel, optimizer, scheduler, step: int,
          base_reference, args: argparse.Namespace, cursor: SceneCursor,
          metrics: Mapping[str, Any]) -> None:
    save_training_checkpoint(
        path, model, optimizer, scheduler, step=step,
        base_checkpoint=base_reference, train_config=_resume_config(args),
        metrics=metrics, data_state=cursor.state_dict(),
    )


def _copy_checkpoint_atomically(source: Path, destination: Path) -> None:
    """Atomically retain a complete checkpoint, using a hard link when possible."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".pt", dir=destination.parent)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_myriad() -> nn.Module:
    """Construct the supplied checkpoint's DINOv3-L/16 without a Hub fetch.

    The frozen encoder has 24 layers, 1024 channels, 16 attention heads,
    4096-wide MLPs and four register tokens. Exact key/shape matching is
    checked by ``load_pretrained_weights`` before any training begins.
    """
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
    from myriad.model import MyriadStepByStep_Large

    config = DINOv3ViTConfig(
        image_size=384, patch_size=16, hidden_size=1024,
        intermediate_size=4096, num_hidden_layers=24,
        num_attention_heads=16, num_register_tokens=4,
    )

    def local_encoder(name: str, *args: Any, **kwargs: Any) -> nn.Module:
        if name != "facebook/dinov3-vitl16-pretrain-lvd1689m":
            raise ValueError(f"Unexpected DINO model requested: {name}")
        if args or kwargs:
            raise ValueError("Unexpected DINO construction arguments")
        encoder = DINOv3ViTModel(config)
        # The checkpoint was saved with encoder layers named ``layer.*``.
        # Transformers 5.17 nests them as ``model.layer.*``. Register the
        # same ModuleList under its checkpoint name and retain an unregistered
        # reference for the library's unchanged forward method. There is only
        # one set of layer parameters, so strict loading remains meaningful.
        inner = encoder.model
        encoder.layer = inner.layer
        del encoder.model
        object.__setattr__(encoder, "model", inner)
        return encoder

    # The original constructor calls AutoModel.from_pretrained internally.
    # Scope this replacement to construction; no repository code is changed.
    with patch.object(AutoModel, "from_pretrained", side_effect=local_encoder):
        return MyriadStepByStep_Large()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    amp_dtype = _amp_setting(args.amp, device)
    LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    checkpoint_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"

    saved_info = inspect_checkpoint(args.resume) if args.resume else None
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
        supervise_occluded=not args.exclude_occluded_loss,
    )
    val_data = ArticulatedDataset(
        args.data_root, split="val", points_per_link=args.points_per_link,
        image_size=args.image_size, num_transitions=args.num_transitions,
        time_scale=args.time_scale, seed=args.val_seed, training=False,
    )
    test_data = ArticulatedDataset(
        args.data_root, split=args.test_split, points_per_link=args.points_per_link,
        image_size=args.image_size, num_transitions=args.num_transitions,
        time_scale=args.time_scale, seed=args.test_seed, training=False,
        include_final_image=True,
    )
    # Lazy construction lets CLI and lightweight tests run without the large
    # vision dependencies or checkpoint.
    bare = build_myriad()
    base_reference = load_pretrained_weights(bare, args.base_checkpoint)
    model = FineTuningModel(
        bare, LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(
            lora_lr=args.lora_lr, other_lr=args.other_lr,
            weight_decay=args.weight_decay, lora_weight_decay=args.lora_weight_decay,
        ),
        betas=(0.9, 0.99),
    )
    scheduler = make_scheduler(optimizer, args.max_steps, args.warmup_steps)
    report = model.parameter_report()
    print(json.dumps({"event": "model_ready", "device": str(device), **report}, sort_keys=True), flush=True)

    step = 0
    cursor = SceneCursor(len(train_data), args.seed)
    if args.resume:
        step = resume_training_checkpoint(
            args.resume, model, optimizer, scheduler, base_checkpoint=base_reference
        )
        if saved_info is None or saved_info["data_state"] is None:
            raise ValueError("Resume checkpoint has no training-scene position")
        cursor = SceneCursor.from_state(saved_info["data_state"], size=len(train_data), seed=args.seed)
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
    ) for mode in val_modes)
    selection_mode = "without" if "without" in val_modes else "with"
    selection_settings = {
        "mode": selection_mode, "num_scenes": args.val_num_scenes,
        "seed": args.val_seed, "diffusion_steps": args.val_diffusion_steps,
        "ensemble_size": args.val_ensemble_size, "visible_only": args.val_visible_only,
        "amp_dtype": args.val_amp_dtype,
    }
    best_score = math.inf
    best_step = None
    if args.resume and best_path.exists():
        best_info = inspect_checkpoint(best_path)
        saved_selection = best_info["metrics"].get("best_selection")
        if best_info["train_config"] != _resume_config(args) or saved_selection != selection_settings:
            raise ValueError("Existing best.pt belongs to a different run or validation setup")
        if best_info["step"] > step:
            raise ValueError("best.pt is newer than the checkpoint selected for resume")
        best_score = float(best_info["metrics"]["best_val_minADE_5"])
        best_step = best_info["step"]
    elif not args.resume and best_path.exists():
        raise FileExistsError(f"{best_path} already exists; use --resume or a new --output-dir")
    test_config = TestConfig(
        num_scenes=args.test_num_scenes, ensemble_size=args.test_ensemble_size,
        diffusion_steps=args.test_diffusion_steps, seed=args.test_seed,
        poke_mode=args.test_poke_mode, visible_only=args.test_visible_only,
        amp_dtype=args.test_amp_dtype,
    )
    latest_metrics: dict[str, Any] = dict(saved_info["metrics"]) if saved_info else {}
    stop_step = min(args.max_steps, step + args.run_steps) if args.run_steps else args.max_steps
    while step < stop_step:
        model.train()
        indices = cursor.next_indices(args.batch_size)
        batch = collate_articulated([train_data[index] for index in indices])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = training_loss(model, batch, amp_dtype=amp_dtype)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), args.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
        scheduler.step()
        step += 1
        latest_metrics = {
            "train_loss": float(loss.detach()),
            "grad_norm": float(norm.detach()),
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_other": optimizer.param_groups[1]["lr"],
            "step_seconds": time.perf_counter() - started,
        }
        if step % args.log_every_steps == 0 or step == 1:
            record = {"event": "train", "step": step, **latest_metrics}
            _append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        improved = False
        if step % args.val_every_steps == 0 or step == args.max_steps:
            validation_results = {}
            for mode, val_config in zip(val_modes, val_configs):
                val_metrics = evaluate_validation(model, val_data, val_config, device=device)
                validation_results[mode] = val_metrics
                record = {"event": "validation", "step": step, **val_metrics}
                _append_log(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
                prefix = "val_" if mode == "with" else "val_no_poke_"
                latest_metrics.update({f"{prefix}{key}": value for key, value in val_metrics.items()})
            candidate = float(validation_results[selection_mode]["minADE_5"])
            if candidate < best_score:
                best_score, best_step, improved = candidate, step, True
                latest_metrics.update({
                    "best_val_minADE_5": best_score, "best_val_step": best_step,
                    "best_selection": selection_settings,
                })

        if improved:
            _save(best_path, model, optimizer, scheduler, step,
                  base_reference, args, cursor, latest_metrics)
            record = {"event": "best_checkpoint", "step": step, "path": str(best_path),
                      "selection_mode": selection_mode, "val_minADE_5": best_score}
            _append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if step == stop_step:
            if improved:
                _copy_checkpoint_atomically(best_path, checkpoint_path)
            else:
                _save(checkpoint_path, model, optimizer, scheduler, step,
                      base_reference, args, cursor, latest_metrics)
            record = {"event": "checkpoint", "step": step, "path": str(checkpoint_path)}
            _append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % args.test_every_steps == 0 or step == args.max_steps:
            test_result = run_test_subset(
                model, test_data, test_config, args.output_dir / "tests", step=step,
            )
            record = {"event": "test", **test_result}
            _append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
