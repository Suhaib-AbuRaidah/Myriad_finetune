"""Cached-DINO FP32 training with AdamW on a separate device.

Run from the repository root with ``python -m training_from_scratch.fp32_dual_gpu.train``.
The existing trainer is deliberately not modified by this implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from training_from_scratch.dataset import ArticulatedDataset, collate_articulated
from training_from_scratch.paths import artifact_path
from training_from_scratch.train import SceneCursor, make_scheduler, _resume_config, _append_log, _resolve_device
from training_from_scratch.validation import ValidationConfig, evaluate_validation
from training_from_scratch.testing import TestConfig, run_test_subset
from .checkpoint_io import (
    IMPLEMENTATION, check_resume_config, inspect_checkpoint, make_encoder_reference,
    resolve_encoder_reference, resume_checkpoint, save_training_checkpoint,
)
from .model import CachedFP32Model, build_cached_myriad, training_loss
from .optimizer import DeviceAdamW


DEFAULTS = dict(
    data_root="data/Robotic_arm", batch_size=1, lr=1e-4, weight_decay=.01,
    gradient_accumulation_steps=4, gradient_checkpointing=False, warmup_steps=500,
    max_steps=200000, grad_clip=1., points_per_link=8, image_size=384,
    num_transitions=4, time_scale=4., max_train_scenes=None, seed=42, overlay_alpha=.85,
    feature_cache_dtype="fp16", save_every_steps=1000, log_every_steps=100,
    val_every_steps=1000, val_num_scenes=16, val_ensemble_size=1, val_diffusion_steps=16,
    val_seed=42, val_poke_mode="without", test_every_steps=1000, test_num_scenes=12,
    test_ensemble_size=5, test_diffusion_steps=16, test_seed=42, test_split="test1",
    test_poke_mode="without",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("training_from_scratch/runs/robot_arm_fp32_dual_gpu"))
    parser.add_argument("--device", default="cuda:1", help="FP32 model and forward/backward device")
    parser.add_argument("--optimizer-device", default="cuda:0", help="FP32 AdamW device")
    parser.add_argument("--dino-checkpoint", type=Path, help="Original encoder archive, metadata only")
    parser.add_argument("--cache-dir", type=Path, default=Path("training_from_scratch/cache"))
    parser.add_argument("--dino-feature-cache", type=Path, default=Path("training_from_scratch/feature_cache"))
    parser.add_argument("--memory-check", action="store_true", help="Perform updates without saving checkpoints")
    parser.add_argument("--memory-check-steps", type=int, default=5)
    parser.add_argument("--run-steps", type=int, help="Perform this many updates, then save and stop")
    parser.add_argument("--evaluate-at-end", action="store_true", help="Evaluate after the last bounded/test update")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif name in ("data_root",):
            parser.add_argument(flag, type=Path)
        elif name == "max_train_scenes":
            parser.add_argument(flag, type=int)
        else:
            parser.add_argument(flag, type=type(default))
    args = parser.parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one Python process, not torchrun: the two devices have different roles")
    saved = inspect_checkpoint(args.resume) if args.resume else None
    if saved and saved["kind"] != "training":
        parser.error("Resume requires a training checkpoint (latest.pt)")
    config = saved["train_config"] if saved else {}
    # Inherit the original objective and schedule when migrating an existing run.
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, default))
    args.data_root = Path(args.data_root)
    for key in ("output_dir", "cache_dir", "dino_feature_cache"):
        setattr(args, key, artifact_path(getattr(args, key)))
    args.precision, args.amp, args.dino_precision = "fp32", "none", "fp32"
    args.freeze_dino, args.exclude_occluded_loss, args.world_size = True, False, 1
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "points_per_link",
                "image_size", "num_transitions", "save_every_steps", "log_every_steps",
                "val_every_steps", "test_every_steps", "val_ensemble_size", "test_ensemble_size",
                "val_diffusion_steps", "test_diffusion_steps", "memory_check_steps"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("val_num_scenes", "test_num_scenes", "seed", "val_seed", "test_seed", "warmup_steps"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    if args.run_steps is not None and args.run_steps < 1:
        parser.error("--run-steps must be positive")
    if args.max_train_scenes is not None and args.max_train_scenes < 1:
        parser.error("--max-train-scenes must be positive")
    if args.warmup_steps >= args.max_steps:
        parser.error("--warmup-steps must be below --max-steps")
    for key in ("lr", "grad_clip", "time_scale", "weight_decay", "overlay_alpha"):
        if not math.isfinite(getattr(args, key)):
            parser.error(f"--{key.replace('_', '-')} must be finite")
    if min(args.lr, args.grad_clip, args.time_scale) <= 0 or args.weight_decay < 0:
        parser.error("Need positive lr/grad-clip/time-scale and nonnegative weight decay")
    if not 0 <= args.overlay_alpha <= 1 or args.num_transitions > 4:
        parser.error("Need overlay-alpha in [0,1] and at most four transitions")
    if args.feature_cache_dtype not in ("fp16", "fp32") or args.test_split not in ("test1", "test2"):
        parser.error("Invalid feature-cache-dtype or test-split")
    for key in ("val_poke_mode", "test_poke_mode"):
        if getattr(args, key) not in ("with", "without", "both"):
            parser.error(f"Invalid {key}")
        if args.num_transitions == 1 and getattr(args, key) != "without":
            parser.error("One-transition data supports only without-poke evaluation")
    if args.test_ensemble_size < 5:
        parser.error("Testing needs at least five hypotheses for BEST-5")
    return args, saved


def _memory(model, optimizer, *, step, start_step, args, metrics):
    gpus = []
    for device in sorted({model.training_device, optimizer.optimizer_device}, key=str):
        if device.type != "cuda":
            continue
        torch.cuda.synchronize(device)
        gpus.append(dict(device=str(device), name=torch.cuda.get_device_name(device),
            allocated_MiB=torch.cuda.memory_allocated(device) / 1024**2,
            reserved_MiB=torch.cuda.memory_reserved(device) / 1024**2,
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 1024**2,
            peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 1024**2))
    return dict(event="memory_check" if args.memory_check else "run_complete", step=step,
        completed_updates=step-start_step, device=str(model.training_device),
        optimizer_device=str(optimizer.optimizer_device), precision="fp32", amp="none",
        dino_loaded=False, feature_cache_dtype=args.feature_cache_dtype,
        feature_cache_hits=model.feature_cache.hits, feature_cache_misses=model.feature_cache.misses,
        trainable_weights_dtypes=sorted({str(p.dtype) for p in model.trainable_parameters()}),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size, gpus=gpus, metrics=metrics)


def run(args, saved=None):
    device, opt_device = _resolve_device(args.device), _resolve_device(args.optimizer_device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if opt_device.type == "cuda" and opt_device.index is None:
        opt_device = torch.device("cuda", torch.cuda.current_device())
    # Never reuse the original trainer's output directory during conversion.
    same_run = args.resume and args.resume.resolve().parent == args.output_dir.resolve()
    if same_run and (saved or inspect_checkpoint(args.resume))["train_config"].get("training_implementation") != IMPLEMENTATION:
        raise ValueError("Legacy conversion requires a separate output directory; original run must stay untouched")
    if not same_run and any((args.output_dir / name).exists() for name in ("latest.pt", "best.pt", "metrics.jsonl")):
        raise FileExistsError("Output directory already contains a run; resume its latest.pt or use a new directory")
    saved = saved or (inspect_checkpoint(args.resume) if args.resume else None)
    if saved and saved["frozen_encoder"]:
        reference = resolve_encoder_reference(saved["frozen_encoder"], args.dino_checkpoint)
    else:
        reference = make_encoder_reference(args.dino_checkpoint or Path("myriad_openset.pt"))
    args.dino_identity = reference["identity"]
    config = {**_resume_config(args), "training_implementation": IMPLEMENTATION,
              "feature_cache_dtype": args.feature_cache_dtype, "grad_clip": args.grad_clip}
    if saved:
        check_resume_config(saved["train_config"], config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda" or opt_device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # Use IEEE FP32 even on hardware that offers TF32 matrix multiplication.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        indent=2, sort_keys=True) + "\n")
    log_path = args.output_dir / "metrics.jsonl"

    def emit(record):
        _append_log(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    def dataset(split, *, training=False, seed=args.seed, include_final_image=False):
        return ArticulatedDataset(args.data_root, split=split, points_per_link=args.points_per_link,
            image_size=args.image_size, num_transitions=args.num_transitions, time_scale=args.time_scale,
            seed=seed, max_scenes=args.max_train_scenes if training else None,
            training=training, supervise_occluded=True, overlay_alpha=args.overlay_alpha,
            cache_dir=args.cache_dir, include_final_image=include_final_image)

    train_data = dataset("train", training=True)
    emit(dict(event="building_model", device=str(device), optimizer_device=str(opt_device),
              training_scenes=len(train_data), dino_loaded=False))
    model = CachedFP32Model(build_cached_myriad(image_size=args.image_size),
                           gradient_checkpointing=args.gradient_checkpointing).place(device)
    model.frozen_encoder_reference = reference
    model.configure_cache(args.dino_feature_cache, identity=args.dino_identity,
                          feature_dtype=args.feature_cache_dtype)
    model.train()
    config["optimizer_parameter_names"] = [name for name, p in model.named_parameters() if p.requires_grad]
    optimizer = DeviceAdamW(model.trainable_parameters(), device=opt_device,
                           lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .99))
    scheduler = make_scheduler(optimizer, args.max_steps, args.warmup_steps)
    cursor = SceneCursor(len(train_data), args.seed)
    step = 0
    if args.resume:
        emit(dict(event="restoring_checkpoint", path=str(args.resume)))
        step = resume_checkpoint(args.resume, model, optimizer, scheduler)
        if saved["data_state"] is None:
            raise ValueError("Checkpoint has no training-scene cursor")
        cursor = SceneCursor.from_state(saved["data_state"], size=len(train_data), seed=args.seed)
        emit(dict(event="resumed", step=step, **cursor.state_dict(),
                  source_precision=saved["train_config"].get("precision", "fp32")))
    emit({**model.parameter_report(), "event": "model_ready", "device": str(device),
          "optimizer_device": str(opt_device), "precision": "fp32", "amp": "none", "dino_loaded": False})
    start_step = step
    stop_step = min(args.max_steps, step + args.memory_check_steps) if args.memory_check else (
        min(args.max_steps, step + args.run_steps) if args.run_steps else args.max_steps)
    want_eval = not args.memory_check or args.evaluate_at_end
    val_data = dataset("val", seed=args.val_seed) if args.val_num_scenes and want_eval else None
    test_data = dataset(args.test_split, seed=args.test_seed, include_final_image=True) if args.test_num_scenes and want_eval else None
    val_modes = ("with", "without") if args.val_poke_mode == "both" else (args.val_poke_mode,)
    selection_mode = "without" if "without" in val_modes else "with"
    selection = dict(metric="minADE_1", mode=selection_mode, num_scenes=args.val_num_scenes,
        seed=args.val_seed, diffusion_steps=args.val_diffusion_steps,
        ensemble_size=args.val_ensemble_size, visible_only=False, amp_dtype="none")
    best_score = math.inf
    best_path = args.output_dir / "best.pt"
    if best_path.exists():
        best = inspect_checkpoint(best_path)
        check_resume_config(best["train_config"], config)
        if best["step"] > step or best["metrics"].get("best_selection") != selection:
            raise ValueError("Existing best.pt is newer than resume, or uses different validation settings")
        best_score = best["metrics"]["best_val_minADE_1"]
    metrics = {}

    def save(path, *, weights_only=False):
        save_training_checkpoint(path, model, optimizer, scheduler, step=step, train_config=config,
            metrics=metrics, data_state=cursor.state_dict(), checkpoint_storage="compact",
            weights_only=weights_only)
        emit(dict(event="best_checkpoint" if weights_only else "checkpoint", step=step, path=str(path)))

    empty_batches = 0
    while step < stop_step:
        model.train()
        started = time.perf_counter()
        batches = [collate_articulated([train_data[i] for i in cursor.next_indices(args.batch_size)])
                   for _ in range(args.gradient_accumulation_steps)]
        valid_count = sum(int(batch["flow_loss_mask"].sum()) for batch in batches)
        if not valid_count:
            empty_batches += 1
            if empty_batches >= 10 * len(train_data):
                raise ValueError("Repeated batches have no valid flow targets")
            continue
        empty_batches = 0
        optimizer.zero_grad(set_to_none=True)
        denominator = torch.tensor(valid_count, device=device)
        loss_sum = torch.zeros((), device=device)
        for batch in batches:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            loss = training_loss(model, batch, loss_denominator=denominator)
            loss.backward()
            loss_sum += loss.detach()
        del loss, batch, batches
        optimizer.capture_model_gradients()
        norm = torch.nn.utils.clip_grad_norm_(optimizer.optimizer_parameters(), args.grad_clip,
                                             error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.step()
        step += 1
        metrics = dict(train_loss=float(loss_sum), grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                       step_seconds=time.perf_counter()-started)
        if step % args.log_every_steps == 0 or step == stop_step:
            emit(dict(event="train", step=step, **metrics))
        save_due = not args.memory_check and (step % args.save_every_steps == 0 or step == stop_step)
        if save_due:
            save(args.output_dir / "latest.pt")
        end_eval = step == stop_step and args.evaluate_at_end
        if val_data is not None and (step % args.val_every_steps == 0 or step == args.max_steps or end_eval):
            candidate = None
            for mode in val_modes:
                result = evaluate_validation(model, val_data, ValidationConfig(
                    num_scenes=args.val_num_scenes, ensemble_size=args.val_ensemble_size,
                    diffusion_steps=args.val_diffusion_steps, seed=args.val_seed,
                    visible_only=False, amp_dtype="none", initial_poke=mode == "with"), device=device)
                emit(dict(event="validation", step=step, **result))
                metrics[f"validation_{mode}"] = result
                if mode == selection_mode:
                    candidate = float(result["minADE_1"])
            if candidate is not None and candidate < best_score:
                best_score = candidate
                metrics.update(best_val_minADE_1=best_score, best_val_step=step, best_selection=selection)
                if not args.memory_check:
                    save(best_path, weights_only=True)
        if test_data is not None and (step % args.test_every_steps == 0 or step == args.max_steps or end_eval):
            result = run_test_subset(model, test_data, TestConfig(
                num_scenes=args.test_num_scenes, ensemble_size=args.test_ensemble_size,
                diffusion_steps=args.test_diffusion_steps, seed=args.test_seed,
                poke_mode=args.test_poke_mode, visible_only=False, amp_dtype="none"),
                args.output_dir / "tests", step=step)
            emit(dict(event="test", **result))
    result = _memory(model, optimizer, step=step, start_step=start_step, args=args, metrics=metrics)
    (args.output_dir / "memory.json").write_text(json.dumps(result, indent=2) + "\n")
    emit(result)
    return result


def main(argv=None):
    args, saved = parse_args(argv)
    return run(args, saved)


if __name__ == "__main__":
    main()
