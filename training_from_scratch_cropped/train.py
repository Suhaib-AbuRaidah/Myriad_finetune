"""All-pair cropped FM training: FP32 cuda:1 model, FP32 cuda:0 AdamW."""
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
import time
import numpy as np
import torch

from training_from_scratch.train import make_scheduler, _append_log, _resolve_device
from training_from_scratch.fp32_dual_gpu.optimizer import DeviceAdamW
from .checkpoint_io import save_training_checkpoint, resume_checkpoint, inspect_checkpoint
from .config import parse_args, prepare, write_json, compare_config
from .dataset import collate_articulated
from .features import FeatureCache, fill_cache
from .model import CachedFP32Model, build_cached_myriad, training_loss
from .traversal import PairCursor
from .scoring import BestState
from .validation import evaluate


def synchronize(*devices):
    for device in set(map(torch.device,devices)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def gpu_memory(*devices):
    return [dict(device=str(d), allocated_MiB=torch.cuda.memory_allocated(d)/2**20,
        reserved_MiB=torch.cuda.memory_reserved(d)/2**20,
        peak_allocated_MiB=torch.cuda.max_memory_allocated(d)/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved(d)/2**20)
        for d in sorted(set(map(torch.device, devices)), key=str) if d.type == "cuda"]


def run(args, saved=None):
    if args.precompute_limit is not None:
        raise ValueError("--precompute-limit is only for the resumable precompute command")
    if args.resume and saved["kind"] != "training":
        raise ValueError("Resume requires latest.pt")
    if not args.resume and any((args.output_dir/name).exists() for name in ("latest.pt", "best.pt", "metrics.jsonl")):
        raise FileExistsError("Existing run: resume latest.pt or choose a new output directory")
    device, opt_device = _resolve_device(args.device), _resolve_device(args.optimizer_device)
    if device.type != "cuda" or opt_device.type != "cuda" or device == opt_device:
        raise ValueError("Real training requires two distinct CUDA devices; CPU is supported by unit tests")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def emit(row):
        _append_log(args.output_dir/"metrics.jsonl", row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    train, val, manifest, namespace, config, reference = prepare(args, saved, emit=emit)
    if args.resume:
        latest = args.output_dir/"latest.pt"
        if latest.exists() and inspect_checkpoint(latest)["step"] > saved["step"]:
            raise ValueError("Output has a newer latest.pt; choose a new output directory")
        state = BestState(saved["metrics"]["best_state"])
        if saved["metrics"]["axis_cohort"] != manifest["axis_cohort"]:
            raise ValueError("Checkpoint axis cohort differs from validation manifest")
        if state.score is not None:
            best_path = args.output_dir/"best.pt"
            source_best = best_path if best_path.exists() else args.resume.parent/"best.pt"
            if not source_best.exists():
                raise FileNotFoundError("Saved best state requires its sibling best.pt")
            best_info = inspect_checkpoint(source_best)
            compare_config(best_info["train_config"],config)
            if best_info["step"] > saved["step"] or best_info["metrics"]["best_state"]["score"] != state.score:
                raise ValueError("best.pt is inconsistent with the resumed best-score state")
            if args.save_checkpoints and not best_path.exists():
                try:
                    os.link(source_best,best_path)
                except OSError:
                    shutil.copy2(source_best,best_path)
    cache = FeatureCache(args.feature_cache, namespace, require_cached=not args.precompute)
    # Complete and verify all required train/validation entries before optimizer
    # allocation. No encoder exists in the ensuing training process.
    paths, coverage = fill_cache(cache, [(train, range(len(train))), (val, [r["index"] for r in manifest["pairs"]])],
        checkpoint=reference["path"], device=device, emit=emit)
    if coverage["remaining"]:
        raise ValueError("Incomplete cropped feature cache")
    write_json(args.output_dir/"cache_coverage.json", dict(**coverage, namespace=namespace, images=cache.image_inventory,
                                                        paths=list(map(str,sorted(set(paths))))))
    cache.require_cached = True
    emit(dict(event="precompute_memory",gpus=gpu_memory(device,opt_device)))
    for gpu in {device,opt_device}:
        torch.cuda.reset_peak_memory_stats(gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = CachedFP32Model(build_cached_myriad(image_size=args.image_size),
                            gradient_checkpointing=args.gradient_checkpointing).place(device)
    model.frozen_encoder_reference = reference
    model.configure_cropped_cache(cache)
    config["optimizer_parameter_names"] = [n for n,p in model.named_parameters() if p.requires_grad]
    optimizer = DeviceAdamW(model.trainable_parameters(), device=opt_device,
                           lr=args.lr, weight_decay=args.weight_decay, betas=(.9,.99))
    scheduler = make_scheduler(optimizer, args.max_steps, args.warmup_steps)
    cursor, best, step = PairCursor(train.examples,args.seed), BestState(), 0
    metrics = {}
    if args.resume:
        step = resume_checkpoint(args.resume, model, optimizer, scheduler)
        cursor = PairCursor.from_state(saved["data_state"], train.examples, args.seed)
        best = BestState(saved["metrics"]["best_state"])
        metrics = saved["metrics"].get("last_metrics", {})
    start_step = step
    stop = min(args.max_steps, step+args.run_steps) if args.run_steps else args.max_steps
    emit(dict(event="model_ready", **model.parameter_report(), encoder_loaded=False,
              model_device=str(device), optimizer_device=str(opt_device), start_step=step))
    def save(name, *, weights_only=False):
        if not args.save_checkpoints:
            return
        save_training_checkpoint(args.output_dir/name, model, optimizer, scheduler, step=step,
            train_config=config, metrics=dict(best_state=best.state_dict(), last_metrics=metrics,
                axis_cohort=manifest["axis_cohort"], validation_manifest=manifest["sha256"]),
            data_state=cursor.state_dict(), weights_only=weights_only)
        emit(dict(event="checkpoint", name=name, step=step))
    updates = []
    while step < stop:
        model.train()
        synchronize(device,opt_device)
        started = time.perf_counter()
        batches = []
        completed_epochs = []
        for _ in range(args.gradient_accumulation_steps):
            indices = cursor.next_indices(args.batch_size)
            batches.append(collate_articulated([train[i] for i in indices]))
            if cursor.offset == len(train):
                completed_epochs.append(cursor.epoch)
        examples = sum(len(b["x"]) for b in batches)
        denominator = sum(int(b["flow_loss_mask"].sum()) for b in batches)
        if not denominator:
            raise ValueError("Accumulated pairs have no valid FM targets; abort without silently skipping")
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((),device=device)
        for batch in batches:
            if not batch["flow_loss_mask"].any():
                continue
            batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            loss = training_loss(model,batch,loss_denominator=torch.tensor(denominator,device=device))
            loss.backward()
            loss_sum += loss.detach()
        del batches, batch, loss
        synchronize(device)
        optimizer_start = time.perf_counter()
        optimizer.capture_model_gradients()
        norm = torch.nn.utils.clip_grad_norm_(optimizer.optimizer_parameters(), args.grad_clip,
                                             error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.step()
        synchronize(device,opt_device)
        optimizer_seconds = time.perf_counter()-optimizer_start
        elapsed = time.perf_counter()-started
        step += 1
        metrics = dict(train_loss=float(loss_sum), grad_norm=float(norm),
            lr=optimizer.param_groups[0]["lr"], update_seconds=elapsed,
            optimizer_transfer_update_seconds=optimizer_seconds,
            examples=examples, examples_per_second=examples/elapsed,
            seconds_per_example=elapsed/examples, epoch=cursor.epoch, pair_offset=cursor.offset)
        updates.append(metrics.copy())
        if step % args.log_every_steps == 0 or step == stop:
            emit(dict(event="train",step=step,**metrics,gpus=gpu_memory(device,opt_device)))
        for epoch in completed_epochs:
            coverage = cache.coverage(paths)
            if coverage["remaining"]:
                raise RuntimeError("Cache coverage lost at epoch completion")
            emit(dict(event="epoch_complete",epoch=epoch,visited_pairs=len(train),**coverage))
        if step % args.val_every_steps == 0 or (step == stop and args.evaluate_at_end):
            result = evaluate(model,val,manifest,diffusion_steps=args.val_diffusion_steps,
                              seed=args.val_seed,emit=emit)
            write_json(args.output_dir/"validation"/f"step_{step:08d}.json",result)
            score = result["overall"]["checkpoint_score"]
            improved = best.consider(score,step)
            metrics["validation_axis_score"] = score
            emit(dict(event="validation",step=step,**result["overall"],
                      validation_seconds=result["validation_seconds"],improved=improved))
            if improved:
                save("best.pt",weights_only=True)
            save("latest.pt")
        elif step % args.save_every_steps == 0 or step == stop:
            save("latest.pt")
    result = dict(event="run_complete",start_step=start_step,step=step,completed_updates=step-start_step,
        gpus=gpu_memory(device,opt_device), updates=updates,cache=cache.coverage(paths),
        encoder_loaded=False, best_state=best.state_dict())
    write_json(args.output_dir/"run_report.json",result)
    emit({k:v for k,v in result.items() if k != "updates"})
    return result


def main(argv=None):
    args,saved = parse_args(argv)
    if args.smoke_test:
        from .paths import ROOT
        with tempfile.TemporaryDirectory(prefix=".smoke-",dir=ROOT) as temporary:
            args.output_dir=Path(temporary)
            if not args.feature_cache_explicit:
                args.feature_cache=args.output_dir/"features"
            if not args.scene_cache_explicit:
                args.scene_cache=args.output_dir/"scenes"
            return run(args,saved)
    return run(args,saved)


if __name__ == "__main__":
    main()
