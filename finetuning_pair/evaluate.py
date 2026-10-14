"""Full-split evaluation, persisted rollouts, and presentation exports.

Run ``python -m finetuning_pair.evaluate --help``. Rendering saved results with
``--render-only`` needs neither a checkpoint nor a GPU and never loads a model.
The original MYRIAD implementation is not changed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.paths import artifact_path, PACKAGE_ROOT

MODES = ("with_poke", "without_poke")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_errors(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray, size: int) -> dict:
    distances = np.linalg.norm((predicted.astype(np.float64) - truth) * (size - 1), axis=-1)
    if not valid.any():
        raise ValueError("No valid positions to score")
    selected = distances[valid]
    final = distances[-1][valid[-1]]
    return {"ADE_pixels": float(selected.mean()),
            "RMSE_pixels": float(np.sqrt(np.square(selected).mean())),
            "FDE_pixels": float(final.mean()) if len(final) else None}


def scene_metrics(arrays: dict, errors: np.ndarray) -> dict:
    start, size = int(arrays["known_states"]), arrays["image"].shape[-1]
    truth, valid = arrays["ground_truth"][start:], arrays["valid"][start:]
    first = arrays["predictions"][0, start:]
    px = pixel_errors(first, truth, valid, size)
    best_index = int(np.argmin(errors[:5]))
    oracle = pixel_errors(arrays["predictions"][best_index, start:], truth, valid, size)
    stationary = np.broadcast_to(arrays["ground_truth"][start - 1], truth.shape)
    stationary_squared = np.square(2.0 * (stationary.astype(np.float64) - truth)).sum(-1)
    baseline = float(stationary_squared[valid].mean())
    outside = ((first < 0) | (first > 1)).any(-1)
    link_scores = {}
    squared = np.square(2.0 * (first.astype(np.float64) - truth)).sum(-1)
    for link in np.unique(arrays["link_nodes"]):
        mask = valid & (arrays["link_nodes"] == link)[None]
        if mask.any():
            link_scores[str(int(link))] = float(squared[mask].mean())
    return {
        "minADE_1": float(errors[0]), "minADE_5": float(errors[:5].min()),
        "minADE_N": float(errors.min()), "hypothesis_errors": errors.tolist(),
        "oracle_best_of_5_index": best_index, "first_scored_state": start,
        "points_conditioned": int(truth.shape[1]), "scored_point_states": int(valid.sum()),
        "ADE_1_pixels": px["ADE_pixels"], "RMSE_1_pixels": px["RMSE_pixels"],
        "FDE_1_pixels": px["FDE_pixels"], "oracle_5_ADE_pixels": oracle["ADE_pixels"],
        "stationary_minADE": baseline,
        "stationary_ADE_pixels": pixel_errors(stationary, truth, valid, size)["ADE_pixels"],
        "improves_stationary": float(errors[0]) < baseline,
        "out_of_frame_fraction_1": float(outside[valid].mean()),
        "per_link_minADE_1": link_scores,
    }


def select_cases(records: list[dict], count: int) -> dict[str, list[dict]]:
    valid = [row for row in records if not row["skipped"]]
    return {
        "top_10": sorted(valid, key=lambda r: (r["minADE_1"], r["index"]))[:count],
        "failures": sorted(valid, key=lambda r: (-r["minADE_1"], r["index"]))[:count],
    }


def aggregate_records(records: list[dict]) -> dict:
    scored = [row for row in records if not row["skipped"]]
    if not scored:
        raise ValueError("No scored scenes")
    distribution = {}
    for name in ("minADE_1", "minADE_5", "minADE_N", "ADE_1_pixels", "FDE_1_pixels",
                 "RMSE_1_pixels", "stationary_minADE", "stationary_ADE_pixels",
                 "out_of_frame_fraction_1"):
        values = np.array([r[name] for r in scored if r[name] is not None], dtype=np.float64)
        if len(values):
            distribution[name] = {
                "mean": math.fsum(values.tolist()) / len(values), "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
                "min": float(values.min()), "max": float(values.max()), "count": len(values),
            }
    return {
        "scenes_evaluated": len(scored), "scenes_skipped": len(records) - len(scored),
        "points_conditioned": sum(r["points_conditioned"] for r in scored),
        "scored_point_states": sum(r["scored_point_states"] for r in scored),
        "minADE_1": distribution["minADE_1"]["mean"],
        "minADE_5": distribution["minADE_5"]["mean"],
        "minADE_N": distribution["minADE_N"]["mean"],
        "beats_stationary_fraction": sum(r["improves_stationary"] for r in scored) / len(scored),
        "evaluation_seconds": math.fsum(r["evaluation_seconds"] for r in records),
        "distributions": distribution,
    }


def write_csv(path: Path, records: list[dict]) -> None:
    fields = ("index", "sample_id", "object_id", "mode", "skipped", "minADE_1", "minADE_5",
              "ADE_1_pixels", "FDE_1_pixels", "RMSE_1_pixels", "stationary_minADE",
              "stationary_ADE_pixels", "improves_stationary", "points_conditioned",
              "scored_point_states", "out_of_frame_fraction_1", "evaluation_seconds", "results_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def read_records(root: Path, mode: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((root / mode / "results").glob("*/metrics.json"))]


def common_horizon_comparison(root: Path, records: dict[str, list[dict]]) -> list[dict]:
    indexed = {mode: {r["index"]: r for r in rows if not r["skipped"]}
               for mode, rows in records.items()}
    rows = []
    for index in sorted(set(indexed[MODES[0]]) & set(indexed[MODES[1]])):
        pair = [indexed[m][index] for m in MODES]
        with np.load(root / pair[0]["results_file"], allow_pickle=False) as a, \
                np.load(root / pair[1]["results_file"], allow_pickle=False) as b:
            _, ia, ib = np.intersect1d(a["point_indices"], b["point_indices"], return_indices=True)
            if not len(ia):
                continue
            start = max(int(a["known_states"]), int(b["known_states"]))
            valid = a["valid"][start:, ia] & b["valid"][start:, ib]
            if not valid.any():
                continue
            row = {"index": index, "sample_id": pair[0]["sample_id"],
                   "object_id": pair[0]["object_id"], "first_scored_state": start,
                   "common_points": len(ia), "scored_point_states": int(valid.sum())}
            if not np.allclose(a["ground_truth"][:, ia], b["ground_truth"][:, ib]):
                raise ValueError("Ground-truth points differ between modes")
            for mode, archive, slots in zip(MODES, (a, b), (ia, ib)):
                difference = 2.0 * (archive["predictions"][:, start:, slots].astype(np.float64)
                                    - archive["ground_truth"][start:, slots])
                errors = np.square(difference).sum(-1)[:, valid].mean(-1)
                row[f"{mode}_minADE_1"] = float(errors[0])
                row[f"{mode}_minADE_5"] = float(errors[:5].min())
            row["poke_improvement_minADE_1"] = row["without_poke_minADE_1"] - row["with_poke_minADE_1"]
            rows.append(row)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", action="store_true",
                        help="Evaluate the original pretrained model directly, without adapters or a checkpoint copy")
    parser.add_argument("--dataset-config", type=Path,
                        help="With --pretrained, reuse dataset settings from a previous evaluation's run.json")
    parser.add_argument("--data-root", type=Path, default=Path("data/Robotic_arm"))
    parser.add_argument("--split", choices=("test1", "test2", "val", "train"), default="test1")
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_pair/runs/test1_presentation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "auto", "none"), default="bf16")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=10, help="Number of best and worst scenes per mode")
    parser.add_argument("--max-scenes", type=int, help="Optional smoke-test subset; default evaluates the entire split")
    parser.add_argument("--render-only", action="store_true", help="Regenerate exports exclusively from saved results")
    parser.add_argument("--parallel-devices", nargs=2, metavar=("WITH_POKE_DEVICE", "WITHOUT_POKE_DEVICE"),
                        help="Optional separate GPUs for concurrent runs")
    parser.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir = artifact_path(args.output_dir)
    if not args.render_only and args.worker_mode is None and args.checkpoint is None:
        parser.error("--checkpoint is required for evaluation")
    if args.dataset_config and not args.pretrained:
        parser.error("--dataset-config requires --pretrained")
    if args.ensemble_size < 5 or args.diffusion_steps < 1 or args.cases < 1 or args.seed < 0:
        parser.error("Need at least five hypotheses, positive steps/cases, and a nonnegative seed")
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be positive")
    return args


def checkpoint_path(root: Path, manifest: dict) -> Path:
    frozen = manifest.get("frozen_checkpoint")
    return root / frozen if frozen else Path(manifest["source_checkpoint"])


def prepare_run(args, info: dict, dataset) -> dict:
    root = args.output_dir
    settings = {"data_root": str(args.data_root.resolve()), "split": args.split,
                "device": args.device, "amp_dtype": args.amp_dtype,
                "ensemble_size": args.ensemble_size, "diffusion_steps": args.diffusion_steps,
                "seed": args.seed, "max_scenes": args.max_scenes, "visible_only": False,
                "target_definition": TARGET_DEFINITION, "overlay_alpha": dataset.overlay_alpha,
                "pair_examples": [[i, list(pair)] for i, pair in dataset.examples]}
    scenes = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in dataset.files]
    manifest_path = root / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings or manifest["dataset_files"] != scenes:
            raise ValueError("Existing output belongs to different evaluation settings/data")
        if manifest["source_checkpoint"] != str(args.checkpoint.resolve()):
            raise ValueError("Existing output belongs to a different checkpoint path")
        kind = "pretrained" if args.pretrained else "finetuned"
        if manifest.get("checkpoint_kind", "finetuned") != kind:
            raise ValueError("Existing output belongs to a different checkpoint kind")
        if file_hash(checkpoint_path(root, manifest)) != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum differs from the run manifest")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty output without a run manifest: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if args.pretrained:
        frozen = None
        source = args.checkpoint.resolve()
    else:
        frozen = root / "checkpoint.pt"
        temporary = root / "checkpoint.pt.tmp"
        shutil.copy2(args.checkpoint, temporary)
        temporary.replace(frozen)
        # Inspect the frozen copy, since training may atomically replace best.pt.
        from finetuning_pair.checkpoint_io import inspect_checkpoint
        info = inspect_checkpoint(frozen)
        source = frozen
    manifest = {"schema_version": 1, "status": "running", "settings": settings,
                "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_files": scenes,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_kind": "pretrained" if args.pretrained else "finetuned",
                "frozen_checkpoint": frozen.name if frozen else None,
                "checkpoint_sha256": file_hash(source), "checkpoint_step": info["step"],
                "checkpoint_metadata": {key: info[key] for key in ("base_checkpoint", "train_config", "lora_config")},
                "ranking_metric": "minADE_1", "prediction_index_for_visuals": 0,
                "original_source_hashes": {str(p): file_hash(p) for p in sorted(Path("myriad").glob("*.py"))}}
    write_json(manifest_path, manifest)
    return manifest


def capture_callback(root: Path, mode: str, total: int, records: list[dict], seed: int):
    from finetuning_pair.testing import _scene_inputs
    previous = time.perf_counter()

    def capture(index, sample, predicted, errors):
        nonlocal previous
        now = time.perf_counter()
        record = {"index": index, "sample_id": sample["sample_id"], "object_id": sample["object_id"],
                  "mode": mode, "scene_path": sample["scene_path"], "seed": seed + index,
                  "pair_labels": sample["pair_labels"].tolist(), "joint_index": sample["joint_index"],
                  "target_definition": TARGET_DEFINITION, "cache_key": sample["cache_key"],
                  "skipped": predicted is None, "evaluation_seconds": now - previous}
        folder = root / mode / "results" / f"scene_{index:04d}_{Path(sample['scene_path']).stem}"
        folder.mkdir(parents=True, exist_ok=True)
        if predicted is not None:
            start = 2 if mode == "with_poke" else 1
            known, truth, valid, links = _scene_inputs(sample, start == 2, False)
            selected = sample["point_valid"] & sample["position_valid"][:start].all(0)
            arrays = {"schema_version": np.array(2), "target_definition": np.array(TARGET_DEFINITION),
                      "pair_labels": sample["pair_labels"].numpy(), "joint_index": np.array(sample["joint_index"]),
                      "cache_key": np.array(sample["cache_key"]), "predictions": predicted.float().cpu().numpy(),
                      "ground_truth": truth.numpy(), "valid": valid.numpy(),
                      "visibility": sample["visibility"][:, selected].numpy(),
                      "position_valid": sample["position_valid"][:, selected].numpy(),
                      "point_indices": np.flatnonzero(selected.numpy()), "link_nodes": links.numpy(),
                      "link_labels": sample["point_link_labels"][selected].numpy(),
                      "state_times": sample["state_times"].numpy(), "state_fractions": sample["state_fractions"].numpy(),
                      "known_positions": known.numpy(), "known_states": np.array(start),
                      "image": sample["x"].numpy(), "final_image": sample["final_x"].numpy(),
                      "hypothesis_errors": errors.float().cpu().numpy()}
            with np.load(sample["scene_path"], allow_pickle=False) as source:
                arrays["source_metadata_json"] = source["metadata_json"]
            # Every panel shows the exact conditioned reference. Actual moving
            # renders would misrepresent a parent-frozen relative endpoint.
            frame = np.clip((sample["x"].numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
            arrays["rgb_frames"] = np.repeat(frame[None], len(arrays["state_times"]), axis=0)
            record.update(scene_metrics(arrays, arrays["hypothesis_errors"]))
            record["results_file"] = str((folder / "results.npz").relative_to(root))
            arrays["metadata_json"] = np.array(json.dumps(record, sort_keys=True))
            temporary = folder / "results.npz.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(folder / "results.npz")
        else:
            record["reason"] = "No conditionable and scoreable future points"
        # This JSON is the completion marker; interrupted partial NPZ writes
        # never count as a finished scene when resuming.
        write_json(folder / "metrics.json", record)
        records.append(record)
        previous = time.perf_counter()
        print(json.dumps({"event": "scene_complete", "mode": mode, "index": index,
                          "completed": len(records), "total": total,
                          "minADE_1": record.get("minADE_1"),
                          "seconds": round(record["evaluation_seconds"], 2)}), flush=True)
    return capture


def load_model(root: Path, device_name: str):
    import torch
    from finetuning_pair.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
    from finetuning_pair.model_adapter import FineTuningModel, LoRAConfig
    from finetuning_pair.train import _resolve_device, build_myriad
    manifest = json.loads((root / "run.json").read_text())
    source = checkpoint_path(root, manifest)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bare = build_myriad()
    if manifest.get("checkpoint_kind") == "pretrained":
        from dataclasses import asdict
        reference = load_pretrained_weights(bare, source)
        if asdict(reference) != manifest["checkpoint_metadata"]["base_checkpoint"]:
            raise ValueError("Pretrained checkpoint identity differs from the run manifest")
        model = bare.to(device)
    else:
        info = inspect_checkpoint(source)
        reference = load_pretrained_weights(bare, Path(info["base_checkpoint"]["path"]))
        model = FineTuningModel(bare, LoRAConfig(**info["lora_config"])).to(device)
        load_finetuned_weights(source, model, base_checkpoint=reference)
    model.eval()
    print(json.dumps({"event": "model_ready", "device": str(device),
                      "bf16_supported": torch.cuda.is_bf16_supported(including_emulation=False) if device.type == "cuda" else None}), flush=True)
    return model, device


def evaluate_mode(root, mode, model, device, dataset, settings):
    from finetuning_pair.validation import ValidationConfig, evaluate_validation
    records = read_records(root, mode)
    complete = {r["index"] for r in records}
    pending = [i for i in range(len(dataset)) if i not in complete]
    for record in records:
        if not record["skipped"] and not (root / record["results_file"]).is_file():
            raise ValueError(f"Saved scene archive missing for {mode} scene {record['index']}")
    if pending:
        config = ValidationConfig(num_scenes=len(dataset), ensemble_size=settings["ensemble_size"],
                                  diffusion_steps=settings["diffusion_steps"], seed=settings["seed"],
                                  amp_dtype=settings["amp_dtype"], initial_poke=mode == "with_poke")
        callback = capture_callback(root, mode, len(dataset), records, settings["seed"])
        try:
            summary = evaluate_validation(model, dataset, config, device=device,
                                          scene_indices=pending, scene_callback=callback)
            write_json(root / mode / "validation_pass.json", summary)
        except ValueError as error:
            if str(error) != "Selected validation scenes have no scoreable future positions":
                raise
    write_json(root / mode / "summary.json", aggregate_records(records))


def worker(root: Path, mode: str, device_name: str):
    from finetuning_pair.dataset import ArticulatedDataset
    manifest = json.loads((root / "run.json").read_text())
    settings = manifest["settings"]
    cfg = manifest["checkpoint_metadata"]["train_config"]
    dataset = ArticulatedDataset(settings["data_root"], split=settings["split"],
                                points_per_link=cfg["points_per_link"], image_size=cfg["image_size"],
                                num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
                                seed=settings["seed"], max_scenes=settings["max_scenes"],
                                training=False, include_final_image=True, overlay_alpha=cfg["overlay_alpha"],
                                cache_dir=PACKAGE_ROOT / "cache")
    model, device = load_model(root, device_name)
    evaluate_mode(root, mode, model, device, dataset, settings)
    print(json.dumps({"event": "mode_complete", "mode": mode}), flush=True)


def parallel_evaluation(root: Path, devices: list[str], modes: list[str]):
    if len(set(devices)) != 2:
        raise ValueError("Parallel runs require two different devices")
    processes, logs = [], []
    try:
        for mode, device in zip(MODES, devices):
            if mode not in modes:
                continue
            (root / mode).mkdir(exist_ok=True)
            log = (root / mode / "evaluation.log").open("w")
            logs.append(log)
            command = [sys.executable, "-u", "-m", "finetuning_pair.evaluate", "--worker-mode", mode,
                       "--device", device, "--output-dir", str(root)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append((mode, process))
            # CPU construction/loading temporarily holds several GB per model.
            # Let the first model move to its GPU before loading the next one.
            log_path = root / mode / "evaluation.log"
            while process.poll() is None and '"event": "model_ready"' not in log_path.read_text():
                time.sleep(2)
            if process.poll() is not None and process.returncode != 0:
                raise RuntimeError(f"Model loading failed for {mode}; see {log_path}")
        while any(p.poll() is None for _, p in processes):
            counts = {mode: len(list((root / mode / "results").glob("*/metrics.json"))) for mode in MODES}
            print(json.dumps({"event": "progress", "completed": counts}), flush=True)
            time.sleep(30)
        failures = [mode for mode, p in processes if p.returncode != 0]
        if failures:
            raise RuntimeError(f"Evaluation workers failed: {failures}; see each mode's evaluation.log")
    except KeyboardInterrupt:
        import signal
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _, process in processes:
            process.wait()
        raise
    finally:
        for log in logs:
            log.close()


def main(argv=None):
    args = parse_args(argv)
    if args.worker_mode:
        worker(args.output_dir, args.worker_mode, args.device)
        return
    if args.render_only:
        from finetuning_pair.evaluation_visuals import export_presentation
        export_presentation(args.output_dir, args.cases, force_render=True)
        return
    import torch
    from finetuning_pair.checkpoint_io import inspect_checkpoint
    from finetuning_pair.dataset import ArticulatedDataset

    # Use the frozen configuration when resuming even if best.pt has advanced.
    existing = args.output_dir / "run.json"
    if args.pretrained:
        from dataclasses import asdict
        from finetuning_pair.checkpoint_io import _base_reference
        if existing.exists():
            saved = json.loads(existing.read_text())
            info = {**saved["checkpoint_metadata"], "step": saved["checkpoint_step"]}
        else:
            cfg = {"points_per_link": 8, "image_size": 384, "num_transitions": 4, "time_scale": 4.0,
                   "overlay_alpha": 0.85, "target_definition": TARGET_DEFINITION}
            if args.dataset_config:
                cfg = json.loads(args.dataset_config.read_text())["checkpoint_metadata"]["train_config"]
            info = {"step": None, "train_config": cfg, "lora_config": None,
                    "base_checkpoint": asdict(_base_reference(args.checkpoint))}
        if args.dataset_config:
            requested = json.loads(args.dataset_config.read_text())["checkpoint_metadata"]["train_config"]
            if any(info["train_config"][k] != requested[k] for k in
                   ("points_per_link", "image_size", "num_transitions", "time_scale")):
                raise ValueError("Existing output uses different dataset preparation settings")
    else:
        info = inspect_checkpoint(checkpoint_path(args.output_dir, json.loads(existing.read_text()))
                                  if existing.exists() else args.checkpoint)
    cfg = info["train_config"]
    if cfg.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("Evaluation config has incompatible target definition")
    dataset = ArticulatedDataset(
        args.data_root, split=args.split, points_per_link=cfg["points_per_link"],
        image_size=cfg["image_size"], num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
        seed=args.seed, max_scenes=args.max_scenes, training=False, include_final_image=True, overlay_alpha=cfg["overlay_alpha"],
                                cache_dir=PACKAGE_ROOT / "cache")
    manifest = prepare_run(args, info, dataset)
    records = {mode: read_records(args.output_dir, mode) for mode in MODES}
    pending = {}
    for mode in MODES:
        complete = {r["index"] for r in records[mode]}
        pending[mode] = [i for i in range(len(dataset)) if i not in complete]
        for record in records[mode]:
            if not record["skipped"] and not (args.output_dir / record["results_file"]).is_file():
                raise ValueError(f"Saved scene archive missing for {mode} scene {record['index']}")
    print(json.dumps({"event": "evaluation_start", "checkpoint_step": manifest["checkpoint_step"],
                      "scenes_per_mode": len(dataset), "pending": {m: len(pending[m]) for m in MODES}}), flush=True)
    if any(pending.values()):
        if args.parallel_devices:
            manifest["devices_by_mode"] = dict(zip(MODES, args.parallel_devices))
            write_json(args.output_dir / "run.json", manifest)
            parallel_evaluation(args.output_dir, args.parallel_devices, [m for m in MODES if pending[m]])
        else:
            model, device = load_model(args.output_dir, args.device)
            manifest["devices_by_mode"] = {m: str(device) for m in MODES}
            write_json(args.output_dir / "run.json", manifest)
            for mode in MODES:
                evaluate_mode(args.output_dir, mode, model, device, dataset, manifest["settings"])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    manifest["status"] = "evaluation_complete"
    manifest["evaluation_completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "run.json", manifest)
    from finetuning_pair.evaluation_visuals import export_presentation
    export_presentation(args.output_dir, args.cases)


if __name__ == "__main__":
    main()
