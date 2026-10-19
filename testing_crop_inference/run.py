"""Compare the current best checkpoint on identical points, with live DINO.

python -B -u -m testing_crop_inference.run --output-dir testing_crop_inference/results
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from finetuning_pair.checkpoint_io import _validate_shapes
from finetuning_pair.conditioning import crop_geometry
from finetuning_pair.evaluate import file_hash, write_json
from training_from_scratch.checkpoint_io import _read_checkpoint, resolve_encoder_reference
from training_from_scratch.dataset import ArticulatedDataset, _backproject
from training_from_scratch.features import FeatureCache, load_frozen_encoder
from training_from_scratch.fp32_dual_gpu.model import CachedFP32Model, build_cached_myriad
from training_from_scratch.full_evaluation.metrics import clean, trajectory_metrics
from training_from_scratch.full_evaluation.run import atomic_npz, fit_metrics, load_source
from training_from_scratch.model import build_myriad
from training_from_scratch.testing import _scene_inputs, sample_rollout
from .geometry import SquareCrop, mask_crop, crop_sample, restore_predictions


class OnlineDINOModel(CachedFP32Model):
    """FP32 motion with freshly evaluated frozen DINO; no disk feature writes."""
    def attach_encoder(self, encoder, *, feature_dtype, comparison_cache=None):
        self.live_encoder = encoder
        self.feature_dtype = feature_dtype
        self.comparison_cache = comparison_cache
        self.verify_next = False
        self.feature_checks = []
        self.encoder_calls = 0

    def embed_image(self, image):
        from myriad.model import make_axial_pos_2d
        encoder_device = next(self.live_encoder.parameters()).device
        with torch.no_grad(), torch.autocast(device_type=encoder_device.type, enabled=False):
            features = self.live_encoder(image.to(encoder_device, torch.float32))
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite live DINO features")
        # Match training's feature storage rounding; encoder computation is FP32.
        features = features.to(self.feature_dtype)
        self.encoder_calls += 1
        b, _, h, w = features.shape
        pos = make_axial_pos_2d(h, w, device=self.training_device).reshape(1, h, w, 2).expand(b, -1, -1, -1)
        if self.verify_next and self.comparison_cache is not None:
            old = self.comparison_cache.read(image)
            self.feature_checks.append(dict(cache_found=old is not None,
                max_abs_difference=float((features.cpu().float()-old["feat"].float()).abs().max()) if old else None,
                positions_equal=bool(torch.equal(pos.cpu(), old["pos"])) if old else None))
            self.verify_next = False
        return dict(feat=features.to(self.training_device, torch.float32), pos=pos, L=h*w)


def load_model(path, cfg, device, dino_device, comparison_cache):
    payload = _read_checkpoint(path)
    if cfg.get("precision") != "fp32" or not cfg.get("freeze_dino"):
        raise ValueError("Expected the frozen-DINO FP32 best.pt")
    reference = resolve_encoder_reference(payload["frozen_encoder"])
    model = OnlineDINOModel(build_cached_myriad(image_size=cfg["image_size"])).place(device)
    state = {k: v for k, v in payload["model"].items() if not k.startswith("model.image_embedder.")}
    _validate_shapes(state, model.state_dict(), "Crop inference motion weights")
    model.load_state_dict(state, strict=True)
    del payload, state
    encoder = build_myriad(encoder_only=True)
    load_frozen_encoder(encoder, reference["path"])
    encoder.requires_grad_(False).eval().to(dino_device, torch.float32)
    feature_dtype = {"fp16": torch.float16, "fp32": torch.float32}[cfg.get("feature_cache_dtype", "fp16")]
    cache = FeatureCache(comparison_cache, identity=cfg["dino_identity"], image_size=cfg["image_size"],
                         dtype=torch.float32, feature_dtype=feature_dtype) if comparison_cache.is_dir() else None
    model.attach_encoder(encoder, feature_dtype=feature_dtype, comparison_cache=cache)
    model.requires_grad_(False).eval()
    return model


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "checkpoint.pt"
    if not snapshot.exists():
        os.link(args.checkpoint.resolve(), snapshot)
    sha = file_hash(snapshot)
    if file_hash(args.checkpoint) != sha:
        raise ValueError("Current best.pt differs from this output directory; choose a new output directory")
    payload = _read_checkpoint(snapshot)
    cfg, step = payload["train_config"], payload["step"]
    subset = json.loads(args.subset.read_text()) if args.subset else dict(
        scenes=list(range(320)), sampling="Entire test1 split; no score-based selection",
        evaluation="training_from_scratch/runs/test1_best_full_20261017")
    ds = ArticulatedDataset(cfg["data_root"], split="test1", training=False, seed=args.seed,
        points_per_link=cfg["points_per_link"], image_size=cfg["image_size"],
        num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
        overlay_alpha=cfg["overlay_alpha"], cache_dir=args.scene_cache)
    if args.subset is None:
        subset["scenes"] = list(range(len(ds.files)))
    indices = [i for i, (scene, pair) in enumerate(ds.examples) if scene in subset["scenes"]]
    if args.limit_pairs:
        indices = indices[:args.limit_pairs]
    signatures = [dict(path=str(ds.files[s].resolve()), size=ds.files[s].stat().st_size,
                       mtime_ns=ds.files[s].stat().st_mtime_ns) for s in sorted({ds.examples[i][0] for i in indices})]
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    manifest = clean(dict(checkpoint_sha256=sha, checkpoint_step=step,
        checkpoint_source=str(args.checkpoint.resolve()), train_config=cfg,
        split="test1", scene_subset=subset, pair_indices=indices, sources=signatures,
        variants={"original": None, "crop_10": .1}, seed=args.seed,
        ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps,
        motion_devices=args.devices or [str(args.device)], dino_devices=args.devices or [str(args.dino_device)],
        precision="FP32 DINO and motion, features rounded to checkpoint feature_cache_dtype",
        no_disk_feature_cache=True, scoring="all original valid states 1..4, all original keypoints; inverse crop to original normalized coordinates",
        crop_definition="square union of initial visible parent and child masks, margin per side, clipped extent and shifted inside original square view",
        minimum_padding_source_pixels=2, future_ground_truth_used_to_choose_crop=False,
        point_selection="unchanged original FPS and seed", code_sha256=code,
        torch_version=str(torch.__version__)))
    manifest_path = out / "run.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing run manifest differs; choose a new output directory")
    write_json(manifest_path, manifest)
    return ds, manifest


def launch_workers(args, argv):
    """Disjoint pairs on each GPU; each worker computes its own live DINO."""
    _, manifest = prepare(args)
    out = args.output_dir
    processes = []
    try:
        for index in range(len(args.devices)):
            processes.append(subprocess.Popen([sys.executable, "-B", "-u", "-m", "testing_crop_inference.run",
                *argv, "--worker-index", str(index)]))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("An evaluation worker failed; completed pair records can be resumed")
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("An evaluation worker failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    runtimes = [json.loads((out/f"runtime_worker_{i}.json").read_text()) for i in range(len(args.devices))]
    write_json(out/"runtime.json", dict(workers=runtimes, encoder_calls=sum(r["encoder_calls"] for r in runtimes)))
    if file_hash(out/"checkpoint.pt") != manifest["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed during evaluation")
    from .report import generate
    generate(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("training_from_scratch/runs/robot_arm_fp32_dual_gpu/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("testing_crop_inference/results"))
    p.add_argument("--subset", type=Path, help="Optional existing subset manifest; default evaluates all test1 scenes")
    p.add_argument("--scene-cache", type=Path, default=Path("training_from_scratch/cache"))
    p.add_argument("--comparison-cache", type=Path, default=Path("training_from_scratch/feature_cache"))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--dino-device", default="cuda:0")
    p.add_argument("--devices", nargs="+", help="Independent pair shards; each listed GPU loads DINO and motion together")
    p.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--limit-pairs", type=int)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    if min(args.ensemble_size, args.diffusion_steps) < 1 or args.seed < 0 or (args.limit_pairs is not None and args.limit_pairs < 1):
        p.error("Counts must be positive and seed nonnegative")
    if args.report_only:
        from .report import generate
        return generate(args.output_dir)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.devices and args.worker_index is None:
        return launch_workers(args, list(argv if argv is not None else sys.argv[1:]))
    device, dino_device = torch.device(args.device), torch.device(args.dino_device)
    if args.worker_index is not None:
        device = dino_device = torch.device(args.devices[args.worker_index])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run in the host GPU environment")
    torch.cuda.set_device(device)
    ds, manifest = prepare(args)
    cfg, out = manifest["train_config"], args.output_dir
    print(json.dumps(dict(event="dataset_ready", scenes=len(manifest["sources"]), pairs=len(manifest["pair_indices"]),
                         checkpoint_step=manifest["checkpoint_step"], variants=manifest["variants"])), flush=True)
    model = load_model(out/"checkpoint.pt", cfg, device, dino_device, args.comparison_cache)
    print(json.dumps(dict(event="models_loaded", motion_device=str(device), dino_device=str(dino_device))), flush=True)
    start_run, completed, new_calls = time.perf_counter(), 0, 0
    previous_scene, source = None, None
    old_eval = Path(manifest["scene_subset"]["evaluation"])
    previous_manifest = json.loads((old_eval/"run.json").read_text()) if (old_eval/"run.json").exists() else {}
    indices = manifest["pair_indices"]
    if args.worker_index is not None:
        indices = indices[args.worker_index::len(args.devices)]
    worker_suffix = f"_worker_{args.worker_index}" if args.worker_index is not None else ""
    for index in indices:
        scene, pair = ds.examples[index]
        if all((out/"records"/f"{index:04d}_{variant}.json").exists() for variant in manifest["variants"]):
            completed += 1
            continue
        sample = ds[index]
        if scene != previous_scene:
            source = load_source(sample["scene_path"])
            source["image_size"] = cfg["image_size"]
            previous_scene = scene
        h, w = source["initial_depth"].shape
        original = SquareCrop(*crop_geometry(h, w), cfg["image_size"])
        inputs = _scene_inputs(sample, False, False)
        base = dict(pair_index=index, scene_index=scene, joint_index=pair[2], joint_ordinal=pair[2]+1,
                    parent_link=pair[0], child_link=pair[1], sample_id=sample["sample_id"], object_id=sample["object_id"])
        jp = int(np.flatnonzero(source["joint_indices"] == pair[2])[0])
        base["gt_type"] = {0:"revolute", 1:"prismatic", 4:"fixed"}.get(int(source["joint_types"][jp]), "other")
        if inputs is None:
            for variant in manifest["variants"]:
                write_json(out/"records"/f"{index:04d}_{variant}.json", dict(**base, variant=variant, skipped=True,
                    reason="Original sample has no valid future targets"))
            completed += 1
            continue
        known, truth, valid, _ = inputs
        selected = sample["point_valid"] & sample["position_valid"][:1].all(0)
        pixels = sample["reference_pixels"][selected].numpy().astype(float)
        depth = source["initial_depth"][np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)]
        xyz = _backproject(pixels, depth, source["camera_intrinsic"], source["camera_extrinsic"])
        seg = source["initial_segmentation"]
        parent_mask, child_mask = seg == pair[0], seg == pair[1]
        for variant, margin in manifest["variants"].items():
            record_path = out/"records"/f"{index:04d}_{variant}.json"
            if record_path.exists():
                continue
            started = time.perf_counter()
            crop = original if margin is None else mask_crop(parent_mask, child_mask, margin=margin, image_size=cfg["image_size"])
            cropped = crop_sample(sample, source, crop)
            if variant == "original":
                if not torch.equal(cropped["x"], sample["x"]):
                    raise AssertionError("Original RGB reconstruction differs from existing pipeline")
                np.testing.assert_allclose(cropped["trajectories"], sample["trajectories"], atol=2e-7)
                model.verify_next = len(model.feature_checks) < 3
            crop_known = cropped["trajectories"][:1, selected]
            if not bool(((crop_known >= 0) & (crop_known <= 1)).all()):
                raise AssertionError("Crop lost a supplied initial keypoint")
            raw = sample_rollout(model, cropped["x"], crop_known, sample["state_times"],
                ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps,
                seed=args.seed+index, amp_dtype="none").float().cpu().numpy()
            pred = restore_predictions(raw, crop, original)
            if not np.isfinite(pred).all():
                raise FloatingPointError("Nonfinite predictions")
            # The sampling initial state is an observation, not a prediction.
            np.testing.assert_allclose(pred[:, 0], np.broadcast_to(known[0].numpy(), pred[:, 0].shape), atol=1e-6)
            truth_np, valid_np = truth.numpy(), valid.numpy()
            metrics, tracks = trajectory_metrics(pred, truth_np, valid_np, known_states=1,
                image_size=cfg["image_size"], gt_type=base["gt_type"], source_side=original.side)
            geometry = fit_metrics(pred, truth_np, valid_np, xyz, source, pair)
            geometry.pop("improves_stationary", None)
            cv = cropped["crop_position_valid"][:, selected].numpy()
            inside = ((raw[0] >= 0) & (raw[0] <= 1)).all(-1)
            row = dict(**base, variant=variant, skipped=False, crop=asdict(crop),
                zoom=original.side/crop.side, selected_points=np.flatnonzero(selected.numpy()).tolist(),
                valid_future_count=int(valid_np[1:].sum()),
                gt_outside_crop_fraction=float((~cv[1:])[valid_np[1:]].mean()),
                predicted_outside_crop_fraction=float((~inside[1:])[valid_np[1:]].mean()),
                child_initial_area_fraction=float(child_mask.mean()), **metrics)
            row.update(geometry)
            row["track_metrics"] = tracks
            row["feature_cache_checks"] = list(model.feature_checks)
            if variant == "original" and manifest["checkpoint_sha256"] == previous_manifest.get("checkpoint_sha256"):
                old = old_eval/"predictions"/f"{index:04d}_without_poke.npz"
                if old.exists():
                    with np.load(old, allow_pickle=False) as a:
                        np.testing.assert_array_equal(a["ground_truth"], truth_np)
                        np.testing.assert_array_equal(a["valid"], valid_np)
                        np.testing.assert_array_equal(a["point_indices"], np.flatnonzero(selected.numpy()))
                        if a["predictions"].shape == pred.shape:
                            row["previous_prediction_max_abs_pixel_difference"] = float(np.max(np.abs(a["predictions"]-pred))*(cfg["image_size"]-1))
            row["seconds"] = time.perf_counter()-started
            atomic_npz(out/"predictions"/f"{index:04d}_{variant}.npz", predictions=pred.astype(np.float32),
                cropped_predictions=raw, ground_truth=truth_np, cropped_ground_truth=cropped["trajectories"][:, selected].numpy(),
                valid=valid_np, crop_position_valid=cv, image=cropped["x"].numpy(),
                source_pixels=pixels, reference_world=xyz, point_indices=np.flatnonzero(selected.numpy()),
                crop_matrix=crop.matrix, camera_intrinsic_model=cropped["camera_intrinsic_model"].numpy(),
                source_path=np.array(sample["scene_path"]))
            write_json(record_path, clean(row))
            new_calls += 1
        completed += 1
        status = dict(event="progress", worker=args.worker_index, completed_pairs=completed, total_pairs=len(indices),
                      fresh_dino_calls=model.encoder_calls, seconds=time.perf_counter()-start_run,
                      mean_new_pair_seconds=(time.perf_counter()-start_run)/(new_calls/len(manifest["variants"])) if new_calls else None)
        write_json(out/f"status{worker_suffix}.json", status)
        if completed % 10 == 0 or completed == len(indices):
            print(json.dumps(status), flush=True)
    if file_hash(out/"checkpoint.pt") != manifest["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed during evaluation")
    write_json(out/f"runtime{worker_suffix}.json", dict(encoder_calls=model.encoder_calls, feature_checks=model.feature_checks,
        seconds=time.perf_counter()-start_run,
        peak_allocated_MiB={str(d):torch.cuda.max_memory_allocated(d)/1024**2 for d in {device, dino_device}}))
    if args.worker_index is None:
        from .report import generate
        generate(out)


if __name__ == "__main__":
    main()
