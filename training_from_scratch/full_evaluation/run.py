"""Evaluate a frozen best.pt on the complete test split using cached DINO.

python -m training_from_scratch.full_evaluation.run --checkpoint .../best.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from finetuning_pair.analyze_joints import fit_joint, geometry_metrics, STATES
from finetuning_pair.checkpoint_io import _validate_shapes
from finetuning_pair.evaluate import file_hash, write_json
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning.joint_parame_estim.run import from_normalized, axis_error
from training_from_scratch.checkpoint_io import _read_checkpoint, inspect_checkpoint, resolve_encoder_reference
from training_from_scratch.dataset import ArticulatedDataset, _backproject
from training_from_scratch.fp32_dual_gpu.model import CachedFP32Model, build_cached_myriad
from training_from_scratch.paths import artifact_path
from training_from_scratch.testing import _scene_inputs, sample_rollout
from .metrics import clean, trajectory_metrics, kinematic_reference, attach_reference

SOURCE_KEYS = (
    "joint_indices", "joint_types", "joint_axes_world", "joint_origins_world", "camera_intrinsic",
    "camera_extrinsic", "link_transforms_world", "link_indices", "metadata_json",
    "joint_parent_link_indices", "joint_child_link_indices",
)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_cached_weights(checkpoint, device, feature_cache):
    payload = _read_checkpoint(checkpoint)
    cfg = payload["train_config"]
    if cfg.get("precision") != "fp32" or not cfg.get("freeze_dino"):
        raise ValueError("This evaluation expects the cached-DINO FP32 checkpoint")
    reference = resolve_encoder_reference(payload["frozen_encoder"])
    model = CachedFP32Model(build_cached_myriad(image_size=cfg["image_size"])).place(device)
    state = {name: value for name, value in payload["model"].items() if not name.startswith("model.image_embedder.")}
    _validate_shapes(state, model.state_dict(), "Cached inference model")
    model.load_state_dict(state, strict=True)
    model.frozen_encoder_reference = reference
    model.configure_cache(feature_cache, identity=cfg["dino_identity"], feature_dtype=cfg.get("feature_cache_dtype", "fp16"))
    model.eval()
    return model


def make_dataset(args, cfg):
    return ArticulatedDataset(args.data_root or cfg["data_root"], split=args.split, training=False,
        seed=args.seed, points_per_link=cfg["points_per_link"], image_size=cfg["image_size"],
        num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
        overlay_alpha=cfg["overlay_alpha"], cache_dir=args.cache_dir)


def inventory(ds):
    entries, depths = [], {}
    for scene, path in enumerate(ds.files):
        with np.load(path, allow_pickle=False) as a:
            js, parents, children = a["joint_indices"], a["joint_parent_link_indices"], a["joint_child_link_indices"]
            depth = {int(label): 0 for label in set(a["link_indices"]) - set(children)}
            pending = set(range(len(js)))
            while pending:
                ready = [j for j in pending if int(parents[j]) in depth]
                if not ready:
                    raise ValueError(f"Invalid hierarchy in {path}")
                for j in ready:
                    depth[int(children[j])] = depth[int(parents[j])] + 1
                    depths[scene, int(js[j])] = depth[int(children[j])]
                    pending.remove(j)
            eligible = {pair[2] for pair in ds.pairs[scene]}
            for j, gt in zip(js, a["joint_types"]):
                entries.append(dict(scene_index=scene, joint_index=int(j), joint_ordinal=int(j)+1,
                    chain_depth=depths[scene, int(j)], gt_type={0:"revolute",1:"prismatic",4:"fixed"}.get(int(gt),"other"),
                    eligible=int(j) in eligible))
    return entries, depths


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "checkpoint.pt"
    if not snapshot.exists():
        os.link(args.checkpoint.resolve(), snapshot)
    elif snapshot.stat().st_ino != args.checkpoint.stat().st_ino:
        if file_hash(snapshot) != file_hash(args.checkpoint):
            raise ValueError("The source best.pt changed; use another output directory")
    info = inspect_checkpoint(snapshot)
    cfg = info["train_config"]
    if cfg.get("target_definition") != TARGET_DEFINITION or cfg["num_transitions"] != 4:
        raise ValueError("Requires the four-transition child-relative model")
    ds = make_dataset(args, cfg)
    entries, _ = inventory(ds)
    manifest = clean(dict(schema_version=1, checkpoint_source=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_hash(snapshot), checkpoint_step=info["step"], train_config=cfg,
        split=args.split, scenes=[dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in ds.files],
        pairs=ds.examples, inventory=entries, all_source_joints=len(entries), modes=args.modes,
        ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps, seed=args.seed,
        devices=args.devices, precision="fp32", amp="none", target_definition=TARGET_DEFINITION,
        min_path_pixels=args.min_path_pixels, geometry_states=STATES.tolist(), geometry_hypothesis=0,
        primary_hypothesis=0, feature_cache=str(args.feature_cache), torch_version=str(torch.__version__),
        joint_estimator="existing_unpenalized_robust_reprojection_with_3_source_pixel_stationary_guard"))
    target = out / "run.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Existing evaluation manifest differs; select another output directory")
    write_json(target, manifest)
    print(json.dumps(dict(event="dataset_ready", checkpoint_step=info["step"], scenes=len(ds.files),
        eligible_pairs=len(ds), source_joints=len(entries), excluded_joints=sum(not r["eligible"] for r in entries),
        modes=args.modes, expected_records=len(ds)*len(args.modes))), flush=True)
    return manifest


def load_source(path):
    with np.load(path, allow_pickle=False) as a:
        source = {key: a[key] for key in SOURCE_KEYS}
        source.update(initial_rgb=a["rgb"][0], initial_depth=a["depth"][0],
                      initial_segmentation=a["link_segmentation"][0])
    return source


def fit_metrics(pred, truth, valid, xyz, source, pair, *, control=False):
    k, e = source["camera_intrinsic"], source["camera_extrinsic"]
    h, w = source["initial_depth"].shape
    jp = int(np.flatnonzero(source["joint_indices"] == pair[2])[0])
    lookup = {int(label): node for node, label in enumerate(source["link_indices"])}
    parent, child = lookup[pair[0]], lookup[pair[1]]
    tracks = truth if control else pred[0]
    uv = from_normalized(tracks, h, w, source["image_size"])[STATES]
    fitted = fit_joint(xyz, uv, valid[STATES], k, e)
    row = geometry_metrics(fitted, source, jp, parent, child)
    row.update(skipped=False, improves_stationary=False)
    if row["gt_type"] == "prismatic" and row.get("predicted_type") == "prismatic":
        row["prismatic_axis_error_deg"] = axis_error(row["estimated_axis_world"], source["joint_axes_world"][0, jp])
    if row["gt_type"] == "revolute":
        attach_reference(row, kinematic_reference(source, jp, parent, child))
    return clean(row)


def worker(args):
    out = args.output_dir
    manifest = json.loads((out / "run.json").read_text())
    cfg = manifest["train_config"]
    device = torch.device(args.devices[args.worker_index])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = make_dataset(args, cfg)
    depths = {(r["scene_index"], r["joint_index"]): r["chain_depth"] for r in manifest["inventory"]}
    indices = list(range(args.worker_index, len(ds), len(args.devices)))
    model = None
    completed = 0
    source_scene, source = None, None
    started_run = time.perf_counter()

    def status(stage="inference"):
        write_json(out / f"status_worker_{args.worker_index}.json", dict(stage=stage, worker=args.worker_index,
            completed=completed, expected=len(indices)*len(args.modes), elapsed_seconds=time.perf_counter()-started_run,
            feature_cache_hits=model.feature_cache.hits if model else 0,
            feature_cache_misses=model.feature_cache.misses if model else 0))

    status("loading")
    for index in indices:
        if all((out / "records" / f"{index:04d}_{mode}.json").exists() for mode in args.modes):
            completed += len(args.modes)
            continue
        if model is None:
            model = load_cached_weights(out / "checkpoint.pt", device, args.feature_cache)
            print(json.dumps(dict(event="model_ready", worker=args.worker_index, device=str(device),
                                  step=manifest["checkpoint_step"], precision="fp32", dino_loaded=False)), flush=True)
        scene, pair = ds.examples[index]
        sample = ds[index]
        if source_scene != scene:
            source = load_source(sample["scene_path"])
            source["image_size"] = cfg["image_size"]
            source_scene = scene
            source_path = out / "sources" / f"scene_{scene:04d}.npz"
            if not source_path.exists():
                atomic_npz(source_path, **source)
        for mode in args.modes:
            record_path = out / "records" / f"{index:04d}_{mode}.json"
            if record_path.exists():
                completed += 1
                continue
            started = time.perf_counter()
            row = dict(pair_index=index, scene_index=scene, scene_path=sample["scene_path"],
                sample_id=sample["sample_id"], object_id=sample["object_id"],
                joint_index=pair[2], joint_ordinal=pair[2]+1, chain_depth=depths[scene, pair[2]],
                parent_link=pair[0], child_link=pair[1], mode=mode,
                gt_type={0:"revolute",1:"prismatic",4:"fixed"}.get(int(source["joint_types"][np.flatnonzero(source["joint_indices"] == pair[2])[0]]),"other"))
            inputs = _scene_inputs(sample, mode == "with_poke", False)
            if inputs is None:
                row.update(skipped=True, reason="No conditionable points with valid future targets")
            else:
                known, target, valid, links = inputs
                pred = sample_rollout(model, sample["x"], known, sample["state_times"],
                    ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps,
                    seed=args.seed+index, amp_dtype="none").float().cpu().numpy()
                truth, valid = target.numpy(), valid.numpy()
                start = len(known)
                metrics, tracks = trajectory_metrics(pred, truth, valid, known_states=start,
                    image_size=cfg["image_size"], gt_type=row["gt_type"], min_path_pixels=args.min_path_pixels,
                    source_side=min(source["initial_depth"].shape))
                selected = sample["point_valid"] & sample["position_valid"][:start].all(0)
                pixels = sample["reference_pixels"][selected].numpy().astype(float)
                depth = source["initial_depth"][np.rint(pixels[:,1]).astype(int), np.rint(pixels[:,0]).astype(int)]
                xyz = _backproject(pixels, depth, source["camera_intrinsic"], source["camera_extrinsic"])
                geometry = fit_metrics(pred, truth, valid, xyz, source, pair)
                geometry.pop("improves_stationary")
                row.update(skipped=False, **metrics)
                row.update(geometry)
                if mode == args.modes[0]:
                    row["ground_truth_control"] = dict(scene_index=scene, **fit_metrics(pred, truth, valid, xyz, source, pair, control=True))
                row["track_metrics"] = tracks
                atomic_npz(out / "predictions" / f"{index:04d}_{mode}.npz",
                    predictions=pred, ground_truth=truth, valid=valid, known_states=np.array(start),
                    image=sample["x"].numpy(), link_nodes=links.numpy(), reference_world=xyz,
                    source_pixels=pixels, state_times=sample["state_times"].numpy(),
                    pair_labels=np.array(pair[:2]), joint_index=np.array(pair[2]),
                    point_indices=np.flatnonzero(selected.numpy()),
                    target_definition=np.array(TARGET_DEFINITION), source_file=np.array(f"sources/scene_{scene:04d}.npz"))
            row["seconds"] = time.perf_counter()-started
            write_json(record_path, clean(row))
            completed += 1
            status()
            if completed % 10 == 0:
                print(json.dumps(dict(event="progress", worker=args.worker_index, completed=completed,
                    expected=len(indices)*len(args.modes), pair=index, mode=mode,
                    ADE=row.get("ADE_1_pixels"), seconds=row["seconds"])), flush=True)
    status("complete")
    if model:
        print(json.dumps(dict(event="worker_complete", worker=args.worker_index, completed=completed,
            cache_hits=model.feature_cache.hits, cache_misses=model.feature_cache.misses,
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("training_from_scratch/runs/robot_arm_fp32_dual_gpu/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("training_from_scratch/runs/test1_best_full_20261017"))
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split", default="test1", choices=("test1", "test2", "val"))
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    p.add_argument("--modes", nargs="+", choices=("without_poke", "with_poke"), default=["without_poke", "with_poke"])
    p.add_argument("--feature-cache", type=Path, default=Path("training_from_scratch/feature_cache"))
    p.add_argument("--cache-dir", type=Path, default=Path("training_from_scratch/cache"))
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-path-pixels", type=float, default=1.)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--worker-index", type=int)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    args.output_dir, args.feature_cache, args.cache_dir = map(artifact_path, (args.output_dir, args.feature_cache, args.cache_dir))
    if min(args.ensemble_size, args.diffusion_steps, args.threads) < 1 or args.seed < 0 or not np.isfinite(args.min_path_pixels) or args.min_path_pixels <= 0:
        p.error("Need positive counts/path threshold and nonnegative seed")
    if args.worker_index is not None:
        return worker(args)
    if args.report_only:
        from .report import generate_report
        return generate_report(args.output_dir)
    prepare(args)
    processes, handles = [], []
    try:
        # Each GPU processes disjoint pair indices with an independent model.
        # The per-pair seed makes results independent of scheduling/sharding.
        base = list(argv if argv is not None else sys.argv[1:])
        for index in range(len(args.devices)):
            handle = (args.output_dir / f"worker_{index}.log").open("a")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-B", "-u", "-m", __name__.replace("__main__", "training_from_scratch.full_evaluation.run"),
                *base, "--worker-index", str(index)], stdout=handle, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            time.sleep(20)
            statuses = [json.loads(path.read_text()) for path in args.output_dir.glob("status_worker_*.json")]
            print(json.dumps(dict(event="progress", workers=statuses)), flush=True)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("An inference worker failed; see worker logs. Completed records are resumable.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    from .complete_unscorable import complete
    complete(args.output_dir, args.devices[-1])
    from .report import generate_report
    generate_report(args.output_dir)
    from .audit import audit
    audit(args.output_dir)


if __name__ == "__main__":
    main()
