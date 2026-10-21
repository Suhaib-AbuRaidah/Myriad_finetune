"""Resumable multi-GPU inference; save full visualization inputs without fitting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from finetuning_pair.testing import sample_rollout
from finetuning_pair.checkpoint_io import _validate_shapes
from training_from_scratch.dataset import _backproject
from training_from_scratch.sampling import sampler_attention_context
from training_from_scratch.full_evaluation.run import atomic_npz
from training_from_scratch.full_evaluation.metrics import clean
from training_from_scratch_cropped.checkpoint_io import _read_checkpoint
from training_from_scratch_cropped.config import write_json
from training_from_scratch_cropped.features import FeatureCache
from training_from_scratch_cropped.geometry import SquareCrop
from training_from_scratch_cropped.model import CachedFP32Model, build_cached_myriad
from training_from_scratch_cropped.validation import trajectory_diagnostics
from .common import DEFAULT_OUT, DEFAULT_CHECKPOINT, prepare, make_dataset


def trajectory_scores(prediction, truth, valid, crop_valid, crop, image_hw, kind):
    """Primary metrics are source-pixel ADE; keep crop/full-frame scales explicit."""
    mask = valid[1:]
    if not mask.any():
        return dict(scoreable=False, reason="No valid future targets", per_hypothesis=[])
    rows = []
    normalized_truth = crop.normalize(truth)
    for hypothesis in prediction:
        row = trajectory_diagnostics(hypothesis, truth, valid, crop_valid, image_hw)
        errors = np.linalg.norm(hypothesis-truth, axis=-1)
        row.update(source_RMSE_pixels=float(np.sqrt(np.mean(errors[1:][mask]**2))),
            source_FDE_pixels=float(errors[-1][valid[-1]].mean()) if valid[-1].any() else None,
            crop_ADE_pixels=float(errors[1:][mask].mean()*crop.image_size/crop.side),
            paper_MSE_crop=float(4*np.square(crop.normalize(hypothesis)-normalized_truth)[1:][mask].sum(-1).mean()),
            source_ADE_per_state=[float(errors[t][valid[t]].mean()) if valid[t].any() else None for t in range(len(valid))])
        if kind=="fixed": row["GT_path_normalized_error_percent"] = None
        rows.append(row)
    # Hypothesis zero is always primary; oracle choices are diagnostics only.
    return clean(dict(scoreable=True, primary=rows[0], per_hypothesis=rows,
        oracle_ADE_source_pixels=min(row["pixel_ADE"] for row in rows),
        oracle_paper_MSE_crop=min(row["paper_MSE_crop"] for row in rows)))


def worker(args, manifest):
    device = torch.device(args.devices[args.worker_index]); torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    cfg = manifest["train_config"]
    cache = FeatureCache(manifest["feature_cache"], cfg["cache_namespace"], require_cached=True)
    tasks = [(split, index) for split in ("test1", "val")
        for index in range(len(manifest["splits"][split]["pairs"]))]
    tasks = tasks[args.worker_index::len(args.devices)]
    datasets = {split:make_dataset(manifest, split) for split in ("test1", "val")}
    model = None; completed = 0; started = time.perf_counter()
    current_scene = None; source = None
    def status(stage):
        write_json(args.output/f"status_worker_{args.worker_index}.json", dict(stage=stage,
            worker=args.worker_index, expected=len(tasks), completed=completed,
            seconds=time.perf_counter()-started, cache_hits=cache.hits, cache_misses=cache.misses))
    status("loading")
    for split, index in tasks:
        output = args.output/split
        record = output/"records"/f"pair_{index:05d}.json"
        prediction_file = output/"predictions"/f"pair_{index:05d}.npz"
        if record.exists():
            existing = json.loads(record.read_text())
            if existing["status"]=="predicted" and not prediction_file.exists():
                raise ValueError(f"Missing saved predictions for {record}")
            completed += 1; continue
        dataset = datasets[split]
        scene, pair = dataset.examples[index]
        descriptor = manifest["splits"][split]["scenes"][scene]
        if current_scene != (split, scene):
            with np.load(args.output/split/descriptor["archive"], allow_pickle=False) as arrays:
                source = dict(depth=arrays["depth"][0].copy(), depth_valid=arrays["depth_valid_mask"][0].astype(bool),
                    joint_indices=arrays["joint_indices"], joint_types=arrays["joint_types"])
            current_scene = split, scene
        sample = dataset[index]
        selected = sample["point_valid"] & sample["source_position_valid"][0]
        joint_position = int(np.flatnonzero(source["joint_indices"]==pair[2])[0])
        kind = {0:"revolute", 1:"prismatic", 4:"fixed"}.get(int(source["joint_types"][joint_position]), "other")
        row = dict(split=split, pair_index=index, original_pair_index=manifest["splits"][split]["original_pair_indices"][index],
            scene_index=scene, original_scene_index=descriptor["original_scene_index"],
            object_id=descriptor["object_id"], sample_id=sample["sample_id"], parent_link=pair[0], child_link=pair[1],
            joint_index=pair[2], joint_ordinal=pair[2]+1, gt_type=kind,
            source_archive=descriptor["archive"], source_original_path=descriptor["path"], source_sha256=descriptor["sha256"],
            prediction_file=str(prediction_file.relative_to(output)), seed=manifest["seed"]+index,
            initial_states_supplied=1, ensemble_size=manifest["ensemble_size"], diffusion_steps=manifest["diffusion_steps"])
        if not bool(selected.any()):
            row.update(status="unavailable", reason="No initially conditionable sampled points")
            write_json(record, clean(row)); completed += 1; status("inference"); continue
        if model is None:
            model = CachedFP32Model(build_cached_myriad(image_size=cfg["image_size"])).place(device)
            model.configure_cropped_cache(cache)
            payload = _read_checkpoint(args.output/"checkpoint.pt")
            state = {name:value for name,value in payload["model"].items() if not name.startswith("model.image_embedder.")}
            _validate_shapes(state, model.state_dict(), "Cropped inference model")
            model.load_state_dict(state, strict=True); model.eval()
            del payload, state
            print(json.dumps(dict(event="model_ready", worker=args.worker_index, step=manifest["checkpoint_step"], device=str(device))), flush=True)
        began = time.perf_counter()
        with sampler_attention_context():
            predictions = sample_rollout(model, sample["x"], sample["trajectories"][:1, selected], sample["state_times"],
                ensemble_size=manifest["ensemble_size"], diffusion_steps=manifest["diffusion_steps"],
                seed=row["seed"], amp_dtype="none").float().cpu().numpy()
        if not np.isfinite(predictions).all(): raise FloatingPointError(f"Nonfinite prediction {split}/{index}")
        crop = SquareCrop(*sample["crop_bounds"].tolist())
        uv = crop.unnormalize(predictions)
        truth = sample["source_trajectories"].numpy()[:, selected]
        valid = sample["source_position_valid"].numpy()[:, selected]
        crop_valid = sample["crop_position_valid"].numpy()[:, selected]
        pixels = sample["reference_pixels"].numpy()[selected].astype(float)
        k = sample["camera_intrinsic_source"].numpy(); e = sample["camera_extrinsic"].numpy()
        z = source["depth"][np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)]
        xyz = _backproject(pixels, z, k, e)
        scores = trajectory_scores(uv, truth, valid, crop_valid, crop, sample["source_image_hw"].tolist(), kind)
        image = sample["x"].numpy()
        arrays = dict(schema_version=np.array(manifest["schema_version"]),
            predictions_crop_normalized=predictions, predictions_source_pixels=uv,
            predictions_model_pixels=predictions*(cfg["image_size"]-1),
            ground_truth_source_pixels=truth, ground_truth_crop_normalized=sample["trajectories"].numpy()[:, selected],
            ground_truth_actual_source_pixels=sample["source_actual_trajectories"].numpy()[:, selected],
            ground_truth_actual_crop_normalized=sample["actual_trajectories"].numpy()[:, selected],
            valid=valid, crop_valid=crop_valid, actual_valid=sample["actual_position_valid"].numpy()[:, selected],
            actual_crop_valid=sample["actual_crop_position_valid"].numpy()[:, selected],
            reference_world=xyz, reference_source_pixels=pixels,
            reference_model_pixels=sample["reference_pixels_model"].numpy()[selected],
            local_points=sample["local_points"].numpy()[selected],
            relative_camera_points=sample["relative_camera_points"].numpy()[:, selected],
            point_indices=np.flatnonzero(selected.numpy()), point_link_nodes=sample["point_link_nodes"].numpy()[selected],
            point_link_labels=sample["point_link_labels"].numpy()[selected],
            input_image=image, cropped_rgb=np.rint((sample["original_x"].numpy()+1)*127.5).clip(0,255).astype(np.uint8),
            parent_mask=sample["parent_mask"].numpy(), child_mask=sample["child_mask"].numpy(),
            crop_bounds=sample["crop_bounds"].numpy(), crop_transform=sample["crop_transform"].numpy(),
            crop_inverse=sample["crop_inverse"].numpy(), camera_intrinsic=k,
            camera_intrinsic_model=sample["camera_intrinsic_model"].numpy(), camera_extrinsic=e,
            state_times=sample["state_times"].numpy(), state_fractions=sample["state_fractions"].numpy(),
            pair_labels=sample["pair_labels"].numpy(), joint_index=np.array(pair[2]),
            source_image_hw=sample["source_image_hw"].numpy(), source_archive=np.array(descriptor["archive"]),
            original_scene_path=np.array(descriptor["path"]), known_states=np.array(1), seed=np.array(row["seed"]),
            target_definition=np.array(cfg["target_definition"]))
        atomic_npz(prediction_file, **arrays)
        row.update(status="predicted", scoreable=scores["scoreable"], trajectory_metrics=scores,
            tracks=int(selected.sum()), crop_bounds=sample["crop_bounds"].tolist(),
            feature_cache_path=str(cache.path(sample["x"].unsqueeze(0))),
            prediction_seconds=time.perf_counter()-began)
        write_json(record, clean(row))
        completed += 1; status("inference")
        if completed%20==0:
            print(json.dumps(dict(event="progress", worker=args.worker_index, completed=completed, expected=len(tasks), split=split,
                pair=index, seconds=row["prediction_seconds"], ADE_source_pixels=scores.get("primary", {}).get("pixel_ADE"))), flush=True)
    status("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.worker_index is not None:
        return worker(args, json.loads((args.output/"run.json").read_text()))
    manifest = prepare(args.output, args.checkpoint)
    from finetuning_pair.evaluate import file_hash
    code = {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(args.output/"inference_code.json", code)
    for p in Path(__file__).parent.glob("*.py"):
        snapshot = args.output/"code"/p.name; snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(p.read_text())
    processes, handles = [], []
    try:
        for index in range(len(args.devices)):
            handle = (args.output/f"worker_{index}.log").open("a"); handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-B", "-u", "-m",
                "training_from_scratch_cropped.full_inference.run", *sys.argv[1:], "--worker-index", str(index)],
                stdout=handle, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            time.sleep(20)
            print(json.dumps(dict(event="inference_status", workers=[json.loads(p.read_text())
                for p in args.output.glob("status_worker_*.json")])), flush=True)
        if any(process.returncode!=0 for process in processes): raise RuntimeError("Inference worker failed; completed records are resumable")
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for handle in handles: handle.close()
    from .report import generate_report
    generate_report(args.output)
    from .audit import audit
    audit(args.output)


if __name__=="__main__": main()
