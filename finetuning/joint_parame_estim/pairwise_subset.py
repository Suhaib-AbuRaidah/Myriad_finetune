"""Test parent-child-only MYRIAD rollouts against full-scene rollouts on test1.

Both approaches use the same selected points and the same independent-pair
joint fitter. The full-scene predictions are read from the existing evaluation;
the parent-child predictions are generated and cached once per joint and mode.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from finetuning.dataset import ArticulatedDataset
from finetuning.joint_parame_estim.run import (
    STATES, axis_distance, axis_error, child_poses_from_joint, estimate_pose, fit_model, from_normalized,
    initial_axis, modeled_points, project, scene_data, select_type, unit,
)
from finetuning.testing import sample_rollout


ROOT = Path(__file__).resolve().parents[2]
MODES = ("without_poke", "with_poke")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "finetuning/runs/articulated/best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "finetuning/runs/pairwise_joint_test1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "auto", "none"), default="bf16")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-scenes", type=int, default=4)
    parser.add_argument("--other-scenes", type=int, default=4)
    parser.add_argument("--scene-indices", help="Optional comma-separated test1 indices instead of stratified selection")
    parser.add_argument("--baseline-only", action="store_true", help="Fit saved full-scene tracks without loading MYRIAD")
    args = parser.parse_args()
    if args.ensemble_size < 1 or args.diffusion_steps < 1 or args.seed < 0:
        parser.error("ensemble-size and diffusion-steps must be positive; seed must be nonnegative")
    if args.fixed_scenes < 0 or args.other_scenes < 0:
        parser.error("scene counts must be nonnegative")
    return args


def choose_indices(ds, args):
    if args.scene_indices:
        chosen = [int(item) for item in args.scene_indices.split(",")]
        if not chosen or len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= len(ds):
            raise ValueError("--scene-indices must be unique indices within test1")
        return sorted(chosen)
    fixed, other = [], []
    for index, path in enumerate(ds.files):
        with np.load(path, allow_pickle=False) as archive:
            has_fixed = bool(np.any(archive["joint_types"] == 4))
        (fixed if has_fixed else other).append(index)
    if args.fixed_scenes > len(fixed) or args.other_scenes > len(other):
        raise ValueError("Requested more scenes than available in a stratum")
    rng = np.random.default_rng(args.seed)
    return sorted([int(i) for i in rng.choice(fixed, args.fixed_scenes, replace=False)]
                  + [int(i) for i in rng.choice(other, args.other_scenes, replace=False)])


def hierarchy(source):
    """Joint array positions in base-to-leaf order."""
    pending = set(range(len(source["joint_indices"])))
    reached = {-1}
    while pending:
        ready = [j for j in sorted(pending) if int(source["joint_parent_link_indices"][j]) in reached]
        if not ready:
            raise ValueError(f"Disconnected joint hierarchy: {source['path']}")
        j = ready[0]
        pending.remove(j)
        reached.add(int(source["joint_child_link_indices"][j]))
        yield j


def load_model(checkpoint, device):
    from finetuning.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
    from finetuning.model_adapter import FineTuningModel, LoRAConfig
    from finetuning.train import build_myriad

    info = inspect_checkpoint(checkpoint)
    config = info["train_config"]
    if (config["points_per_link"], config["image_size"], config["num_transitions"], config["time_scale"]) != (8, 384, 4, 4.0):
        raise ValueError("Checkpoint dataset settings differ from the saved full-scene evaluation")
    bare = build_myriad()
    reference = load_pretrained_weights(bare, Path(info["base_checkpoint"]["path"]))
    model = FineTuningModel(bare, LoRAConfig(**info["lora_config"])).to(device)
    load_finetuned_weights(checkpoint, model, base_checkpoint=reference)
    model.eval()
    return model


def pair_record(source, sample, joint_pos, mode, approach, indices, predictions,
                validity, scene_index, parent_poses):
    """Fit one joint using the already fitted parent chain pose."""
    labels = source["link_indices"]
    parent = int(source["joint_parent_link_indices"][joint_pos])
    child = int(source["joint_child_link_indices"][joint_pos])
    by_label = {int(label): node for node, label in enumerate(labels)}
    if parent not in by_label or child not in by_label:
        return None, parent_poses
    nodes = source["point_link_nodes"][indices]
    parent_mask, child_mask = nodes == by_label[parent], nodes == by_label[child]
    if not parent_mask.any() or not child_mask.any():
        return None, parent_poses
    world = source["world"][indices]
    k, e = source["camera_intrinsic"], source["camera_extrinsic"]
    h, w = source["rgb"].shape[1:3]
    image = from_normalized(predictions, h, w)[STATES]
    truth = from_normalized(sample["trajectories"][:, indices].numpy(), h, w)[STATES]
    valid = validity[STATES] & source["point_valid"][indices][None]
    if valid[:, child_mask].sum() < 8:
        return None, parent_poses
    xyz, uv, mask = world[child_mask], image[:, child_mask], valid[:, child_mask]
    child_init = [estimate_pose(xyz, uv[s], mask[s], k, e) for s in range(len(STATES))]
    axis0, pivot0, theta0 = initial_axis(child_init, parent_poses, xyz.mean(0))
    fixed = fit_model("fixed", xyz, uv, mask, parent_poses, k, e, np.zeros(0), max_iter=0)
    if fixed is None:
        return None, parent_poses
    rev = fit_model("revolute", xyz, uv, mask, parent_poses, k, e, np.r_[axis0, pivot0, theta0])
    translations = [rp.T @ (tc - tp) if gc and gp else np.zeros(3)
                    for (_, tc, gc), (rp, tp, gp) in zip(child_init, parent_poses)]
    travel = np.asarray(translations)
    direction = np.linalg.svd(travel, full_matrices=False)[2][0]
    prism = fit_model("prismatic", xyz, uv, mask, parent_poses, k, e,
                      np.r_[direction, travel @ direction])
    fits = {"fixed": fixed, "prismatic": prism, "revolute": rev}
    fixed_xyz = modeled_points("fixed", np.zeros(0), xyz, parent_poses)
    fixed_xy = np.asarray([project(points, k, e) for points in fixed_xyz])
    motion = float(np.median(np.linalg.norm((uv - fixed_xy)[mask], axis=1)))
    predicted_type, scores = select_type(fits, motion)
    chosen = fits[predicted_type]
    gt_type = {0: "revolute", 1: "prismatic", 4: "fixed"}.get(int(source["joint_types"][joint_pos]), "other")
    axis = unit(chosen["params"][:3]) if predicted_type != "fixed" else np.full(3, np.nan)
    pivot = chosen["params"][3:6] if predicted_type == "revolute" else np.full(3, np.nan)
    gt_axis = source["joint_axes_world"][0, joint_pos].astype(float)
    gt_origin = source["joint_origins_world"][0, joint_pos].astype(float)
    def ade(part):
        ok = valid[:, part]
        errors = np.linalg.norm(image[:, part] - truth[:, part], axis=-1)
        return float(errors[ok].mean()) if ok.any() else float("nan")
    record = dict(scene_index=scene_index, sample_id=source["path"].stem,
                joint_index=int(source["joint_indices"][joint_pos]), mode=mode,
                approach=approach, parent_link=parent, child_link=child,
                gt_type=gt_type, predicted_type=predicted_type,
                n_child_points=int(child_mask.sum()), n_child_observations=int(mask.sum()),
                parent_ade_px=ade(parent_mask), child_ade_px=ade(child_mask),
                motion_px=motion, selected_rmse_px=chosen["rmse_px"],
                fixed_rmse_px=fixed["rmse_px"],
                prismatic_rmse_px=prism["rmse_px"] if prism else float("nan"),
                revolute_rmse_px=rev["rmse_px"] if rev else float("nan"),
                axis_error_deg=(axis_error(axis, gt_axis) if predicted_type == gt_type == "revolute"
                                else float("nan")),
                pivot_distance=(axis_distance(pivot, axis, gt_origin)
                                if predicted_type == gt_type == "revolute" else float("nan")),
                score_fixed=scores.get("fixed", float("nan")),
                score_prismatic=scores.get("prismatic", float("nan")),
                score_revolute=scores.get("revolute", float("nan")))
    return record, child_poses_from_joint(predicted_type, chosen["params"], parent_poses)


def stats(rows, total):
    def distribution(values):
        numbers = np.asarray(list(values), float)
        numbers = numbers[np.isfinite(numbers)]
        return {"n": len(numbers), "mean": float(numbers.mean()) if len(numbers) else None,
                "median": float(np.median(numbers)) if len(numbers) else None}
    gt_fixed = [r for r in rows if r["gt_type"] == "fixed"]
    gt_rev = [r for r in rows if r["gt_type"] == "revolute"]
    return {"joints_evaluated": len(rows), "joints_total": total,
            "joint_type_accuracy": (sum(r["gt_type"] == r["predicted_type"] for r in rows) / len(rows)
                                    if rows else None),
            "fixed_recall": (sum(r["predicted_type"] == "fixed" for r in gt_fixed) / len(gt_fixed)
                             if gt_fixed else None),
            "revolute_recall": (sum(r["predicted_type"] == "revolute" for r in gt_rev) / len(gt_rev)
                                if gt_rev else None),
            "confusion": {kind: {pred: sum(r["gt_type"] == kind and r["predicted_type"] == pred
                                      for r in rows) for pred in ("fixed", "prismatic", "revolute")}
                          for kind in ("fixed", "prismatic", "revolute")},
            "child_ade_px": distribution(r["child_ade_px"] for r in rows),
            "parent_ade_px": distribution(r["parent_ade_px"] for r in rows),
            "axis_error_deg": distribution(r["axis_error_deg"] for r in rows),
            "pivot_distance": distribution(r["pivot_distance"] for r in rows),
            "selected_rmse_px": distribution(r["selected_rmse_px"] for r in rows)}


def main():
    args = parse_args()
    ds = ArticulatedDataset(ROOT / "data/Robotic_arm", split="test1", training=False,
                            points_per_link=8, image_size=384, seed=args.seed)
    indices = choose_indices(ds, args)
    if not indices:
        raise ValueError("No scenes selected")
    if not args.baseline_only and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable here. Run --baseline-only or rerun on a CUDA machine.")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    settings = {"schema_version": 1, "scene_indices": indices, "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                "ensemble_size": args.ensemble_size, "diffusion_steps": args.diffusion_steps,
                "seed": args.seed, "amp_dtype": args.amp_dtype,
                "point_selection": "saved_full_scene_conditionable_points",
                "forecast_states": STATES.tolist(), "hypothesis_index": 0,
                "joint_fitter": "stationary_base_forward_chain_unpenalized_robust_SSE"}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError(f"Existing output has different settings: {manifest}")
    manifest.write_text(json.dumps(settings, indent=2) + "\n")
    model = load_model(checkpoint, torch.device(args.device)) if not args.baseline_only else None
    rows = []
    total = 0
    for index in indices:
        sample = ds[index]
        source = scene_data(ds, index)
        total += len(source["joint_indices"])
        by_label = {int(label): node for node, label in enumerate(source["link_indices"])}
        for mode in MODES:
            identity = [(np.eye(3), np.zeros(3), True) for _ in STATES]
            full_poses = {-1: identity}
            pair_poses = {-1: identity}
            existing = (ROOT / "finetuning/runs/test1_presentation" / mode / "results"
                        / f"scene_{index:04d}_{source['path'].stem}" / "results.npz")
            with np.load(existing, allow_pickle=False) as archive:
                selected = archive["point_indices"].astype(int)
                nodes = archive["link_nodes"].astype(int)
                full_predictions = archive["predictions"][0].astype(float)
                validity = archive["position_valid"].astype(bool)
                known_states = int(archive["known_states"])
                saved_known = archive["known_positions"]
            if known_states != (2 if mode == "with_poke" else 1):
                raise ValueError(f"Unexpected saved known state count: {existing}")
            if not np.array_equal(nodes, source["point_link_nodes"][selected]):
                raise ValueError(f"Point order mismatch: {existing}")
            for j in hierarchy(source):
                parent = int(source["joint_parent_link_indices"][j])
                child = int(source["joint_child_link_indices"][j])
                if parent not in by_label or child not in by_label:
                    continue
                parent_full_poses = full_poses[parent]
                parent_pair_poses = pair_poses[parent]
                pair_mask = np.isin(nodes, [by_label[parent], by_label[child]])
                pair_indices = selected[pair_mask]
                if not np.array_equal(sample["trajectories"][:known_states, pair_indices].numpy(),
                                      saved_known[:, pair_mask]):
                    raise ValueError(f"Known point positions differ from saved run: {existing}")
                pair_validity = validity[:, pair_mask]
                baseline, full_poses[child] = pair_record(
                    source, sample, j, mode, "full_scene", pair_indices,
                    full_predictions[:, pair_mask], pair_validity, index, parent_full_poses)
                if baseline is not None:
                    rows.append(baseline)
                if model is None:
                    pair_poses[child] = parent_pair_poses
                    continue
                child_mask = nodes[pair_mask] == by_label[child]
                parent_mask = nodes[pair_mask] == by_label[parent]
                if pair_validity[STATES][:, child_mask].sum() < 8 or not parent_mask.any():
                    pair_poses[child] = parent_pair_poses
                    continue
                file = out / "predictions" / mode / f"scene_{index:04d}_joint_{int(source['joint_indices'][j])}.npz"
                if file.exists():
                    with np.load(file, allow_pickle=False) as archive:
                        if not np.array_equal(archive["point_indices"], pair_indices):
                            raise ValueError(f"Cached point indices differ: {file}")
                        paired_predictions = archive["predictions"][0].astype(float)
                else:
                    known = sample["trajectories"][:known_states, pair_indices]
                    predicted = sample_rollout(model, sample["x"], known, sample["state_times"],
                                               ensemble_size=args.ensemble_size,
                                               diffusion_steps=args.diffusion_steps,
                                               seed=args.seed + index, amp_dtype=args.amp_dtype)
                    file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(file, point_indices=pair_indices,
                                        predictions=predicted.float().cpu().numpy())
                    paired_predictions = predicted[0].float().cpu().numpy()
                paired, pair_poses[child] = pair_record(
                    source, sample, j, mode, "parent_child", pair_indices,
                    paired_predictions, pair_validity, index, parent_pair_poses)
                if paired is not None:
                    rows.append(paired)
                print(f"scene {index:04d} {mode} joint {int(source['joint_indices'][j])}: "
                      f"full={baseline['predicted_type'] if baseline else 'unfit'} "
                      f"pair={paired['predicted_type'] if paired else 'unfit'}",
                      flush=True)
        print(f"Completed scene {index:04d}", flush=True)
    if not rows:
        raise RuntimeError("No joints could be evaluated in the selected scenes")
    with (out / "per_joint.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"settings": settings, "baseline_only": args.baseline_only,
              "evaluations": {mode: {method: stats([r for r in rows if r["mode"] == mode
                                                     and r["approach"] == method], total)
                                    for method in (("full_scene",) if model is None
                                                   else ("full_scene", "parent_child"))}
                              for mode in MODES}}
    # The common cohort is the fair comparison when a paired inference/fitting fails.
    if model is not None:
        result["matched_evaluations"] = {}
        for mode in MODES:
            mode_rows = [r for r in rows if r["mode"] == mode]
            pair_keys = {(r["scene_index"], r["joint_index"]) for r in mode_rows
                         if r["approach"] == "parent_child"}
            full_keys = {(r["scene_index"], r["joint_index"]) for r in mode_rows
                         if r["approach"] == "full_scene"}
            keys = pair_keys & full_keys
            result["matched_evaluations"][mode] = {
                method: stats([r for r in mode_rows if r["approach"] == method
                               and (r["scene_index"], r["joint_index"]) in keys], len(keys))
                for method in ("full_scene", "parent_child")}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
