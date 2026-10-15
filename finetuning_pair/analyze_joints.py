"""Reproducible first-scene evaluation of child-relative checkpoints.

Run from the repository root with ``python -m finetuning_pair.analyze_joints``.
All joint fits use predicted states 2--4 and initial RGB-D, with an identity
parent pose because these checkpoints predict frozen-parent relative motion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from contextlib import nullcontext
from unittest.mock import patch

import cv2
import numpy as np
import torch

from finetuning_pair.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
from finetuning_pair.dataset import ArticulatedDataset, _backproject
from finetuning_pair.evaluate import file_hash, scene_metrics, write_json
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.model_adapter import FineTuningModel, LoRAConfig
from finetuning_pair.paths import artifact_path
from finetuning_pair.testing import _scene_inputs, sample_rollout
from finetuning_pair.train import build_myriad
from finetuning.joint_parame_estim.run import (
    axis_distance, axis_error, estimate_pose, fit_model, from_normalized,
    initial_axis, select_type, unit,
)

STATES = np.array([2, 3, 4])
KINDS = ("fixed", "prismatic", "revolute")
METRICS = ("minADE_1", "minADE_5", "ADE_1_pixels", "FDE_1_pixels", "RMSE_1_pixels",
           "common_ADE_pixels", "stationary_ADE_pixels", "out_of_frame_fraction_1",
           "axis_error_deg", "pivot_distance", "forced_revolute_axis_error_deg",
           "forced_revolute_pivot_distance", "rotation_geodesic_deg", "rotation_endpoint_deg",
           "angle_magnitude_mae_deg", "selected_fit_rmse_pixels", "motion_pixels",
           "ADE_state_1_pixels", "ADE_state_2_pixels", "ADE_state_3_pixels", "ADE_state_4_pixels",
           "PCK_2_pixels", "PCK_5_pixels", "PCK_10_pixels", "PCK_20_pixels", "oracle_5_ADE_pixels",
           "pred_rotation_magnitude_deg", "gt_rotation_mean_deg", "rotation_magnitude_bias_deg",
           "kinematic_pivot_distance", "forced_revolute_kinematic_pivot_distance", "stored_pivot_axis_offset")


def dense_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, **kwargs):
    """Exact token mask for CPU SDPA, avoiding the CUDA FlexAttention compiler."""
    b = torch.arange(B or 1, device=device)[:, None, None, None]
    h = torch.arange(H or 1, device=device)[None, :, None, None]
    q = torch.arange(Q_LEN, device=device)[None, None, :, None]
    k = torch.arange(KV_LEN, device=device)[None, None, None, :]
    return mask_mod(b, h, q, k).expand(B or 1, H or 1, Q_LEN, KV_LEN)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fit_joint(xyz, uv, valid, k, e):
    """No true joint type/axis/angle enters estimation or selection."""
    identity = [(np.eye(3), np.zeros(3), True) for _ in range(len(uv))]
    poses = [estimate_pose(xyz, uv[s], valid[s], k, e) for s in range(len(uv))]
    axis, pivot, theta = initial_axis(poses, identity, xyz.mean(0))
    fixed = fit_model("fixed", xyz, uv, valid, identity, k, e, np.zeros(0), max_iter=0)
    if fixed is None:
        return None
    rev = fit_model("revolute", xyz, uv, valid, identity, k, e, np.r_[axis, pivot, theta])
    travel = np.asarray([t if good else np.zeros(3) for _, t, good in poses])
    direction = np.linalg.svd(travel, full_matrices=False)[2][0]
    prism = fit_model("prismatic", xyz, uv, valid, identity, k, e, np.r_[direction, travel @ direction])
    initial_xy = (xyz @ e[:3, :3].T + e[:3, 3]) @ k.T
    initial_xy = initial_xy[:, :2] / initial_xy[:, 2:]
    motion = float(np.median(np.linalg.norm(uv - initial_xy[None], axis=-1)[valid]))
    fits = dict(fixed=fixed, prismatic=prism, revolute=rev)
    kind, scores = select_type(fits, motion)
    return dict(kind=kind, fits=fits, scores=scores, motion=motion,
                pnp_successes=sum(p[2] for p in poses))


def geometry_metrics(fitted, source, joint_pos, parent, child):
    gt_type = {0: "revolute", 1: "prismatic", 4: "fixed"}.get(int(source["joint_types"][joint_pos]), "other")
    row = dict(gt_type=gt_type, fit_available=fitted is not None)
    if fitted is None:
        return row
    kind = fitted["kind"]
    selected = fitted["fits"][kind]
    row.update(predicted_type=kind, type_correct=kind == gt_type,
               motion_pixels=fitted["motion"], pnp_successes=fitted["pnp_successes"],
               selected_fit_rmse_pixels=selected["rmse_px"])
    for name, fit in fitted["fits"].items():
        row[f"score_{name}"] = fitted["scores"].get(name)
        row[f"fit_rmse_{name}"] = fit["rmse_px"] if fit else None
        row[f"fit_parameters_{name}"] = fit["params"] if fit else None
    row["estimated_axis_world"] = unit(selected["params"][:3]) if kind != "fixed" else None
    row["estimated_pivot_world"] = selected["params"][3:6] if kind == "revolute" else None
    axis_gt = unit(source["joint_axes_world"][0, joint_pos].astype(float))
    pivot_gt = source["joint_origins_world"][0, joint_pos].astype(float)
    row.update(gt_axis_world=axis_gt, gt_pivot_world=pivot_gt)
    rev = fitted["fits"]["revolute"]
    if gt_type == "revolute" and rev is not None:
        axis = unit(rev["params"][:3])
        pivot = rev["params"][3:6]
        # An axis line has no identifiable pivot coordinate along its direction.
        aligned_pivot = pivot + axis * ((pivot_gt - pivot) @ axis)
        row.update(forced_revolute_axis_error_deg=axis_error(axis, axis_gt),
                   forced_revolute_pivot_distance=axis_distance(pivot, axis, pivot_gt),
                   forced_revolute_axis_world=axis, forced_revolute_pivot_world=pivot,
                   gauge_aligned_pivot_world=aligned_pivot)
        if kind == "revolute":
            row.update(axis_error_deg=row["forced_revolute_axis_error_deg"],
                       pivot_distance=row["forced_revolute_pivot_distance"])
    # Compare actual relative rotations, including type errors (fixed/prismatic
    # estimates have identity rotation). This also measures rotation angle error.
    transforms = source["link_transforms_world"]
    pred_rot = [cv2.Rodrigues(unit(selected["params"][:3]) * a)[0]
                for a in selected["params"][6:]] if kind == "revolute" else [np.eye(3)] * len(STATES)
    geodesics, magnitudes, gt_angles = [], [], []
    for slot, state in enumerate(STATES):
        relative = transforms[0, parent] @ np.linalg.inv(transforms[state, parent]) @ transforms[state, child] @ np.linalg.inv(transforms[0, child])
        gt_rotation = relative[:3, :3]
        err = math.degrees(np.linalg.norm(cv2.Rodrigues(pred_rot[slot] @ gt_rotation.T)[0]))
        gt_angle = math.degrees(np.linalg.norm(cv2.Rodrigues(gt_rotation)[0]))
        pred_angle = math.degrees(np.linalg.norm(cv2.Rodrigues(pred_rot[slot])[0]))
        geodesics.append(err)
        magnitudes.append(abs(gt_angle - pred_angle))
        gt_angles.append(gt_angle)
    row.update(rotation_geodesic_deg=float(np.mean(geodesics)), rotation_endpoint_deg=geodesics[-1],
               angle_magnitude_mae_deg=float(np.mean(magnitudes)),
               rotation_geodesic_per_state_deg=geodesics, gt_rotation_magnitude_deg=gt_angles)
    return clean(row)


def distribution(values):
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(x):
        return dict(n=0, mean=None, median=None, p90=None, p95=None, min=None, max=None, std=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()), std=float(x.std()))


def summarize(rows):
    scored = [r for r in rows if not r.get("skipped")]
    fitted = [r for r in scored if r.get("fit_available")]
    confusion = {g: {p: sum(r["gt_type"] == g and r["predicted_type"] == p for r in fitted)
                     for p in KINDS} for g in KINDS}
    recall = {g: (confusion[g][g] / sum(confusion[g].values()) if sum(confusion[g].values()) else None) for g in KINDS}
    correct = sum(r["type_correct"] for r in fitted)
    geom = [r for r in fitted if r.get("axis_error_deg") is not None]
    return dict(pairs_total=len(rows), pairs_scored=len(scored), joints_fitted=len(fitted),
                type_correct=correct, type_accuracy=correct/len(fitted) if fitted else None,
                correct_type_all_pairs=correct/len(rows) if rows else None,
                beats_stationary_fraction=sum(r["improves_stationary"] for r in scored)/len(scored) if scored else None,
                confusion=confusion, recall=recall,
                balanced_accuracy=float(np.mean([v for v in recall.values() if v is not None])) if fitted else None,
                metrics={m: distribution(r.get(m) for r in scored) for m in METRICS},
                joint_success={f"axis_le_{angle}_pivot_le_{dist}":
                               sum(r["axis_error_deg"] <= angle and r["pivot_distance"] <= dist for r in geom) /
                               max(1, sum(r["gt_type"] == "revolute" for r in rows))
                               for angle, dist in ((5, .01), (10, .05), (15, .1), (30, .1))})


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def report(out, manifest, rows):
    modes = manifest["modes"]
    reference = {}
    # Derive extra trajectory diagnostics solely from persisted predictions.
    for row in rows:
        if row.get("skipped"):
            continue
        path = out / "predictions" / f"{row['pair_index']:04d}_{row['mode']}.npz"
        with np.load(path, allow_pickle=False) as a:
            start = int(a["known_states"])
            distances = np.linalg.norm((a["predictions"][0,start:]-a["ground_truth"][start:]) * (a["image"].shape[-1]-1), axis=-1)
            scored = distances[a["valid"][start:]]
            for threshold in (2,5,10,20):
                row[f"PCK_{threshold}_pixels"] = float(np.mean(scored <= threshold))
            if row["gt_type"] == "revolute" and str(row["pair_index"]) not in reference:
                transforms = a["link_transforms_world"].astype(float)
                lookup = {int(label):i for i,label in enumerate(a["link_indices"])}
                pa, ch = [lookup[int(label)] for label in a["pair_labels"]]
                matrices, translations = [], []
                for state in range(1,5):
                    motion = transforms[0,pa] @ np.linalg.inv(transforms[state,pa]) @ transforms[state,ch] @ np.linalg.inv(transforms[0,ch])
                    matrices.append(np.eye(3)-motion[:3,:3])
                    translations.append(motion[:3,3])
                matrix, rhs = np.concatenate(matrices), np.concatenate(translations)
                axis = np.linalg.svd(matrix, full_matrices=False)[2][-1]
                if axis @ np.asarray(row["gt_axis_world"]) < 0:
                    axis = -axis
                pivot = np.linalg.lstsq(np.vstack([matrix, axis]), np.r_[rhs, 0.], rcond=1e-7)[0]
                nominal = np.asarray(row["gt_pivot_world"])
                pivot += axis * ((nominal-pivot) @ axis)
                reference[str(row["pair_index"])] = clean(dict(scene_index=row["scene_index"], joint_index=row["joint_index"],
                    kinematic_axis_world=axis, kinematic_pivot_world=pivot,
                    stored_pivot_world=nominal, stored_pivot_axis_offset=axis_distance(pivot,axis,nominal),
                    stored_axis_orientation_error_deg=axis_error(axis,np.asarray(row["gt_axis_world"])),
                    transform_fit_residual=float(np.linalg.norm(matrix@pivot-rhs)),
                    stored_pivot_transform_residual=float(np.linalg.norm(matrix@nominal-rhs))))
        ref = reference.get(str(row["pair_index"]))
        if ref:
            row.update(kinematic_gt_axis_world=ref["kinematic_axis_world"], kinematic_gt_pivot_world=ref["kinematic_pivot_world"],
                       stored_pivot_axis_offset=ref["stored_pivot_axis_offset"])
            if row.get("axis_error_deg") is not None:
                row["kinematic_pivot_distance"] = axis_distance(np.asarray(row["estimated_pivot_world"]),
                    np.asarray(row["estimated_axis_world"]), np.asarray(ref["kinematic_pivot_world"]))
            if row.get("forced_revolute_axis_error_deg") is not None:
                row["forced_revolute_kinematic_pivot_distance"] = axis_distance(np.asarray(row["forced_revolute_pivot_world"]),
                    np.asarray(row["forced_revolute_axis_world"]), np.asarray(ref["kinematic_pivot_world"]))
            control = row.get("ground_truth_control")
            if control and control.get("axis_error_deg") is not None:
                control["kinematic_pivot_distance"] = axis_distance(np.asarray(control["estimated_pivot_world"]),
                    np.asarray(control["estimated_axis_world"]), np.asarray(ref["kinematic_pivot_world"]))
        if row.get("fit_available"):
            angles = np.array(row["fit_parameters_revolute"][6:]) if row["predicted_type"] == "revolute" else np.zeros(3)
            pred_magnitude = float(np.rad2deg(abs((angles+np.pi) % (2*np.pi)-np.pi)).mean())
            gt_magnitude = float(np.mean(row["gt_rotation_magnitude_deg"]))
            row.update(pred_rotation_magnitude_deg=pred_magnitude, gt_rotation_mean_deg=gt_magnitude,
                       rotation_magnitude_bias_deg=pred_magnitude-gt_magnitude)
    write_json(out / "geometry_reference.json", reference)
    consistency = dict(joints_checked=len(reference),
        stored_pivots_off_kinematic_axis_over_1e_4=sum(r["stored_pivot_axis_offset"] > 1e-4 for r in reference.values()),
        stored_pivot_axis_offset=distribution(r["stored_pivot_axis_offset"] for r in reference.values()),
        stored_axis_orientation_error_deg=distribution(r["stored_axis_orientation_error_deg"] for r in reference.values()),
        transform_fit_residual=distribution(r["transform_fit_residual"] for r in reference.values()))
    write_json(out / "dataset_geometry_consistency.json", consistency)
    summary = {}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = dict(overall=summarize(subset),
            per_joint={str(j): summarize([r for r in subset if r["joint_index"] == j])
                       for j in sorted({r["joint_index"] for r in subset})},
            per_chain_depth={str(d): summarize([r for r in subset if r["chain_depth"] == d])
                             for d in sorted({r["chain_depth"] for r in subset})},
            per_scene={str(s): summarize([r for r in subset if r["scene_index"] == s])
                       for s in sorted({r["scene_index"] for r in subset})})
        for joint, stats in summary[mode]["per_joint"].items():
            inventory = [r for r in manifest["inventory"] if r["joint_index"] == int(joint)]
            stats["source_joint_count"] = len(inventory)
            stats["eligible_joint_count"] = sum(r["eligible"] for r in inventory)
            stats["scored_source_coverage"] = stats["pairs_scored"] / len(inventory)
        summary[mode]["overall"]["source_joint_count"] = manifest["all_source_joints"]
        summary[mode]["overall"]["scored_source_coverage"] = summary[mode]["overall"]["pairs_scored"]/manifest["all_source_joints"]
    comparisons = {}
    if "without_poke" in modes and "with_poke" in modes:
        indexed = {mode: {r["pair_index"]:r for r in rows if r["mode"] == mode} for mode in modes}
        for metric in ("common_ADE_pixels", "axis_error_deg", "pivot_distance", "kinematic_pivot_distance", "forced_revolute_axis_error_deg", "rotation_geodesic_deg"):
            matched = [(r["scene_index"], r[metric], indexed["with_poke"][i][metric])
                       for i,r in indexed["without_poke"].items()
                       if i in indexed["with_poke"] and r.get(metric) is not None and indexed["with_poke"][i].get(metric) is not None]
            by_scene = {}
            for scene, a, b in matched:
                by_scene.setdefault(scene, []).append(b-a)
            if by_scene:
                sums = np.array([sum(v) for v in by_scene.values()])
                counts = np.array([len(v) for v in by_scene.values()])
                sampled = np.random.default_rng(42).integers(0,len(sums),(2000,len(sums)))
                boot = sums[sampled].sum(1)/counts[sampled].sum(1)
                comparisons[metric] = dict(n=len(matched), scenes=len(by_scene),
                    without_mean=float(np.mean([r[1] for r in matched])), with_mean=float(np.mean([r[2] for r in matched])),
                    with_minus_without=float(sums.sum()/counts.sum()),
                    scene_bootstrap_difference_ci95=np.percentile(boot,[2.5,97.5]).tolist())
    write_json(out / "matched_mode_comparison.json", comparisons)
    ranking = {}
    for mode in modes:
        ranking[mode] = {}
        for metric in ("ADE_1_pixels", "axis_error_deg", "pivot_distance"):
            eligible = [r for r in rows if r["mode"] == mode and r.get(metric) is not None]
            ordered = sorted(eligible, key=lambda r:r[metric])
            def case(r):
                return {k:r[k] for k in ("pair_index","scene_index","joint_index",metric)}
            ranking[mode][metric] = dict(best=[case(r) for r in ordered[:10]],
                                        worst=[case(r) for r in ordered[-10:][::-1]])
    write_json(out / "ranked_cases.json", ranking)
    write_json(out / "summary.json", clean(summary))
    csv_file(out / "per_joint.csv", rows)
    write_json(out / "per_joint.json", clean(rows))
    gt = [r["ground_truth_control"] for r in rows if r.get("ground_truth_control")]
    write_json(out / "ground_truth_control.json", clean(summarize(gt)))
    text = [f"# First {len(manifest['scenes'])} test1 scenes: child-relative joint evaluation", "",
            f"Checkpoint step: {manifest['checkpoint_step']}. Scene count: {len(manifest['scenes'])}. "
            f"Eligible pairs: {len(manifest['pairs'])}; all source joints: {manifest['all_source_joints']}. "
            f"Hypotheses: {manifest['ensemble_size']}; diffusion steps: {manifest['diffusion_steps']}. "
            f"Device: {manifest['device']}; precision: {manifest['amp_dtype']}.", "",
            "Scenes are the first filenames in ascending order, never a random subset. Joint 1 is dataset joint_index=0; "
            "chain depth is also reported separately for branched trees. Every pair has equal weight in overall means. "
            "Image-only (without_poke) is the primary inference setting. With-poke receives the GT relative state-1 positions.", "",
            "## Overall and per-joint results", "",
            "| Mode / joint | Scored / source | Type correct / fitted | ADE px | FDE px | Axis mean / median / p90 (deg) | Pivot mean / median / p90 | Axis cohort | Rotation geodesic deg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "N/A" if v is None else f"{v:.4g}"
    if "without_poke" in summary and "with_poke" in summary:
        bare, hint = summary["without_poke"]["overall"], summary["with_poke"]["overall"]
        metrics = bare["metrics"]
        at = text.index("## Overall and per-joint results")
        text[at:at] = ["## Main findings", "",
            f"Image-only inference is weak on this subset: mean ADE {fmt(metrics['ADE_1_pixels']['mean'])} px "
            f"exceeds the stationary baseline's {fmt(metrics['stationary_ADE_pixels']['mean'])} px. "
            f"Error grows from {fmt(metrics['ADE_state_1_pixels']['mean'])} px at state 1 to "
            f"{fmt(metrics['ADE_state_4_pixels']['mean'])} px at state 4. Estimated mean rotation magnitude "
            f"is {fmt(metrics['pred_rotation_magnitude_deg']['mean'])}° versus GT {fmt(metrics['gt_rotation_mean_deg']['mean'])}°, "
            "consistent with substantial underprediction of motion as well as wrong axes.", "",
            f"The GT first-motion hint substantially improves forecasting and geometry: on matched states 2–4, "
            f"ADE changes from {fmt(comparisons['common_ADE_pixels']['without_mean'])} to "
            f"{fmt(comparisons['common_ADE_pixels']['with_mean'])} px. "
            "This measures the benefit of an observed true motion hint and should not be treated as image-only performance.", "",
            f"Overall type accuracy is {100*bare['type_accuracy']:.1f}% / {100*hint['type_accuracy']:.1f}% "
            f"(without/with poke), but fixed recall is only {bare['confusion']['fixed']['fixed']}/{sum(bare['confusion']['fixed'].values())} / "
            f"{hint['confusion']['fixed']['fixed']}/{sum(hint['confusion']['fixed'].values())}. Balanced accuracy is "
            f"{100*bare['balanced_accuracy']:.1f}% / {100*hint['balanced_accuracy']:.1f}%. "
            "There are no true prismatic joints, so prismatic recall cannot be measured. "
            "Joint 4 has only eight eligible examples, and its mean geometry errors are sensitive to individual failures.", "",
            "Pivot errors have long tails; compare medians and p90 with means. The independently checked stored-pivot "
            "inconsistency below also affects the nominal pivot benchmark. All original and transform-consistent scores are retained.", ""]
    for mode in modes:
        groups = [("All", summary[mode]["overall"])] + [(f"Joint {int(j)+1} (index {j})", s) for j, s in summary[mode]["per_joint"].items()]
        for name, s in groups:
            m = s["metrics"]
            ax, pi = m["axis_error_deg"], m["pivot_distance"]
            text.append(f"| {mode} / {name} | {s['pairs_scored']}/{s['source_joint_count']} | {s['type_correct']}/{s['joints_fitted']} | "
                        f"{fmt(m['ADE_1_pixels']['mean'])} | {fmt(m['FDE_1_pixels']['mean'])} | "
                        f"{' / '.join(fmt(ax[k]) for k in ('mean','median','p90'))} | "
                        f"{' / '.join(fmt(pi[k]) for k in ('mean','median','p90'))} | {ax['n']} | {fmt(m['rotation_geodesic_deg']['mean'])} |")
    text.extend(["", "## Pivot reference consistency", "",
        f"**{consistency['stored_pivots_off_kinematic_axis_over_1e_4']}/{consistency['joints_checked']} GT revolute joints** "
        "have a stored pivot more than 0.0001 world units from the rotation axis independently implied by their link transforms. "
        "This discrepancy persists when fitting true trajectories, with near-zero reprojection error. "
        "The pivot values in the table above intentionally use the original stored joint origins. "
        "They therefore include reference inconsistency as well as prediction/estimation error.", "",
        "For a second reference, solve (I − R_t)q = t_t from the exact frozen-parent link transforms, "
        "then choose the axis point closest to the stored origin. These scoring references never enter trajectory prediction or joint fitting. "
        "The resulting transform-consistent pivot errors are:", "",
        "| Mode / joint | n | Kinematic pivot mean | Median | p90 |", "|---|---:|---:|---:|---:|"])
    for mode in modes:
        groups = [("All",summary[mode]["overall"])] + [(f"Joint {int(j)+1}",s) for j,s in summary[mode]["per_joint"].items()]
        for name,s in groups:
            d=s["metrics"]["kinematic_pivot_distance"]
            text.append(f"| {mode} / {name} | {d['n']} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['p90'])} |")
    text.extend(["", "## Definitions and interpretation", "",
        "- ADE/FDE/RMSE are Euclidean pixel errors at the checkpoint's model image resolution. Without-poke scores states 1–4; with-poke scores states 2–4. common_ADE_pixels scores states 2–4 for both. Hypothesis 0 supplies all primary and joint-geometry metrics.",
        "- minADE_1 is the repository's paper-scale **mean squared** Euclidean error: mean(4 × squared normalized distance). minADE_5 selects the best complete trajectory of five using GT, so it is an oracle diagnostic, not deployable selection.",
        "- The stationary baseline holds the latest supplied positions: state 0 without-poke and state 1 with-poke. PCK is the fraction of scored point-states within the given model-pixel threshold, averaged equally across pairs.",
        "- Joint fitting uses initial depth and calibrated camera, predicted states 2–4, known pair labels, and identity parent motion. GT future in-frame validity selects correspondences; future occlusion is unknown. No GT axis/type/pose enters the model fit or type selection.",
        "- Fixed, prismatic and revolute candidates reuse the existing robust reprojection estimator (Huber residuals, one trimming/refit pass, 3 source-pixel stationary guard, lowest unpenalized robust SSE). Geometry measures the combination of trajectory prediction and this estimator.",
        "- Axis error = acos(abs(dot(unit estimated axis, unit GT axis))) in degrees, in [0,90]. Opposite directions represent the same axis. Main axis/pivot metrics include only correctly classified revolute joints; read cohort counts and type recall alongside them.",
        "- Pivot distance = norm(cross(GT origin − estimated pivot, unit estimated axis)), in dataset world units. It measures the GT pivot's perpendicular distance to the predicted axis. Gauge-aligning the predicted pivot along its axis gives the same point distance; raw pivot-to-pivot Euclidean distance is unidentifiable along the axis and is not scored. No meter/cm conversion is assumed.",
        "- forced_revolute_* reports the revolute candidate for every GT revolute joint with a fit, regardless of selected type, reducing type-selection bias. Fixed joints have no physical axis/pivot error.",
        "- rotation_geodesic_deg is the mean SO(3) rotation difference across states 2–4, with identity rotation for fixed/prismatic predictions. angle_magnitude_mae_deg compares principal rotation magnitudes. These include type mistakes and complement axis orientation error; axis orientation is distinct from rotation amount.",
        "- summary.json includes full mean/median/std/p90/p95/min/max/count distributions, per-joint, per-depth and per-scene groups, confusion matrices, per-class recall, stationary comparison and joint-success thresholds. per_joint.csv retains every pair, including skips and all fitted parameters.",
        "- ground_truth_control.json runs the identical fitter on true relative tracks (first mode only). This checks estimator/convention error separately from model prediction error.", "",
        "## Detailed metrics", ""])
    for mode in modes:
        s = summary[mode]["overall"]
        ade = s["metrics"]["ADE_1_pixels"]["mean"]
        stationary = s["metrics"]["stationary_ADE_pixels"]["mean"]
        change = f"{100*(ade/stationary-1):.1f}% higher" if ade > stationary else f"{100*(1-ade/stationary):.1f}% lower"
        text.extend([f"For **{mode}**, mean ADE is {fmt(ade)} px versus {fmt(stationary)} px for the stationary baseline "
                     f"({change}). It beats stationary on "
                     f"{100*s['beats_stationary_fraction']:.1f}% of eligible pairs. "
                     f"Revolute recall is {fmt(s['recall']['revolute'])}; fixed recall is {fmt(s['recall']['fixed'])}. "
                     "Per-joint geometry errors are conditional on the correctly typed revolute cohort shown in the table.", ""])
    text.extend(["### Matched comparisons", "",
        "Both modes below use identical paired examples and states 2–4 for trajectory comparison. "
        "Axis/pivot comparisons retain only joints with a metric in both modes. Negative deltas favor with-poke. "
        "The 95% interval resamples scenes (2,000 bootstrap draws), preserving correlation among a scene's joints.", "",
        "| Metric | Matched n | Without | With | With − without | 95% CI of difference |",
        "|---|---:|---:|---:|---:|---:|"])
    for metric, comparison in comparisons.items():
        text.append(f"| {metric} | {comparison['n']} | {fmt(comparison['without_mean'])} | {fmt(comparison['with_mean'])} | "
                    f"{fmt(comparison['with_minus_without'])} | {', '.join(fmt(v) for v in comparison['scene_bootstrap_difference_ci95'])} |")
    text.extend(["", "### Ground-truth trajectory control", ""])
    control = summarize(gt)
    text.extend([f"On true tracks, the fitter classifies {control['type_correct']}/{control['joints_fitted']} joints correctly. "
                 f"The correctly typed revolute control has mean/median axis error "
                 f"{fmt(control['metrics']['axis_error_deg']['mean'])}/{fmt(control['metrics']['axis_error_deg']['median'])} degrees "
                 f"and mean/median pivot distance {fmt(control['metrics']['pivot_distance']['mean'])}/{fmt(control['metrics']['pivot_distance']['median'])} world units. "
                 f"With the transform-consistent pivot reference, mean/median control pivot distance is "
                 f"{fmt(control['metrics']['kinematic_pivot_distance']['mean'])}/{fmt(control['metrics']['kinematic_pivot_distance']['median'])}. "
                 "The control separates prediction error from fitting limits and the stored-pivot discrepancy documented above.", ""])
    for mode in modes:
        s = summary[mode]["overall"]
        text.extend([f"### {mode}", "", f"Type accuracy: {fmt(s['type_accuracy'])}; balanced accuracy: {fmt(s['balanced_accuracy'])}; "
                     f"fraction beating stationary: {fmt(s['beats_stationary_fraction'])}.", "",
                     "| Metric | n | Mean | Median | p90 | p95 | Max |", "|---|---:|---:|---:|---:|---:|---:|"])
        for metric, stats in s["metrics"].items():
            text.append(f"| {metric} | {stats['n']} | " + " | ".join(fmt(stats[k]) for k in ("mean","median","p90","p95","max")) + " |")
        text.extend(["", "Confusion matrix (GT rows / predicted columns):", "", "| GT | fixed | prismatic | revolute |", "|---|---:|---:|---:|"])
        for kind in KINDS:
            text.append(f"| {kind} | " + " | ".join(str(s["confusion"][kind][p]) for p in KINDS) + " |")
        text.append("")
    text.extend(["## Exclusions and examples", "",
                 "The following source joints had no eligible initially visible child point and were excluded before inference:", "",
                 "| Scene index | Joint ordinal | Joint index |", "|---|---:|---:|"])
    for entry in manifest["inventory"]:
        if not entry["eligible"]:
            text.append(f"| {entry['scene_index']} | {entry['joint_index']+1} | {entry['joint_index']} |")
    text.extend(["", "ranked_cases.json lists the ten best and worst examples for ADE, axis error and pivot distance in each mode. "
                 "ARTIFACTS.md documents the saved arrays and the model-free visualization command.", ""])
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    plot_results(out, summary, rows)


def plot_results(out, summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for mode in summary:
        for ax, metric, title in zip(axes, ("ADE_1_pixels", "axis_error_deg", "pivot_distance"),
                                    ("Trajectory ADE (model pixels)", "Axis orientation error (degrees)", "Pivot to stored GT (world units)")):
            x = sorted(r[metric] for r in rows if r["mode"] == mode and r.get(metric) is not None)
            if x:
                ax.plot(x, np.arange(1, len(x)+1)/len(x), label=f"{mode} (n={len(x)})")
            ax.set(xlabel=title, ylabel="Cumulative fraction", ylim=(0, 1.02))
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    axes[1].set_xlim(0,90)
    axes[2].set_xscale("symlog", linthresh=.01)
    axes[2].set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out / "error_distributions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for mode, result in summary.items():
        joints = result["per_joint"]
        for ax, metric, title in zip(axes, ("ADE_1_pixels", "axis_error_deg", "pivot_distance"),
                                    ("Mean ADE (pixels)", "Mean axis error (degrees)", "Median pivot distance (world units)")):
            key = "median" if metric == "pivot_distance" else "mean"
            ax.plot([int(j)+1 for j in joints], [joints[j]["metrics"][metric][key] for j in joints], "o-", label=mode)
            ax.set(xlabel="Joint ordinal (dataset index + 1)", ylabel=title)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "per_joint_metrics.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("finetuning_pair/runs/pair/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("finetuning_pair/runs/test1_first50_analysis"))
    p.add_argument("--data-root", type=Path, default=Path("data/Robotic_arm"))
    p.add_argument("--first-scenes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp-dtype", default="none", choices=("none", "auto", "bf16"))
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--modes", nargs="+", default=["without_poke", "with_poke"], choices=("without_poke", "with_poke"))
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    out = artifact_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        manifest = json.loads((out / "run.json").read_text())
        rows = [json.loads(f.read_text()) for f in sorted((out / "records").glob("*.json"))]
        report(out, manifest, rows)
        return
    if min(args.first_scenes, args.ensemble_size, args.diffusion_steps, args.threads) < 1:
        p.error("Counts must be positive")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    snapshot = out / "checkpoint.pt"
    if not snapshot.exists():
        os.link(args.checkpoint, snapshot)
    info = inspect_checkpoint(snapshot)
    cfg = info["train_config"]
    if cfg["target_definition"] != TARGET_DEFINITION or cfg["num_transitions"] != 4:
        raise ValueError("Requires four-state-transition child-relative checkpoint")
    ds = ArticulatedDataset(args.data_root, split="test1", training=False, seed=args.seed,
                           first_scenes=args.first_scenes,
                           points_per_link=cfg["points_per_link"], image_size=cfg["image_size"],
                           num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
                           overlay_alpha=cfg["overlay_alpha"], cache_dir="finetuning_pair/cache")
    ds.examples = [(s, pair) for s, pair in ds.examples if s < args.first_scenes]
    source_joint_count = 0
    inventory = []
    depths = {}
    for s, path in enumerate(ds.files[:args.first_scenes]):
        with np.load(path) as a:
            parents, children, js = a["joint_parent_link_indices"], a["joint_child_link_indices"], a["joint_indices"]
            source_joint_count += len(js)
            depth = {int(x): 0 for x in set(a["link_indices"]) - set(children)}
            pending = set(range(len(js)))
            while pending:
                ready = [i for i in pending if int(parents[i]) in depth]
                if not ready:
                    raise ValueError("Invalid hierarchy")
                for i in ready:
                    depth[int(children[i])] = depth[int(parents[i])] + 1
                    depths[s, int(js[i])] = depth[int(children[i])]
                    pending.remove(i)
            for j, gt in zip(js, a["joint_types"]):
                inventory.append(dict(scene_index=s, joint_index=int(j), gt_type=int(gt),
                                      eligible=any(si == s and pair[2] == j for si, pair in ds.examples)))
    manifest = dict(schema_version=1, checkpoint_sha256=file_hash(snapshot), checkpoint_step=info["step"],
                    checkpoint_source=str(args.checkpoint.resolve()), train_config=cfg,
                    device=args.device, amp_dtype=args.amp_dtype, ensemble_size=args.ensemble_size,
                    diffusion_steps=args.diffusion_steps, seed=args.seed, modes=args.modes,
                    scenes=[dict(path=str(f.resolve()), size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in ds.files[:args.first_scenes]],
                    pairs=ds.examples, all_source_joints=source_joint_count, inventory=inventory,
                    target_definition=TARGET_DEFINITION, torch_version=str(torch.__version__))
    manifest = clean(manifest)
    if (out / "run.json").exists() and json.loads((out / "run.json").read_text()) != manifest:
        raise ValueError("Run manifest differs; choose another output directory")
    write_json(out / "run.json", manifest)
    print(json.dumps(dict(event="dataset_ready", scenes=len(manifest["scenes"]), pairs=len(ds), source_joints=source_joint_count)), flush=True)
    pending = [(i, mode) for i in range(len(ds)) for mode in args.modes if not (out / "records" / f"{i:04d}_{mode}.json").exists()]
    model = None
    if pending:
        bare = build_myriad()
        reference = load_pretrained_weights(bare, Path(info["base_checkpoint"]["path"]))
        model = FineTuningModel(bare, LoRAConfig(**info["lora_config"])).to(args.device)
        load_finetuned_weights(snapshot, model, base_checkpoint=reference)
        model.eval()
        print(json.dumps(dict(event="model_ready", device=args.device)), flush=True)
    for index in range(len(ds)):
        scene_index, pair = ds.examples[index]
        if all((out / "records" / f"{index:04d}_{mode}.json").exists() for mode in args.modes):
            continue
        sample = ds[index]
        with np.load(sample["scene_path"]) as a:
            source = {k: a[k] for k in ("joint_indices", "joint_types", "joint_axes_world", "joint_origins_world",
                                        "camera_intrinsic", "camera_extrinsic", "link_transforms_world", "link_indices", "depth",
                                        "rgb", "link_segmentation", "metadata_json", "joint_parent_link_indices", "joint_child_link_indices")}
        jp = int(np.flatnonzero(source["joint_indices"] == pair[2])[0])
        by_label = {int(v): i for i, v in enumerate(source["link_indices"])}
        for mode in args.modes:
            record_path = out / "records" / f"{index:04d}_{mode}.json"
            if record_path.exists():
                continue
            started = time.perf_counter()
            row = dict(pair_index=index, scene_index=scene_index, scene_path=sample["scene_path"],
                       joint_index=pair[2], joint_ordinal=pair[2]+1, chain_depth=depths[scene_index, pair[2]],
                       parent_link=pair[0], child_link=pair[1], mode=mode,
                       gt_type={0:"revolute",1:"prismatic",4:"fixed"}.get(int(source["joint_types"][jp]),"other"))
            inputs = _scene_inputs(sample, mode == "with_poke", False)
            if inputs is None:
                row.update(skipped=True, reason="No conditionable and scoreable points")
            else:
                known, truth, valid, links = inputs
                attention = patch("myriad.model.create_block_mask", dense_block_mask) if args.device == "cpu" else nullcontext()
                with attention:
                    pred = sample_rollout(model, sample["x"], known, sample["state_times"],
                        ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps,
                        seed=args.seed+index, amp_dtype=args.amp_dtype).float().cpu().numpy()
                start = len(known)
                truth, valid = truth.numpy(), valid.numpy()
                errs = (np.square(2*(pred[:, start:] - truth[start:])).sum(-1)*valid[start:]).sum((1,2))/valid[start:].sum()
                arrays = dict(predictions=pred, ground_truth=truth, valid=valid, known_states=np.array(start),
                              image=sample["x"].numpy(), link_nodes=links.numpy())
                row.update(skipped=False, **scene_metrics(arrays, errs))
                distance = np.linalg.norm((pred[0]-truth)*(cfg["image_size"]-1), axis=-1)
                row["common_ADE_pixels"] = float(distance[STATES][valid[STATES]].mean())
                for state in range(start, 5):
                    row[f"ADE_state_{state}_pixels"] = float(distance[state][valid[state]].mean()) if valid[state].any() else None
                selected = sample["point_valid"] & sample["position_valid"][:start].all(0)
                pix = sample["reference_pixels"][selected].numpy().astype(float)
                depth = source["depth"][0, np.rint(pix[:,1]).astype(int), np.rint(pix[:,0]).astype(int)]
                xyz = _backproject(pix, depth, source["camera_intrinsic"], source["camera_extrinsic"])
                h, w = source["depth"].shape[1:]
                uv = from_normalized(pred[0], h, w, cfg["image_size"])[STATES]
                fit = fit_joint(xyz, uv, valid[STATES], source["camera_intrinsic"], source["camera_extrinsic"])
                row.update(geometry_metrics(fit, source, jp, by_label[pair[0]], by_label[pair[1]]))
                if mode == args.modes[0]:
                    gt_uv = from_normalized(truth, h, w, cfg["image_size"])[STATES]
                    control = fit_joint(xyz, gt_uv, valid[STATES], source["camera_intrinsic"], source["camera_extrinsic"])
                    row["ground_truth_control"] = dict(skipped=False, improves_stationary=False,
                        **geometry_metrics(control, source, jp, by_label[pair[0]], by_label[pair[1]]))
                pred_dir = out / "predictions"
                pred_dir.mkdir(exist_ok=True)
                np.savez_compressed(pred_dir / f"{index:04d}_{mode}.npz", **arrays,
                                    reference_world=xyz, source_pixels=pix, state_times=sample["state_times"].numpy(),
                                    pair_labels=np.array(pair[:2]), joint_index=np.array(pair[2]),
                                    hypothesis_errors=errs, point_indices=np.flatnonzero(selected.numpy()),
                                    state_fractions=sample["state_fractions"].numpy(),
                                    target_definition=np.array(TARGET_DEFINITION),
                                    camera_intrinsic=source["camera_intrinsic"], camera_extrinsic=source["camera_extrinsic"],
                                    initial_rgb=source["rgb"][0], initial_depth=source["depth"][0],
                                    initial_segmentation=source["link_segmentation"][0],
                                    source_metadata_json=source["metadata_json"],
                                    link_indices=source["link_indices"], link_transforms_world=source["link_transforms_world"],
                                    joint_axes_world=source["joint_axes_world"], joint_origins_world=source["joint_origins_world"],
                                    joint_indices=source["joint_indices"], joint_types=source["joint_types"],
                                    joint_parent_link_indices=source["joint_parent_link_indices"],
                                    joint_child_link_indices=source["joint_child_link_indices"],
                                    metrics_json=np.array(json.dumps(clean(row))))
            row["seconds"] = time.perf_counter()-started
            write_json(record_path, clean(row))
            print(json.dumps(dict(event="pair_complete", index=index, mode=mode, scene=scene_index,
                                  joint=pair[2], seconds=round(row["seconds"],2), ADE=row.get("ADE_1_pixels"))), flush=True)
    rows = [json.loads(f.read_text()) for f in sorted((out / "records").glob("*.json"))]
    report(out, manifest, rows)
    write_json(out / "status.json", dict(status="complete", completed=len(rows), expected=len(ds)*len(args.modes)))
    print(json.dumps(dict(event="complete", output=str(out))), flush=True)


if __name__ == "__main__":
    main()
