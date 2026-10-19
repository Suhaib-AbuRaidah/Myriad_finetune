"""Explicit trajectory normalization, geometry references, and aggregation."""
from __future__ import annotations

import math
import numpy as np

from finetuning_pair.evaluate import scene_metrics

KINDS = ("fixed", "prismatic", "revolute")
METRICS = (
    "minADE_1", "minADE_5", "ADE_1_pixels", "FDE_1_pixels", "RMSE_1_pixels",
    "ADE_source_pixels", "common_ADE_pixels", "oracle_5_ADE_pixels",
    "ADE_percent_full_path", "common_ADE_percent_full_path", "ADE_percent_forecast_path",
    "ADE_percent_path_weighted", "gt_path_pixels", "stationary_ADE_pixels",
    "axis_error_deg", "prismatic_axis_error_deg", "pivot_distance", "kinematic_pivot_distance",
    "forced_revolute_axis_error_deg", "forced_revolute_pivot_distance",
    "forced_revolute_kinematic_pivot_distance", "rotation_geodesic_deg",
    "rotation_endpoint_deg", "angle_magnitude_mae_deg", "selected_fit_rmse_pixels",
    "out_of_frame_fraction_1", "PCK_2_pixels", "PCK_5_pixels", "PCK_10_pixels", "PCK_20_pixels",
    "ADE_state_1_pixels", "ADE_state_2_pixels", "ADE_state_3_pixels", "ADE_state_4_pixels",
)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def distribution(values):
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    if not len(x):
        return dict(n=0, mean=None, median=None, std=None, p90=None, p95=None, min=None, max=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()))


def trajectory_metrics(predictions, truth, valid, *, known_states, image_size,
                       gt_type, min_path_pixels=1., source_side=None):
    """Primary percentages average per-track ADE/path, then pairs equally.

    Only complete, GT-moving tracks with path >= min_path_pixels enter a
    percentage. Fixed, near-stationary, and incomplete tracks retain absolute
    metrics and explicit exclusion counts. No denominator clipping is used.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pred.ndim != 4 or pred.shape[1:] != gt.shape or gt.shape[:2] != valid.shape:
        raise ValueError("Expected predictions [E,T,N,2], truth [T,N,2], valid [T,N]")
    if not 1 <= known_states < len(gt) or image_size < 2 or min_path_pixels <= 0:
        raise ValueError("Invalid known states, image size, or minimum path")
    score_mask = valid[known_states:]
    if not score_mask.any() or not np.isfinite(pred).all() or not np.isfinite(gt[valid]).all():
        raise ValueError("Need finite predictions/targets and valid future points")
    difference = pred[:, known_states:] - gt[None, known_states:]
    squared = (4 * np.square(difference).sum(-1) * score_mask).sum((1, 2)) / score_mask.sum()
    arrays = dict(predictions=pred, ground_truth=gt, valid=valid, known_states=known_states,
                  image=np.empty((3, image_size, image_size), dtype=np.uint8),
                  link_nodes=np.zeros(gt.shape[1], dtype=int))
    row = scene_metrics(arrays, squared)
    distances = np.linalg.norm((pred[0] - gt) * (image_size - 1), axis=-1)
    common_start = min(2, len(gt) - 1)
    row["common_ADE_pixels"] = float(distances[common_start:][valid[common_start:]].mean())
    for state in range(known_states, len(gt)):
        row[f"ADE_state_{state}_pixels"] = float(distances[state][valid[state]].mean()) if valid[state].any() else None
    for threshold in (2, 5, 10, 20):
        row[f"PCK_{threshold}_pixels"] = float((distances[known_states:][score_mask] <= threshold).mean())
    row["ADE_source_pixels"] = row["ADE_1_pixels"] * source_side / image_size if source_side else None
    complete = valid.all(0)
    path = np.linalg.norm(np.diff(gt, axis=0) * (image_size - 1), axis=-1).sum(0)
    forecast_path = np.linalg.norm(np.diff(gt[known_states-1:], axis=0) * (image_size - 1), axis=-1).sum(0)
    ade = distances[known_states:].mean(0)
    common_ade = distances[common_start:].mean(0)
    eligible = complete & (path >= min_path_pixels) & (gt_type != "fixed")
    forecast_eligible = complete & (forecast_path >= min_path_pixels) & (gt_type != "fixed")
    row.update(
        tracks_total=int(len(path)), complete_tracks=int(complete.sum()),
        normalized_tracks=int(eligible.sum()), incomplete_tracks=int((~complete).sum()),
        fixed_tracks_excluded=int(complete.sum()) if gt_type == "fixed" else 0,
        short_path_tracks_excluded=int((complete & (path < min_path_pixels)).sum()) if gt_type != "fixed" else 0,
        gt_path_pixels=float(path[complete].mean()) if complete.any() else None,
        ADE_percent_full_path=float(np.mean(100 * ade[eligible] / path[eligible])) if eligible.any() else None,
        common_ADE_percent_full_path=float(np.mean(100 * common_ade[eligible] / path[eligible])) if eligible.any() else None,
        ADE_percent_path_weighted=float(100 * ade[eligible].sum() / path[eligible].sum()) if eligible.any() else None,
        ADE_percent_forecast_path=float(np.mean(100 * ade[forecast_eligible] / forecast_path[forecast_eligible])) if forecast_eligible.any() else None,
    )
    track_rows = [dict(track=i, complete=bool(complete[i]), path_pixels=float(path[i]) if complete[i] else None,
                       ADE_pixels=float(ade[i]) if complete[i] else None,
                       percentage_eligible=bool(eligible[i]),
                       ADE_percent_path=float(100 * ade[i] / path[i]) if eligible[i] else None)
                  for i in range(len(path))]
    return clean(row), clean(track_rows)


def kinematic_reference(source, joint_pos, parent, child):
    """Axis-line reference consistent with the frozen-parent target transforms."""
    from finetuning.joint_parame_estim.run import axis_distance, axis_error, unit
    transforms = source["link_transforms_world"].astype(float)
    nominal = source["joint_origins_world"][0, joint_pos].astype(float)
    stored_axis = unit(source["joint_axes_world"][0, joint_pos].astype(float))
    blocks, translations = [], []
    for state in range(1, len(transforms)):
        motion = transforms[0, parent] @ np.linalg.inv(transforms[state, parent]) @ transforms[state, child] @ np.linalg.inv(transforms[0, child])
        blocks.append(np.eye(3) - motion[:3, :3])
        translations.append(motion[:3, 3])
    matrix, rhs = np.concatenate(blocks), np.concatenate(translations)
    singular = np.linalg.svd(matrix, full_matrices=False)
    if singular[1][0] < 1e-7:
        return dict(observable=False, reason="No measurable relative rotation")
    axis = singular[2][-1]
    if axis @ stored_axis < 0:
        axis *= -1
    pivot = np.linalg.lstsq(np.vstack([matrix, axis]), np.r_[rhs, axis @ nominal], rcond=1e-7)[0]
    return clean(dict(observable=True, axis=axis, pivot=pivot, stored_pivot=nominal,
        stored_axis_error_deg=axis_error(axis, stored_axis),
        stored_pivot_axis_offset=axis_distance(pivot, axis, nominal),
        transform_residual=float(np.linalg.norm(matrix @ pivot - rhs)),
        stored_pivot_transform_residual=float(np.linalg.norm(matrix @ nominal - rhs))))


def attach_reference(row, reference):
    from finetuning.joint_parame_estim.run import axis_distance
    row["kinematic_reference"] = reference
    if not reference.get("observable"):
        return
    for axis_key, pivot_key, metric in (
        ("estimated_axis_world", "estimated_pivot_world", "kinematic_pivot_distance"),
        ("forced_revolute_axis_world", "forced_revolute_pivot_world", "forced_revolute_kinematic_pivot_distance"),
    ):
        if row.get(axis_key) is not None and row.get(pivot_key) is not None:
            row[metric] = axis_distance(np.asarray(row[pivot_key]), np.asarray(row[axis_key]), np.asarray(reference["pivot"]))


def summarize(rows, *, source_count=None):
    scored = [r for r in rows if not r.get("skipped")]
    fitted = [r for r in scored if r.get("fit_available")]
    confusion = {g: {p: sum(r["gt_type"] == g and r.get("predicted_type") == p for r in fitted)
                     for p in KINDS} for g in KINDS}
    recalls = {g: confusion[g][g] / sum(r["gt_type"] == g for r in scored)
               if any(r["gt_type"] == g for r in scored) else None for g in KINDS}
    correct = sum(bool(r.get("type_correct")) for r in fitted)
    return clean(dict(
        pairs=len(rows), scored=len(scored), fitted=len(fitted), fit_failures=len(scored)-len(fitted),
        source_count=source_count, source_coverage=len(scored)/source_count if source_count else None,
        scenes=len({r["scene_index"] for r in rows}),
        type_accuracy=correct/len(scored) if scored else None,
        type_accuracy_given_fit=correct/len(fitted) if fitted else None,
        balanced_accuracy=float(np.mean([v for v in recalls.values() if v is not None])) if any(v is not None for v in recalls.values()) else None,
        confusion=confusion, recall=recalls,
        gt_types={g: sum(r["gt_type"] == g for r in scored) for g in KINDS},
        fit_failures_by_type={g: sum(r["gt_type"] == g and not r.get("fit_available") for r in scored) for g in KINDS},
        beats_stationary_fraction=float(np.mean([r["improves_stationary"] for r in scored])) if scored else None,
        tracks={key: sum(r.get(key, 0) for r in scored) for key in (
            "tracks_total", "complete_tracks", "normalized_tracks", "incomplete_tracks",
            "fixed_tracks_excluded", "short_path_tracks_excluded")},
        metrics={key: distribution(r.get(key) for r in scored) for key in METRICS},
        joint_success={f"axis_le_{a}_pivot_le_{p}":
            sum(r.get("axis_error_deg") is not None and r["axis_error_deg"] <= a and
                r.get("kinematic_pivot_distance") is not None and r["kinematic_pivot_distance"] <= p for r in scored) /
            max(1, sum(r["gt_type"] == "revolute" for r in scored)) for a, p in ((5, .01), (10, .05), (15, .1), (30, .1))},
    ))
