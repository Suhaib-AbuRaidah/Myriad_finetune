"""One initial-only rollout per pair; fitting and metrics in source coordinates."""
import time
import cv2
import numpy as np
import torch

from finetuning_pair.testing import sample_rollout
from finetuning_pair.analyze_joints import geometry_metrics
from training_from_scratch.dataset import _backproject
from training_from_scratch.full_evaluation.metrics import kinematic_reference, attach_reference, clean, distribution
from training_from_scratch.sampling import sampler_attention_context
from finetuning_pair.checkpoint_io import capture_rng_state, restore_rng_state
from .geometry import SquareCrop
from .fitting import fit_predictions
from .scoring import axis_error, checkpoint_score, DEFINITION
from .selection import validate_cohort

METRICS = ("axis_error_deg", "pivot_distance", "rotation_geodesic_deg", "angle_magnitude_mae_deg",
           "trajectory_ADE", "pixel_ADE", "GT_path_normalized_error_percent", "prismatic_axis_error_deg")


def trajectory_diagnostics(pred, truth, valid, crop_valid, image_hw):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    d = np.linalg.norm(pred-truth, axis=-1)
    score = valid[1:]
    if not score.any() or not np.isfinite(pred).all():
        raise ValueError("No scoreable finite future prediction")
    h, w = image_hw
    full = SquareCrop(round((h-min(h,w))/2), round((w-min(h,w))/2), min(h,w), 384)
    norm = np.linalg.norm(full.normalize(pred)-full.normalize(truth), axis=-1)
    complete = valid.all(0)
    path = np.linalg.norm(np.diff(truth, axis=0), axis=-1).sum(0)
    eligible = complete & (path >= 1.)
    return dict(pixel_ADE=float(d[1:][score].mean()), trajectory_ADE=float(norm[1:][score].mean()),
        paper_style_MSE=float(4*np.square(norm[1:][score]).mean()),
        GT_path_normalized_error_percent=float((100*d[1:,eligible].mean(0)/path[eligible]).mean()) if eligible.any() else None,
        path_metric_tracks=int(eligible.sum()), original_valid_future=int(score.sum()),
        future_outside_crop=int((score & ~crop_valid[1:]).sum()),
        future_outside_crop_fraction=float((score & ~crop_valid[1:]).sum()/score.sum()))


def summarize(rows, cohort):
    fitted = [r for r in rows if r.get("fit_available")]
    kinds = ("fixed", "prismatic", "revolute")
    confusion = {g:{p:sum(r["gt_type"] == g and r.get("predicted_type", "failure") == p for r in rows)
                    for p in (*kinds, "failure")} for g in kinds}
    total = sum(r.get("original_valid_future", 0) for r in rows)
    outside = sum(r.get("future_outside_crop", 0) for r in rows)
    return clean(dict(pairs=len(rows), axis_cohort_size=len(cohort),
        checkpoint_score=checkpoint_score(rows, cohort) if cohort else None,
        definition=DEFINITION, fit_failures=len(rows)-len(fitted),
        prediction_failures=sum("prediction_failure" in r for r in rows),
        axis_failures=sum(r.get("axis_failure", False) for r in rows),
        type_accuracy=sum(r.get("type_correct", False) for r in rows)/len(rows), confusion=confusion,
        metric_coverage={m:sum(r.get(m) is not None for r in rows) for m in METRICS},
        metrics={m:distribution(r.get(m) for r in rows) for m in METRICS},
        future_outside_crop=outside, original_valid_future=total,
        future_outside_crop_fraction=outside/total if total else None))


def score_prediction(sample, pred, descriptor):
    row = dict(index=descriptor["index"], scene=descriptor["scene"],
               joint_ordinal=descriptor["joint_ordinal"], gt_type=descriptor["joint_type"],
               fit_available=False)
    selected = sample["point_valid"].numpy() & sample["source_position_valid"][0].numpy()
    truth = sample["source_trajectories"].numpy()[:, selected]
    valid = sample["source_position_valid"].numpy()[:, selected]
    cropped_valid = sample["crop_position_valid"].numpy()[:, selected]
    score_mask = valid[1:]
    row.update(original_valid_future=int(score_mask.sum()),
        future_outside_crop=int((score_mask & ~cropped_valid[1:]).sum()))
    crop = SquareCrop(*sample["crop_bounds"].tolist())
    uv = crop.unnormalize(pred)
    with np.load(sample["scene_path"], allow_pickle=False) as a:
        source = {k:a[k] for k in ("joint_indices", "joint_types", "joint_axes_world", "joint_origins_world",
                    "link_indices", "link_transforms_world", "camera_intrinsic", "camera_extrinsic")}
        depth = a["depth"][0].copy()
        depth[~a["depth_valid_mask"][0].astype(bool)] = 0
        seg = a["link_segmentation"][0]
    jp = int(np.flatnonzero(source["joint_indices"] == sample["joint_index"])[0])
    lookup = {int(x):i for i,x in enumerate(source["link_indices"])}
    pa, ch = [lookup[int(x)] for x in sample["pair_labels"]]
    gt_axis = source["joint_axes_world"][0,jp]
    row["gt_axis_world"] = gt_axis if row["gt_type"] != "fixed" else None
    try:
        row.update(trajectory_diagnostics(uv, truth, valid, cropped_valid, sample["source_image_hw"].tolist()))
    except ValueError as error:
        row["prediction_failure"] = str(error)
    else:
        pixels = sample["reference_pixels"].numpy()[selected].astype(float)
        z = depth[np.rint(pixels[:,1]).astype(int), np.rint(pixels[:,0]).astype(int)]
        xyz = _backproject(pixels, z, source["camera_intrinsic"], source["camera_extrinsic"])
        try:
            fit = fit_predictions(xyz, uv, source["camera_intrinsic"], source["camera_extrinsic"],
                initial_depth=depth, initial_segmentation=seg, pair_labels=sample["pair_labels"].numpy())
            row.update(geometry_metrics(fit, source, jp, pa, ch))
            if fit is not None:
                row["fitting_diagnostics"] = fit["diagnostics"]
                candidate = fit["fits"].get("revolute")
                raw = np.asarray(candidate["params"][:3],float) if candidate is not None else None
                if raw is None or not np.isfinite(raw).all() or np.linalg.norm(raw) < 1e-12:
                    # The shared diagnostic helper's unit() has a default axis
                    # for degenerate vectors. That default cannot count as a fit.
                    row.pop("forced_revolute_axis_world", None)
                if row["gt_type"] == "prismatic":
                    prism = fit["fits"].get("prismatic")
                    row["prismatic_axis_error_deg"] = axis_error(prism["params"][:3] if prism else None,gt_axis)
            if row["gt_type"] == "revolute":
                reference = kinematic_reference(source, jp, pa, ch)
                attach_reference(row, reference)
                row["pivot_distance"] = row.get("forced_revolute_kinematic_pivot_distance")
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as error:
            row["fit_failure"] = str(error)
            row["fit_available"] = False
            row.pop("forced_revolute_axis_world", None)
    if row["gt_type"] == "revolute":
        predicted_axis = row.get("forced_revolute_axis_world")
        row["axis_error_deg"] = axis_error(predicted_axis, gt_axis)
        row["axis_failure"] = predicted_axis is None or not np.isfinite(predicted_axis).all() or np.linalg.norm(predicted_axis) < 1e-12
    if row["gt_type"] == "fixed":
        row.pop("gt_axis_world", None)
        row.pop("gt_pivot_world", None)
        row["GT_path_normalized_error_percent"] = None
    return clean(row), uv


def evaluate(model, dataset, manifest, *, diffusion_steps=16, seed=42, emit=None, predictions_dir=None):
    validate_cohort(manifest)
    rows, started = [], time.perf_counter()
    rng = capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        with sampler_attention_context():
            for descriptor in manifest["pairs"]:
                index = descriptor["index"]
                cv2.setRNGSeed((seed+index) % (2**31-1))
                sample = dataset[index]
                selected = sample["point_valid"] & sample["source_position_valid"][0]
                prediction = sample_rollout(model, sample["x"], sample["trajectories"][:1, selected],
                    sample["state_times"], ensemble_size=1, diffusion_steps=diffusion_steps,
                    seed=seed+index, amp_dtype="none")[0].float().cpu().numpy()
                row, uv = score_prediction(sample, prediction, descriptor)
                rows.append(row)
                if predictions_dir is not None:
                    predictions_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(predictions_dir/f"pair_{index:05d}.npz", predictions_source_pixels=uv,
                        point_indices=np.flatnonzero(selected.numpy()), crop_transform=sample["crop_transform"].numpy(),
                        crop_inverse=sample["crop_inverse"].numpy())
                if emit:
                    emit(dict(event="validation_pair", completed=len(rows), total=100,
                              index=index, axis_error_deg=row.get("axis_error_deg")))
    finally:
        model.train(was_training)
        restore_rng_state(rng)
    overall = summarize(rows, manifest["axis_cohort"])
    per_joint = {}
    for ordinal in sorted({r["joint_ordinal"] for r in rows}):
        group = [r for r in rows if r["joint_ordinal"] == ordinal]
        per_joint[str(ordinal)] = summarize(group, [r["index"] for r in group if r["gt_type"] == "revolute"])
    return dict(overall=overall, per_joint_ordinal=per_joint, rows=rows,
                validation_seconds=time.perf_counter()-started, hypotheses_per_pair=1,
                diffusion_steps=diffusion_steps, initial_states_supplied=1, manifest=manifest["sha256"])
