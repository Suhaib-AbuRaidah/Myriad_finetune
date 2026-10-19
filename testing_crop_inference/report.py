"""Paired original-coordinate scores; the production joint fitter is unchanged."""
from pathlib import Path
import json

import numpy as np

from finetuning_pair.evaluate import write_json
from training_from_scratch.full_evaluation.metrics import clean, summarize


KEYS = ("ADE_1_pixels", "oracle_5_ADE_pixels", "ADE_percent_full_path", "minADE_1", "minADE_5",
        "forced_revolute_axis_error_deg", "forced_revolute_kinematic_pivot_distance",
        "axis_error_deg", "kinematic_pivot_distance", "rotation_geodesic_deg", "FDE_1_pixels")


def paired(rows, original, key):
    comparisons = []
    for row in rows:
        old = original[row["pair_index"]]
        if row.get(key) is not None and old.get(key) is not None:
            comparisons.append((row["scene_index"], float(row[key])-float(old[key])))
    if not comparisons:
        return dict(n=0)
    values = np.array([r[1] for r in comparisons])
    scenes = sorted({r[0] for r in comparisons})
    sums = np.array([sum(v for s, v in comparisons if s == scene) for scene in scenes])
    counts = np.array([sum(s == scene for s, _ in comparisons) for scene in scenes])
    rng = np.random.default_rng(20261018)
    draws = rng.integers(len(scenes), size=(3000, len(scenes)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    return dict(n=len(values), mean_delta=float(values.mean()), median_delta=float(np.median(values)),
                fraction_improved=float((values < 0).mean()),
                scene_bootstrap_95CI=np.percentile(means, [2.5, 97.5]).tolist())


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def generate(out):
    out = Path(out)
    manifest = json.loads((out/"run.json").read_text())
    # JSON is serialized with sorted keys; never infer comparison column order
    # from a deserialized mapping (crop_10 sorts before original).
    variants = ["original", "crop_10"]
    if set(manifest["variants"]) != set(variants):
        raise ValueError("Expected exactly original and crop_10 variants")
    rows = {v: [] for v in variants}
    for v in variants:
        for i in manifest["pair_indices"]:
            path = out/"records"/f"{i:04d}_{v}.json"
            if not path.exists():
                raise ValueError(f"Incomplete run: {path}")
            rows[v].append(json.loads(path.read_text()))
    scored = {v: [r for r in data if not r.get("skipped")] for v, data in rows.items()}
    original = {r["pair_index"]: r for r in scored["original"]}
    result = dict(checkpoint_step=manifest["checkpoint_step"], checkpoint_sha256=manifest["checkpoint_sha256"],
                  scenes=len(manifest["sources"]), eligible_pairs=len(manifest["pair_indices"]),
                  all_pairs={v: summarize(rows[v]) for v in variants},
                  moving_pairs={v: summarize([r for r in rows[v] if r["gt_type"] != "fixed"]) for v in variants},
                  by_joint={}, by_gt_type={}, by_zoom={}, by_gt_path={}, paired={})
    for ordinal in sorted({r["joint_ordinal"] for r in scored["original"]}):
        result["by_joint"][str(ordinal)] = {v: summarize([r for r in scored[v] if r["joint_ordinal"] == ordinal]) for v in variants}
    for kind in ("revolute", "prismatic", "fixed"):
        result["by_gt_type"][kind] = {v: summarize([r for r in scored[v] if r["gt_type"] == kind]) for v in variants}
    for v in variants:
        result["paired"][v] = {key: paired(scored[v], original, key) for key in KEYS}
    for lo, hi in ((1, 1.25), (1.25, 1.5), (1.5, 2), (2, float("inf"))):
        ids = {r["pair_index"] for r in scored["crop_10"] if lo <= r["zoom"] < hi}
        result["by_zoom"][f"{lo}..{hi}"] = {v: summarize([r for r in scored[v] if r["pair_index"] in ids]) for v in variants}
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, float("inf"))):
        ids = {r["pair_index"] for r in scored["original"] if r.get("gt_path_pixels") is not None and lo <= r["gt_path_pixels"] < hi}
        result["by_gt_path"][f"{lo}..{hi}"] = {v: summarize([r for r in scored[v] if r["pair_index"] in ids]) for v in variants}
    crop = scored["crop_10"]
    result["crop_statistics"] = dict(mean_zoom=float(np.mean([r["zoom"] for r in crop])),
        median_zoom=float(np.median([r["zoom"] for r in crop])), max_zoom=max(r["zoom"] for r in crop),
        pairs_actually_zoomed=sum(r["zoom"] > 1 for r in crop),
        pair_mean_gt_outside_crop_fraction=float(np.mean([r["gt_outside_crop_fraction"] for r in crop])),
        pairs_with_gt_exiting_crop=sum(r["gt_outside_crop_fraction"] > 0 for r in crop),
        same_trajectory_length_magnification="exactly original_side/crop_side for every track; no geometric motion changes")
    write_json(out/"summary.json", clean(result))

    # Check paired coverage and all saved coordinate systems independently.
    checked, camera_max, restore_max = 0, 0., 0.
    for v in variants:
        if {r["pair_index"] for r in scored[v]} != set(original):
            raise AssertionError("Variants have different scored pair cohorts")
        for r in scored[v]:
            b = original[r["pair_index"]]
            assert r["selected_points"] == b["selected_points"]
            assert r["valid_future_count"] == b["valid_future_count"]
            with np.load(out/"predictions"/f'{r["pair_index"]:04d}_{v}.npz', allow_pickle=False) as a:
                with np.load(out/"predictions"/f'{r["pair_index"]:04d}_original.npz', allow_pickle=False) as o:
                    np.testing.assert_array_equal(a["valid"], o["valid"])
                    np.testing.assert_array_equal(a["ground_truth"], o["ground_truth"])
                source_path = Path(str(a["source_path"]))
                with np.load(source_path, allow_pickle=False) as source:
                    h, w = source["depth"].shape[-2:]
                    side = min(h, w)
                    source_top, source_left = round((h-side)/2), round((w-side)/2)
                    e = source["camera_extrinsic"]
                raw = a["cropped_predictions"].astype(float)
                pixels_crop = raw*(manifest["train_config"]["image_size"]-1)
                hraw = np.concatenate([pixels_crop, np.ones((*pixels_crop.shape[:-1], 1))], axis=-1)
                pixels_source = (hraw @ np.linalg.inv(a["crop_matrix"]).T)[..., :2]
                restored = ((pixels_source-[source_left, source_top]+.5)*manifest["train_config"]["image_size"]/side-.5)/(manifest["train_config"]["image_size"]-1)
                error = float(np.max(np.abs(restored-a["predictions"])))
                restore_max = max(restore_max, error)
                if error > 5e-7:
                    raise AssertionError("Saved inverse-crop predictions are inconsistent")
                xyz = a["reference_world"]
                xyz_h = np.column_stack([xyz, np.ones(len(xyz))])
                camera = (xyz_h @ e.T)[:, :3]
                pixel_h = camera @ a["camera_intrinsic_model"].T
                expected = pixel_h[:, :2]/pixel_h[:, 2:]
                observed = a["cropped_ground_truth"][0]*(manifest["train_config"]["image_size"]-1)
                camera_max = max(camera_max, float(np.max(np.abs(expected-observed))))
                if camera_max > .01:
                    raise AssertionError("Camera calibration and cropped initial positions disagree")
            checked += 1
    checks = [r.get("previous_prediction_max_abs_pixel_difference") for r in scored["original"]]
    checks = [r for r in checks if r is not None]
    audit = dict(passed=True, checked_records=checked, equal_pair_point_and_validity_cohorts=True,
        maximum_inverse_crop_normalized_error=restore_max, maximum_initial_camera_pixel_error=camera_max,
        previous_original_prediction_max_pixel_difference=max(checks) if checks else None,
        fitted_algorithm="unchanged finetuning_pair.analyze_joints.fit_joint; no optimization/tuning loop",
        checkpoint_sha256=manifest["checkpoint_sha256"])
    write_json(out/"audit.json", audit)

    lines = ["# Parent–child crop evaluation on test1", "",
        f'Checkpoint: step **{manifest["checkpoint_step"]:,}**, SHA-256 `{manifest["checkpoint_sha256"]}`.', "",
        f'{result["scenes"]} scenes; {result["eligible_pairs"]} eligible pairs; '
        f'{len(original)} scoreable pairs. Original-image and cropped-image inference use identical keypoints, '
        'seeds, five hypotheses and 16 diffusion steps. Only state 0 is supplied (no motion hint).', "",
        'The fixed crop is the square bounding box of the **initial visible parent ∪ child masks**, with '
        '10% margin on each side (at least two source pixels). It is shifted inside the image and capped '
        'at the original view size. RGB is resized to 384×384 with the original role overlays. '
        'The background/context inside the square is retained. No future states determine the crop.', "",
        'DINOv3 and the motion model both run in FP32. In the two-worker run, cuda:0 and cuda:1 each '
        'process disjoint pairs with their own motion model and live encoder. '
        'Embeddings pass through the same FP16 storage rounding used in training and are widened to FP32; '
        'no cropped feature files are cached. Predictions are inverse-transformed to the original view '
        'before scoring and joint fitting. Masks, initial/future coordinates, flow targets and intrinsics '
        'are transformed consistently. Initial depth, 3D points, extrinsics and physical/world joint units remain valid.', "",
        '**The existing joint fitter is unchanged. No fitting-algorithm search or iteration was performed.**', "",
        '## Overall paired metrics', "",
        '| Metric (lower is better) | Original | Cropped | Crop − original; scene-bootstrap 95% CI |',
        '|---|---:|---:|---|']
    display = [("ADE, original 384-view pixels", "ADE_1_pixels"),
               ("Oracle best-of-5 ADE, pixels", "oracle_5_ADE_pixels"),
               ("ADE / GT full path, %", "ADE_percent_full_path"),
               ("minADE_1 (training definition)", "minADE_1"),
               ("Axis error, all fitted GT-revolute pairs, degrees", "forced_revolute_axis_error_deg"),
               ("Axis error, correctly typed revolute only, degrees", "axis_error_deg"),
               ("Pivot distance to transform-consistent axis, all fitted GT-revolute", "forced_revolute_kinematic_pivot_distance"),
               ("Pivot distance, correctly typed revolute only", "kinematic_pivot_distance"),
               ("Rotation geodesic, degrees", "rotation_geodesic_deg"), ("Endpoint error, pixels", "FDE_1_pixels")]
    for label, key in display:
        a, b = (result["all_pairs"][v]["metrics"][key] for v in variants)
        delta = result["paired"]["crop_10"][key]
        d = "—"
        if delta["n"]:
            low, high = delta["scene_bootstrap_95CI"]
            d = f'{number(delta["mean_delta"])} [{number(low)}, {number(high)}]; n={delta["n"]}'
        lines.append(f'| {label} | {number(a["mean"], 5)} (n={a["n"]}) | {number(b["mean"], 5)} (n={b["n"]}) | {d} |')
    for v in variants:
        a = result["all_pairs"][v]
        lines.extend(["", f'**{v}:** type accuracy {number(100*a["type_accuracy"], 2)}%; '
            f'{a["fitted"]} fits, {a["fit_failures"]} fit failures; confusion matrix (GT → predicted): '
            f'`{json.dumps(a["confusion"], sort_keys=True)}`.'])
    lines.extend(["", 'The primary pixel ADE is the mean Euclidean error of hypothesis 0 over original-valid states 1–4, '
        'then averaged equally over pairs. Path percentage is averaged per complete non-fixed track with '
        'GT path ≥1 original-view pixel, then over pairs. `minADE_1` is the existing **4 × mean squared '
        'normalized 2D error**, rather than Euclidean pixel ADE. Oracle best-of-5 uses future GT only for scoring.', "",
        'Axis errors are sign-invariant (0–90°). All-revolute candidate scores avoid changing the axis cohort '
        'when joint-type predictions change; fit counts still matter. Correct-type-only metrics can compare '
        'different subsets. Pivot distances use dataset world units and the transform-consistent GT axis. '
        'The stored-origin alternative and all fitted parameters remain in per-pair records.', "",
        '## Per-joint metrics', "",
        '| Joint ordinal | Pairs | ADE original → crop (px) | Axis original → crop (°), all GT revolute | Pivot original → crop | Type accuracy original → crop |',
        '|---|---:|---|---|---|---|'])
    for ordinal, group in result["by_joint"].items():
        a, b = group["original"], group["crop_10"]
        def transition(key):
            return f'{number(a["metrics"][key]["mean"])} → {number(b["metrics"][key]["mean"])}'
        lines.append(f'| {ordinal} | {a["scored"]} | {transition("ADE_1_pixels")} | '
            f'{transition("forced_revolute_axis_error_deg")} | {transition("forced_revolute_kinematic_pivot_distance")} | '
            f'{number(100*a["type_accuracy"], 1)}% → {number(100*b["type_accuracy"], 1)}% |')
    cs = result["crop_statistics"]
    lines.extend(["", '## Crop size and coverage', "",
        f'Mean zoom **{cs["mean_zoom"]:.3f}×**, median {cs["median_zoom"]:.3f}×, maximum {cs["max_zoom"]:.3f}×; '
        f'{cs["pairs_actually_zoomed"]}/{len(crop)} scored pairs actually zoom in. Every trajectory length in '
        'crop pixels is multiplied by exactly its pair’s zoom factor.', "",
        f'{cs["pairs_with_gt_exiting_crop"]} pairs have originally valid future targets outside the crop; '
        f'the pair-mean outside fraction is {100*cs["pair_mean_gt_outside_crop_fraction"]:.2f}%. '
        '**All of these targets still count in the primary metrics.** No prediction is clipped to crop bounds.', "",
        '| Zoom | Pairs | ADE original → crop (px) | Axis original → crop (°) |', '|---|---:|---|---|'])
    for label, group in result["by_zoom"].items():
        a, b = group["original"], group["crop_10"]
        lines.append(f'| {label} | {a["scored"]} | {number(a["metrics"]["ADE_1_pixels"]["mean"])} → '
            f'{number(b["metrics"]["ADE_1_pixels"]["mean"])} | '
            f'{number(a["metrics"]["forced_revolute_axis_error_deg"]["mean"])} → '
            f'{number(b["metrics"]["forced_revolute_axis_error_deg"]["mean"])} |')
    lines.extend(["", '## Checks and interpretation limits', "",
        f'Paired coverage/coordinate audit: **passed** for {checked} records. Maximum inverse-crop error '
        f'{restore_max:.3g} normalized units; initial 3D reprojection disagreement {camera_max:.3g} crop pixels.', "",
        f'Original predictions versus the previous full evaluation: maximum disagreement '
        f'{number(audit["previous_original_prediction_max_pixel_difference"], 6)} original-view pixels.', "",
        'This evaluates a model trained on the original full images under a new crop distribution. '
        'A gain would support investigating crop-aware training; a regression does not prove crop-aware '
        'retraining cannot help. Zoom also changes context, apparent scale and positional features, so '
        'this experiment does not isolate trajectory length alone. Initial segmentation is assumed known, '
        'as in the original pipeline. This already-inspected test split is development evidence.', "",
        'See `summary.json` for moving-only, per-joint, per-GT-type, GT-path and zoom groups; '
        '`records/` for full joint parameters and trajectory metrics; `predictions/` for both coordinate '
        'systems and camera matrices; `run.json` for provenance; `runtime.json` for memory and live-feature checks.', ""])
    (out/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(event="report_complete", path=str(out/"REPORT.md"), audit=audit)), flush=True)
    return result
