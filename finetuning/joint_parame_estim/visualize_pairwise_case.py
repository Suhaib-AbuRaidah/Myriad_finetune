"""Visualize a saved full-scene versus parent-child joint prediction.

Uses cached trajectories and reruns only the inexpensive joint fits; MYRIAD is
never loaded. Run from the repository root with the project conda environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finetuning.dataset import ArticulatedDataset
from finetuning.joint_parame_estim.pairwise_subset import ROOT, hierarchy, pair_record
from finetuning.joint_parame_estim.run import (
    STATES, axis_error, estimate_pose, fit_model, from_normalized, initial_axis,
    modeled_points, project, scene_data, unit,
)


def load_case(scene_index: int, joint_index: int, mode: str, run_dir: Path):
    dataset = ArticulatedDataset(ROOT / "data/Robotic_arm", split="test1", training=False,
                                 points_per_link=8, image_size=384, seed=42)
    sample = dataset[scene_index]
    source = scene_data(dataset, scene_index)
    positions = np.where(source["joint_indices"] == joint_index)[0]
    if len(positions) != 1:
        raise ValueError(f"Joint {joint_index} does not occur exactly once in scene {scene_index}")
    joint_pos = int(positions[0])
    result_path = (ROOT / "finetuning/runs/test1_presentation" / mode / "results"
                   / f"scene_{scene_index:04d}_{source['path'].stem}" / "results.npz")
    with np.load(result_path, allow_pickle=False) as saved:
        full_indices = saved["point_indices"].astype(int)
        full_tracks = saved["predictions"][0].astype(float)
        validity = saved["position_valid"].astype(bool)
    full_lookup = {int(index): offset for offset, index in enumerate(full_indices)}
    by_label = {int(label): node for node, label in enumerate(source["link_indices"])}
    identity = [(np.eye(3), np.zeros(3), True) for _ in STATES]
    poses = {name: {-1: identity} for name in ("full_scene", "parent_child")}
    records = {}
    case = {}
    h, w = source["rgb"].shape[1:3]

    for current_pos in hierarchy(source):
        parent = int(source["joint_parent_link_indices"][current_pos])
        child = int(source["joint_child_link_indices"][current_pos])
        number = int(source["joint_indices"][current_pos])
        cache = run_dir / "predictions" / mode / f"scene_{scene_index:04d}_joint_{number}.npz"
        with np.load(cache, allow_pickle=False) as saved:
            indices = saved["point_indices"].astype(int)
            pair_tracks = saved["predictions"][0].astype(float)
        full_for_pair = full_tracks[:, [full_lookup[int(i)] for i in indices]]
        valid_for_pair = validity[:, [full_lookup[int(i)] for i in indices]]
        parent_node, child_node = by_label[parent], by_label[child]
        nodes = source["point_link_nodes"][indices]
        for approach, tracks in (("full_scene", full_for_pair), ("parent_child", pair_tracks)):
            parent_poses = poses[approach][parent]
            record, child_poses = pair_record(source, sample, current_pos, mode, approach,
                                              indices, tracks, valid_for_pair,
                                              scene_index, parent_poses)
            if record is None:
                raise RuntimeError(f"Joint {number} could not be fitted for {approach}")
            poses[approach][child] = child_poses
            if current_pos != joint_pos:
                continue
            if record["predicted_type"] != "revolute":
                raise RuntimeError(f"Expected revolute fit, got {record['predicted_type']}")
            image = from_normalized(tracks, h, w)
            truth = from_normalized(sample["trajectories"][:, indices].numpy(), h, w)
            child_mask = nodes == child_node
            parent_mask = nodes == parent_node
            mask = valid_for_pair[STATES][:, child_mask] & source["point_valid"][indices][child_mask][None]
            xyz = source["world"][indices][child_mask]
            child_init = [estimate_pose(xyz, image[state, child_mask], mask[step],
                                        source["camera_intrinsic"], source["camera_extrinsic"])
                          for step, state in enumerate(STATES)]
            axis0, pivot0, theta0 = initial_axis(child_init, parent_poses, xyz.mean(0))
            fitted = fit_model("revolute", xyz, image[STATES][:, child_mask], mask,
                               parent_poses, source["camera_intrinsic"], source["camera_extrinsic"],
                               np.r_[axis0, pivot0, theta0])
            if not np.isclose(fitted["rmse_px"], record["selected_rmse_px"], atol=1e-5):
                raise RuntimeError("Recomputed joint fit differs from the saved evaluation")
            modeled = modeled_points("revolute", fitted["params"], xyz, parent_poses)
            fit_pixels = np.asarray([project(points, source["camera_intrinsic"],
                                             source["camera_extrinsic"]) for points in modeled])
            gt_axis = unit(source["joint_axes_world"][0, current_pos])
            records[approach] = record
            case[approach] = dict(pred=image, truth=truth, valid=valid_for_pair,
                                  child_mask=child_mask, parent_mask=parent_mask,
                                  axis=unit(fitted["params"][:3]), fit_pixels=fit_pixels,
                                  parent_poses=parent_poses, indices=indices)
        if current_pos == joint_pos:
            break
    return source, records, case, gt_axis, parent_node, joint_pos


def relative_error(case):
    """Child point displacement relative to the parent 2D centroid, from state 1."""
    pred, truth = case["pred"], case["truth"]
    child, parent = case["child_mask"], case["parent_mask"]
    pred_rel = pred[:, child] - pred[:, parent].mean(axis=1)[:, None]
    truth_rel = truth[:, child] - truth[:, parent].mean(axis=1)[:, None]
    pred_motion = pred_rel[STATES] - pred_rel[1]
    truth_motion = truth_rel[STATES] - truth_rel[1]
    return pred_motion, truth_motion, np.linalg.norm(pred_motion-truth_motion, axis=-1)


def parent_rotation_errors(source, parent_node, parent_poses):
    with np.load(source["path"], allow_pickle=False) as archive:
        transforms = archive["link_transforms_world"]
    reference_inv = np.linalg.inv(transforms[0, parent_node])
    degrees = []
    for step, state in enumerate(STATES):
        gt = transforms[state, parent_node] @ reference_inv
        est = parent_poses[step][0]
        cosine = np.clip((np.trace(est.T @ gt[:3, :3])-1)/2, -1, 1)
        degrees.append(float(np.degrees(np.arccos(cosine))))
    return degrees


def plot_tracks(source, case, records, gt_axis, out, scene_index, joint_index, mode):
    colors = dict(truth="#00e5ff", pred="#ffcc00", fitted="#ff4dd8")
    all_xy = np.concatenate([entry["truth"][STATES][:, entry["child_mask"]].reshape(-1, 2)
                             for entry in case.values()] +
                            [entry["pred"][STATES].reshape(-1, 2) for entry in case.values()])
    xlo, ylo = np.maximum(all_xy.min(axis=0)-35, 0)
    xhi, yhi = np.minimum(all_xy.max(axis=0)+35,
                         np.array([source["rgb"].shape[2], source["rgb"].shape[1]]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    motion_limits = []
    for entry in case.values():
        pred_motion, truth_motion, _ = relative_error(entry)
        motion_limits.extend([pred_motion.reshape(-1, 2), truth_motion.reshape(-1, 2)])
    motion_points = np.concatenate(motion_limits)
    motion_lo = motion_points.min(axis=0)-8
    motion_hi = motion_points.max(axis=0)+8
    for col, (approach, title) in enumerate((("full_scene", "Full-scene prediction"),
                                              ("parent_child", "Parent-child prediction"))):
        entry = case[approach]
        ax = axes[0, col]
        ax.imshow(source["rgb"][4])
        child = entry["child_mask"]
        for point in range(int(child.sum())):
            gt_path = entry["truth"][STATES][:, child][:, point]
            pred_path = entry["pred"][STATES][:, child][:, point]
            fit_path = entry["fit_pixels"][:, point]
            ax.plot(*gt_path.T, color=colors["truth"], lw=1.5, alpha=.8)
            ax.plot(*pred_path.T, color=colors["pred"], lw=1.5, alpha=.9)
            ax.plot(*fit_path.T, color=colors["fitted"], lw=1, alpha=.8, linestyle="--")
            ax.plot(*gt_path[-1], marker="o", ms=5, color=colors["truth"])
            ax.plot(*pred_path[-1], marker="x", ms=6, mew=1.7, color=colors["pred"])
            ax.text(*(gt_path[-1]+[2, -3]), str(point), color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=.6, pad=.5))
        parent_gt = entry["truth"][STATES][:, entry["parent_mask"]].mean(axis=1)
        parent_pred = entry["pred"][STATES][:, entry["parent_mask"]].mean(axis=1)
        ax.plot(*parent_gt.T, color="white", lw=2.5, label="True parent centroid")
        ax.plot(*parent_pred.T, color="#ff6333", lw=2.5, label="Predicted parent centroid")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(yhi, ylo)
        ax.set_title(f"{title}\nChild ADE {records[approach]['child_ade_px']:.1f} px | "
                     f"axis error {records[approach]['axis_error_deg']:.1f}°")
        ax.set_xlabel("Image x (pixels)")
        ax.set_ylabel("Image y (pixels)")
        ax.grid(alpha=.15)

        ax = axes[1, col]
        pred_motion, truth_motion, rel_err = relative_error(entry)
        for point in range(pred_motion.shape[1]):
            ax.plot(*truth_motion[:, point].T, color=colors["truth"], lw=1.5, alpha=.75)
            ax.plot(*pred_motion[:, point].T, color=colors["pred"], lw=1.5, alpha=.85)
            ax.plot(*truth_motion[-1, point], marker="o", ms=4, color=colors["truth"])
            ax.plot(*pred_motion[-1, point], marker="x", ms=5, mew=1.5, color=colors["pred"])
        ax.axhline(0, color="gray", lw=.7)
        ax.axvline(0, color="gray", lw=.7)
        ax.set_xlim(motion_lo[0], motion_hi[0])
        ax.set_ylim(motion_lo[1], motion_hi[1])
        ax.set_aspect("auto")
        ax.set_title(f"Child motion relative to 2D parent centroid (states 2–4)\n"
                     f"Mean relative-motion error {rel_err.mean():.1f} px")
        ax.set_xlabel("Relative x displacement from observed state 1 (px)")
        ax.set_ylabel("Relative y displacement from observed state 1 (px)")
        ax.grid(alpha=.2)
    axes[0, 0].legend([plt.Line2D([0], [0], color=colors["truth"]),
                plt.Line2D([0], [0], color=colors["pred"]),
                plt.Line2D([0], [0], color=colors["fitted"], linestyle="--")],
               ["Ground truth", "MYRIAD prediction", "Fitted revolute model"],
               loc="lower left", fontsize=9, facecolor="black", labelcolor="white")
    fig.suptitle(f"Scene {scene_index} · joint {joint_index} · {mode.replace('_', ' ')}"
                 " · same child keypoints", fontsize=17)
    fig.savefig(out / "tracks_and_relative_motion.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    left = fig.add_subplot(121)
    right = fig.add_subplot(122, projection="3d")
    for i, (name, label) in enumerate((("full_scene", "Full scene"),
                                        ("parent_child", "Parent-child"))):
        entry = case[name]
        valid = entry["valid"][STATES][:, entry["child_mask"]]
        child_error = np.linalg.norm(entry["pred"][STATES][:, entry["child_mask"]]
                                     - entry["truth"][STATES][:, entry["child_mask"]], axis=-1)
        per_state = [child_error[step, valid[step]].mean() for step in range(len(STATES))]
        left.bar(STATES + (i-.5)*.34, per_state, width=.34, label=label)
    left.set_xticks(STATES)
    left.set_xlabel("Forecast state")
    left.set_ylabel("Mean child-point error (source-image px)")
    left.set_title("Absolute trajectory error by state")
    left.grid(axis="y", alpha=.2)
    left.legend()
    for name, vector, color in (("True", gt_axis, "#128678"),
                                ("Full scene", case["full_scene"]["axis"], "#e89c00"),
                                ("Parent-child", case["parent_child"]["axis"], "#bb338b")):
        vector = vector if vector @ gt_axis >= 0 else -vector
        right.quiver(0, 0, 0, *vector, color=color, linewidth=3, arrow_length_ratio=.12,
                     label=name)
    right.set_xlim(-1, 1); right.set_ylim(-1, 1); right.set_zlim(-1, 1)
    right.set_xlabel("World X"); right.set_ylabel("World Y"); right.set_zlabel("World Z")
    right.set_title("Axis directions (sign aligned with ground truth)")
    right.legend(loc="upper left")
    fig.savefig(out / "ade_and_3d_axes.png", dpi=180)
    plt.close(fig)


def parent_swap_diagnostic(source, case, gt_axis, parent_node, out):
    """Hold child tracks fixed while changing the parent motion supplied to the fitter."""
    with np.load(source["path"], allow_pickle=False) as archive:
        transforms = archive["link_transforms_world"]
    reference_inv = np.linalg.inv(transforms[0, parent_node])
    true_parent = []
    for state in STATES:
        pose = transforms[state, parent_node] @ reference_inv
        true_parent.append((pose[:3, :3], pose[:3, 3], True))
    parent_options = {"Full scene": case["full_scene"]["parent_poses"],
                      "Parent-child": case["parent_child"]["parent_poses"],
                      "Ground truth": true_parent}
    row_names = ("Full scene", "Parent-child")
    column_names = tuple(parent_options)
    matrix = np.empty((2, 3))
    report = {}
    for row, child_name in enumerate(row_names):
        entry = case["full_scene" if row == 0 else "parent_child"]
        child = entry["child_mask"]
        xyz = source["world"][entry["indices"]][child]
        uv = entry["pred"][STATES][:, child]
        mask = entry["valid"][STATES][:, child]
        k, e = source["camera_intrinsic"], source["camera_extrinsic"]
        child_init = [estimate_pose(xyz, uv[step], mask[step], k, e)
                      for step in range(len(STATES))]
        report[child_name] = {}
        for col, parent_name in enumerate(column_names):
            parents = parent_options[parent_name]
            axis0, pivot0, theta0 = initial_axis(child_init, parents, xyz.mean(0))
            fitted = fit_model("revolute", xyz, uv, mask, parents, k, e,
                               np.r_[axis0, pivot0, theta0])
            error = axis_error(fitted["params"][:3], gt_axis)
            matrix[row, col] = error
            report[child_name][parent_name] = {"axis_error_deg": error,
                                               "fit_rmse_px": fitted["rmse_px"]}
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    picture = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=35)
    fig.colorbar(picture, ax=ax, label="Axis orientation error (degrees)")
    ax.set_xticks(range(3), column_names)
    ax.set_yticks(range(2), row_names)
    ax.set_xlabel("Parent motion supplied to the joint fitter")
    ax.set_ylabel("Predicted child trajectory")
    ax.set_title("Swap one input at a time: 3D axis from the same child tracks")
    for row in range(2):
        for col in range(3):
            item = report[row_names[row]][column_names[col]]
            ax.text(col, row, f"{item['axis_error_deg']:.1f}°\n"
                    f"fit {item['fit_rmse_px']:.1f} px", ha="center", va="center",
                    color="black", fontsize=11)
    fig.savefig(out / "parent_motion_swap.png", dpi=180)
    plt.close(fig)
    return report


def animate(source, case, records, out):
    frames = []
    approaches = ("full_scene", "parent_child")
    for state in range(5):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for col, (ax, approach) in enumerate(zip(axes[1:], approaches)):
            entry = case[approach]
            ax.imshow(source["rgb"][state])
            for part, marker, color in (("parent_mask", "o", "#ff6333"),
                                        ("child_mask", "x", "#ffcc00")):
                points = entry["pred"][state, entry[part]]
                ax.scatter(points[:, 0], points[:, 1], marker=marker, c=color, s=35,
                           linewidths=1.7)
            ax.set_title(f"{'Full scene' if col == 0 else 'Parent-child'}\n"
                         f"ADE {records[approach]['child_ade_px']:.1f} px · "
                         f"axis error {records[approach]['axis_error_deg']:.1f}°")
        axes[0].imshow(source["rgb"][state])
        entry = case["full_scene"]
        for part, marker, color in (("parent_mask", "o", "#ff6333"),
                                    ("child_mask", "x", "#00e5ff")):
            points = entry["truth"][state, entry[part]]
            axes[0].scatter(points[:, 0], points[:, 1], marker=marker, c=color, s=35,
                            linewidths=1.7)
        axes[0].set_title("Ground truth")
        for ax in axes:
            ax.set_xlim(80, 430); ax.set_ylim(440, 80)
            ax.axis("off")
        fig.suptitle(f"Scene 33 · joint 1 · with poke · state {state}"
                     + (" (observed)" if state < 2 else " (forecast)"))
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(image).resize((1200, 420)))
        plt.close(fig)
    frames[0].save(out / "ground_truth_full_pair.gif", save_all=True,
                   append_images=frames[1:], duration=900, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-index", type=int, default=33)
    parser.add_argument("--joint-index", type=int, default=1)
    parser.add_argument("--mode", choices=("with_poke", "without_poke"), default="with_poke")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "finetuning/runs/pairwise_joint_test1")
    args = parser.parse_args()
    source, records, case, gt_axis, parent_node, joint_pos = load_case(
        args.scene_index, args.joint_index, args.mode, args.run_dir)
    out = args.run_dir / "diagnostics" / f"scene_{args.scene_index:04d}_joint_{args.joint_index}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    plot_tracks(source, case, records, gt_axis, out,
                args.scene_index, args.joint_index, args.mode)
    swaps = parent_swap_diagnostic(source, case, gt_axis, parent_node, out)
    if (args.scene_index, args.joint_index, args.mode) == (33, 1, "with_poke"):
        animate(source, case, records, out)
    report = {"scene_index": args.scene_index, "joint_index": args.joint_index,
              "mode": args.mode, "ground_truth_axis_world": gt_axis.tolist(),
              "parent_motion_swap": swaps,
              "methods": {name: {"child_ade_px": records[name]["child_ade_px"],
                                 "parent_ade_px": records[name]["parent_ade_px"],
                                 "axis_error_deg": axis_error(case[name]["axis"], gt_axis),
                                 "fit_rmse_px": records[name]["selected_rmse_px"],
                                 "estimated_axis_world": case[name]["axis"].tolist(),
                                 "parent_rotation_error_deg_by_state": parent_rotation_errors(
                                     source, parent_node, case[name]["parent_poses"]),
                                 "relative_motion_error_px": float(relative_error(case[name])[2].mean())}
                          for name in case}}
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved visualizations in {out}")


if __name__ == "__main__":
    main()
