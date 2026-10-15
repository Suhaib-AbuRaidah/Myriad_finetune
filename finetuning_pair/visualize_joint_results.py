"""Render saved first-50 evaluation results without a model, GPU, or source data.

Example: python -m finetuning_pair.visualize_joint_results --scene-index 0 --joint-index 0
Joint indices are the zero-based IDs stored by the dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from finetuning_pair.paths import artifact_path


def render_case(run_dir, pair_index, mode, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from finetuning.joint_parame_estim.run import project, unit

    out = artifact_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{pair_index:04d}_{mode}"
    record = json.loads((run_dir / "records" / f"{stem}.json").read_text())
    reference_path = run_dir / "geometry_reference.json"
    reference = json.loads(reference_path.read_text()).get(str(pair_index)) if reference_path.exists() else None
    with np.load(run_dir / "predictions" / f"{stem}.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    size = saved["image"].shape[-1]
    image = np.clip((saved["image"].transpose(1, 2, 0)+1)*127.5, 0, 255).astype(np.uint8)
    pred, truth, valid = saved["predictions"]*(size-1), saved["ground_truth"]*(size-1), saved["valid"]
    colors = plt.get_cmap("tab10")(np.arange(truth.shape[1]) % 10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes[:2]:
        ax.imshow(image)
        ax.set(xlim=(0, size-1), ylim=(size-1, 0))
        ax.axis("off")
    for point, color in enumerate(colors):
        t = np.where(valid[:, point, None], truth[:, point], np.nan)
        axes[0].plot(t[:,0], t[:,1], "o--", color=color, ms=3, alpha=.9)
        for hypothesis in range(len(pred)):
            axes[0].plot(pred[hypothesis,:,point,0], pred[hypothesis,:,point,1], color=color,
                         alpha=.8 if hypothesis == 0 else .18, lw=1.8 if hypothesis == 0 else .7)
        axes[1].plot(t[:,0], t[:,1], "o--", color="cyan", alpha=.8, ms=3)
        axes[1].plot(pred[0,:,point,0], pred[0,:,point,1], "o-", color="orangered", alpha=.8, ms=3)
    axes[0].set_title("All 5 sampled tracks; dashed = GT")
    axes[1].set_title(f"Hypothesis 0 (orange), GT (cyan)\nADE={record['ADE_1_pixels']:.2f} px; FDE={record['FDE_1_pixels']:.2f} px")
    ax = axes[2]
    ax.imshow(saved["initial_rgb"])
    k, e = saved["camera_intrinsic"], saved["camera_extrinsic"]
    centroid = saved["reference_world"].mean(0)
    length = max(.1, float(np.linalg.norm(np.ptp(saved["reference_world"], axis=0))))
    for name, axis, pivot, color in (
        ("Stored ground truth", record["gt_axis_world"], record["gt_pivot_world"], "cyan"),
        ("Transform ground truth", reference["kinematic_axis_world"] if reference else None,
         reference["kinematic_pivot_world"] if reference else None, "limegreen"),
        ("Predicted", record.get("estimated_axis_world"), record.get("estimated_pivot_world"), "orangered"),
    ):
        if "ground truth" in name and record["gt_type"] == "fixed":
            continue
        if axis is None or pivot is None:
            continue
        axis, pivot = unit(np.asarray(axis)), np.asarray(pivot)
        center = pivot + axis * ((centroid-pivot) @ axis)
        uv = project(np.array([center-length*axis, center+length*axis]), k, e)
        ax.plot(uv[:,0], uv[:,1], color=color, lw=2, label=name)
        origin_uv = project(pivot[None], k, e)[0]
        ax.plot(*origin_uv, "x", color=color, ms=8)
    h, w = saved["initial_rgb"].shape[:2]
    ax.set(xlim=(0,w-1), ylim=(h-1,0))
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    axis_err, pivot_err = record.get("axis_error_deg"), record.get("pivot_distance")
    annotation = f"Axis={axis_err:.2f}°; pivot distance={pivot_err:.4g}" if axis_err is not None else "Axis/pivot metric N/A for selected type"
    ax.set_title(f"GT {record['gt_type']} → predicted {record.get('predicted_type')}\n{annotation}")
    fig.suptitle(f"Scene {record['scene_index']} | joint index {record['joint_index']} | {mode} | frozen-parent motion")
    fig.tight_layout()
    fig.savefig(out / "trajectories_and_axis.png", dpi=160)
    plt.close(fig)
    Image.fromarray(image).save(out / "model_input.png")
    # All animation backgrounds use the same conditioned initial image.
    frames = []
    for state in range(len(truth)):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(image)
        for point, color in enumerate(colors):
            t = np.where(valid[:state+1, point, None], truth[:state+1, point], np.nan)
            ax.plot(t[:,0], t[:,1], "o--", color="cyan", ms=4, lw=1)
            ax.plot(pred[0,:state+1,point,0], pred[0,:state+1,point,1], "o-", color="orangered", ms=4, lw=1)
        ax.set(xlim=(0,size-1),ylim=(size-1,0),title=f"State {state} | orange prediction; cyan GT")
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        plt.close(fig)
    frames[0].save(out / "relative_motion.gif", save_all=True, append_images=frames[1:], duration=650, loop=0)
    (out / "metrics.json").write_text(json.dumps(record, indent=2)+"\n")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("finetuning_pair/runs/test1_first50_analysis"))
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--joint-index", type=int, default=0)
    p.add_argument("--mode", default="without_poke", choices=("without_poke","with_poke"))
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    run = artifact_path(args.run_dir)
    manifest = json.loads((run / "run.json").read_text())
    matches = [i for i, (s, pair) in enumerate(manifest["pairs"]) if s == args.scene_index and pair[2] == args.joint_index]
    if len(matches) != 1:
        raise ValueError("Requested scene/joint has no eligible pair; see run.json inventory")
    out = args.output_dir or run / "visualizations" / f"scene_{args.scene_index:03d}_joint_{args.joint_index}_{args.mode}"
    print(render_case(run, matches[0], args.mode, out))


if __name__ == "__main__":
    main()
