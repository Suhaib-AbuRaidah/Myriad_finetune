"""Summarize trajectories and produce a per-example visualization catalog."""
import argparse
import json
from pathlib import Path
from training_from_scratch.full_evaluation.metrics import distribution
from training_from_scratch_cropped.config import write_json
from .common import DEFAULT_OUT

METRICS = ("pixel_ADE", "source_RMSE_pixels", "source_FDE_pixels", "crop_ADE_pixels",
    "trajectory_ADE", "paper_style_MSE", "paper_MSE_crop", "GT_path_normalized_error_percent",
    "future_outside_crop_fraction")


def summarize(rows):
    scoreable = [row for row in rows if row.get("scoreable")]
    return dict(pairs=len(rows), predicted=sum(row["status"]=="predicted" for row in rows),
        scoreable=len(scoreable), unavailable=[row for row in rows if row["status"]!="predicted"],
        metrics={metric:distribution(row["trajectory_metrics"]["primary"].get(metric) for row in scoreable) for metric in METRICS},
        oracle_ADE_source_pixels=distribution(row["trajectory_metrics"]["oracle_ADE_source_pixels"] for row in scoreable))


def generate_report(output):
    output = Path(output); manifest = json.loads((output/"run.json").read_text())
    result = {}; catalog = []
    for split in ("test1", "val"):
        rows = [json.loads(path.read_text()) for path in sorted((output/split/"records").glob("*.json"))]
        expected = len(manifest["splits"][split]["pairs"])
        assert len(rows)==expected and [row["pair_index"] for row in rows]==list(range(expected))
        for row in rows:
            if row["status"]=="predicted": assert (output/split/row["prediction_file"]).exists()
            catalog.append(dict(split=split, pair_index=row["pair_index"], original_pair_index=row["original_pair_index"],
                scene_index=row["scene_index"], original_scene_index=row["original_scene_index"],
                sample_id=row["sample_id"], object_id=row["object_id"], joint_ordinal=row["joint_ordinal"],
                gt_type=row["gt_type"], status=row["status"], scoreable=row.get("scoreable", False),
                prediction_file=str(Path(split)/row["prediction_file"]),
                source_archive=str(Path(split)/row["source_archive"]),
                record_file=str(Path(split)/"records"/f"pair_{row['pair_index']:05d}.json"),
                crop_bounds=row.get("crop_bounds"), primary_metrics=row.get("trajectory_metrics", {}).get("primary")))
        ids = sorted({row["object_id"] for row in rows})
        summary = summarize(rows)
        summary.update(scenes=len(manifest["splits"][split]["scenes"]), objects=len(ids),
            source_joints=len(manifest["splits"][split]["inventory"]),
            source_exclusions=[r for r in manifest["splits"][split]["inventory"] if not r["eligible"]],
            per_joint={str(joint):summarize([row for row in rows if row["joint_ordinal"]==joint])
                for joint in sorted({row["joint_ordinal"] for row in rows})},
            per_object={obj:summarize([row for row in rows if row["object_id"]==obj]) for obj in ids},
            per_type={kind:summarize([row for row in rows if row["gt_type"]==kind]) for kind in sorted({row["gt_type"] for row in rows})})
        result[split] = summary
    write_json(output/"summary.json", result)
    write_json(output/"visualization_catalog.json", dict(schema_version=manifest["schema_version"],
        checkpoint_step=manifest["checkpoint_step"], checkpoint_sha256=manifest["checkpoint_sha256"],
        source_archive_paths_in_pair_arrays="relative to the split directory",
        examples=catalog, relative_trajectory_background="Use initial RGB/conditioned crop for frozen-parent relative trajectories; rendered future RGB belongs to actual articulated motion"))
    lines = ["# Cropped-model inference archive", "",
        f"Frozen checkpoint: step **{manifest['checkpoint_step']}**, SHA256 `{manifest['checkpoint_sha256']}`.", "",
        "Initial-only inference, FP32 motion model, frozen DINO, 16 diffusion steps and five hypotheses per pair. "
        "Hypothesis zero is primary; all five are saved. No joint fitting or fitting improvement loop was run.", "",
        "| Split | Scenes | Objects | Source joints | Eligible pairs | Predicted | Scoreable | Mean ADE source px | Median ADE source px | Mean ADE / GT path % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, summary in result.items():
        metric = summary["metrics"]
        lines.append(f"| {split} | {summary['scenes']} | {summary['objects']} | {summary['source_joints']} | {summary['pairs']} | "
            f"{summary['predicted']} | {summary['scoreable']} | {metric['pixel_ADE']['mean']:.3f} | "
            f"{metric['pixel_ADE']['median']:.3f} | {metric['GT_path_normalized_error_percent']['mean']:.3f} |")
    lines += ["", "The validation subset spans all 12 validation objects, with six or seven scenes per object, "
        "chosen without consulting inference scores. Every initially eligible joint in each selected scene is included. "
        "Joint exclusions due to initially invisible/invalid-depth children are recorded in `run.json` and `summary.json`. "
        "Predictions with no scoreable future targets are retained and explicitly excluded only from metric denominators.", "",
        "## Per-joint trajectory results", "",
        "| Split | Joint ordinal | Pairs | Mean source ADE px | Median source ADE px | Mean source FDE px | Mean ADE / GT path % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for split, summary in result.items():
        for joint, group in summary["per_joint"].items():
            m = group["metrics"]
            def fmt(value): return "undefined" if value is None else f"{value:.3f}"
            lines.append(f"| {split} | {joint} | {group['pairs']} | {fmt(m['pixel_ADE']['mean'])} | {fmt(m['pixel_ADE']['median'])} | "
                f"{fmt(m['source_FDE_pixels']['mean'])} | {fmt(m['GT_path_normalized_error_percent']['mean'])} |")
    lines += ["", "## Saved for each example", "",
        "- `split/predictions/pair_NNNNN.npz`: five predicted trajectories in crop-normalized, crop-pixel and original-image "
        "pixel coordinates; GT relative and actual trajectories; all validity masks; point identities and link labels; "
        "initial world anchors and local 3D points; relative GT camera points; model input image; clean cropped RGB; "
        "parent/child masks; crop bounds, transform and inverse; original/model intrinsics and extrinsics; state times, "
        "fractions, seed and target convention.",
        "- `split/records/pair_NNNNN.json`: stable object/scene/joint identifiers, original dataset indices, "
        "all per-hypothesis trajectory metrics, per-state ADE, cache reference and source archive checksum.",
        "- `split/scenes/SAMPLE.npz`: the complete source scene, including all five RGB/depth/segmentation frames, "
        "mesh points/normals/labels, GT link transforms and joint parameters. Local hard links preserve these files "
        "without duplicating their data; checksums detect modifications.",
        "- `visualization_catalog.json`: one searchable entry per joint example, with paths, crop information and "
        "primary metrics. This supports later presentation visualizations without model inference.",
        "- `checkpoint.pt`, `run.json`, `image_inventory.json`, `inference_code.json`, `code/`, `summary.json`, "
        "and `audit.json`: frozen weights, split selection, feature/crop references, implementation snapshot, summaries "
        "and verification evidence.", "",
        "## Coordinate and metric conventions", "",
        "The crop is selected from the initial parent–child mask union, with the checkpoint's padding and pixel-center "
        "resize convention. Future trajectories remain unclipped even when outside the crop. Source validity, "
        "crop validity and actual-motion validity are saved separately.", "",
        "Source-pixel ADE is the mean Euclidean error over valid states 1–4 and keypoints, then averaged equally "
        "across scoreable pairs. Path-normalized error averages per-track ADE divided by full GT path length, "
        "using complete moving tracks of at least one source pixel; fixed-joint percentages are undefined. "
        "Crop-paper MSE and full-frame paper-style MSE have separate names and denominators. "
        "No metric selects the primary hypothesis.", "",
        "Frozen-parent relative tracks must be displayed on the initial image or its crop. The actual rendered "
        "future frames include parent motion and are a different visualization coordinate convention; actual GT "
        "tracks are saved for those frames.", "",
        "Joint parameters will be evaluated later with the frozen `candidate.py`, as requested. Any future fitting "
        "development should use validation predictions and preserve test1 as the subsequent evaluation split."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(event="archive_summary", splits={name:{key:s[key] for key in
        ("scenes", "pairs", "predicted", "scoreable")} for name,s in result.items()})), flush=True)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUT)
    generate_report(parser.parse_args().output)
