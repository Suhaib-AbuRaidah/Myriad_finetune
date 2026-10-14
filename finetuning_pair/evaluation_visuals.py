"""Regenerate presentation figures exclusively from persisted scene results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from finetuning_pair.evaluate import (MODES, aggregate_records, common_horizon_comparison,
                                 file_hash, read_records, select_cases, write_csv, write_json)


GT = (0, 143, 158)
PRED = (227, 101, 33)
INK = (24, 39, 54)
BACKGROUND = (248, 250, 252)
LABELS = {"with_poke": "With poke", "without_poke": "Without poke"}


def font(size: int, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def number(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def object_label(object_id: str) -> str:
    parts = object_id.rsplit("_", 1)
    if len(parts) == 2:
        return f"{parts[0].removeprefix('rec_').replace('_', ' ')} ({parts[1][:8]})"
    return object_id or "unspecified"


def trajectory_panel(arrays: dict, state: int, kind: str, side: int) -> Image.Image:
    panel = Image.fromarray(arrays["rgb_frames"][state]).resize((side, side), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    valid = arrays["valid"]
    paths = []
    if kind in ("ground_truth", "overlay"):
        paths.append((arrays["ground_truth"], GT, False))
    if kind in ("prediction", "overlay", "oracle"):
        sample = int(np.argmin(arrays["hypothesis_errors"][:5])) if kind == "oracle" else 0
        paths.append((arrays["predictions"][sample], PRED, True))
    for positions, color, is_prediction in paths:
        for point in range(positions.shape[1]):
            previous = None
            for t in range(state + 1):
                x, y = positions[t, point]
                inside = bool(valid[t, point]) and np.isfinite([x, y]).all() and 0 <= x <= 1 and 0 <= y <= 1
                xy = (round(float(x) * (side-1)), round(float(y) * (side-1))) if inside else None
                if previous is not None and xy is not None:
                    draw.line((previous, xy), fill=(*color, 175), width=max(2, side // 200))
                previous = xy
            if previous is not None:
                radius = max(3, side // 120)
                x, y = previous
                draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                             fill=(*color, 245), outline=(255, 255, 255, 245), width=1)
            elif is_prediction and valid[state, point]:
                position = positions[state, point].astype(np.float64)
                if np.isfinite(position).all() and ((position < 0) | (position > 1)).any():
                    tip = np.clip(position, 0, 1) * (side-1)
                    direction = position - .5
                    direction /= np.linalg.norm(direction)
                    normal = np.array([-direction[1], direction[0]])
                    base = tip - direction * 14
                    triangle = [tuple(tip), tuple(base + normal * 7), tuple(base - normal * 7)]
                    draw.polygon(triangle, fill=(*color, 245), outline=(255, 255, 255, 245))
    return Image.alpha_composite(panel, layer).convert("RGB")


def title_lines(record: dict, rank: int, category: str):
    name = "Best case" if category == "top_10" else "Failure case"
    return (
        f"{LABELS[record['mode']]} | {name} {rank:02d} | Pair {record.get('pair_labels')} | Example {record['index']:03d}",
        f"minADE_1 {record['minADE_1']:.6f} | ADE {number(record['ADE_1_pixels'])} px | "
        f"Final error {number(record['FDE_1_pixels'])} px | {object_label(record['object_id'])}",
    )


def render_case(arrays: dict, record: dict, output: Path, rank: int, category: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    states = arrays["ground_truth"].shape[0]
    known = int(arrays["known_states"])
    first, second = title_lines(record, rank, category)
    side, top, label_height, footer = 448, 112, 40, 100
    rows = (("Ground truth", "ground_truth"), ("Prediction 1", "prediction"), ("Comparison", "overlay"))
    sheet = Image.new("RGB", (side * states, top + len(rows) * (side + label_height) + footer), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), first, font=font(30, True), fill=INK)
    draw.text((20, 58), second, font=font(23), fill=INK)
    for row, (label, kind) in enumerate(rows):
        y = top + row * (side + label_height)
        for state in range(states):
            status = "observed" if state < known else "predicted"
            header = f"{label} | State {state}" + (f" ({status})" if kind != "ground_truth" else "")
            draw.text((state * side + 12, y + 8), header, font=font(20, True), fill=INK)
            sheet.paste(trajectory_panel(arrays, state, kind, side), (state * side, y + label_height))
    draw.text((20, sheet.height - 88), "TEAL: ground truth   |   ORANGE: prediction 1   |   Ranking: minADE_1 (lower is better)",
              font=font(23), fill=INK)
    draw.text((20, sheet.height - 57), "Rendered state images are comparison backgrounds; the model receives only the reference image.",
              font=font(19), fill=INK)
    draw.text((20, sheet.height - 28),
              f"Orange border arrows mark out-of-frame predictions. Out-of-frame scored positions: {100*record['out_of_frame_fraction_1']:.1f}%.",
              font=font(19), fill=INK)
    sheet.save(output / "overview.png", dpi=(200, 200))

    final_side = 640
    final = Image.new("RGB", (2 * final_side, final_side + 156), BACKGROUND)
    final_draw = ImageDraw.Draw(final)
    final_draw.text((18, 12), first, font=font(27, True), fill=INK)
    final_draw.text((18, 54), second, font=font(18), fill=INK)
    for column, (label, kind) in enumerate((("Ground truth", "ground_truth"), ("Ground truth + prediction 1", "overlay"))):
        final_draw.text((column * final_side + 18, 91), label, font=font(23, True), fill=INK)
        final.paste(trajectory_panel(arrays, states-1, kind, final_side), (column * final_side, 126))
    final_draw.text((18, 774), "TEAL: ground truth | ORANGE: prediction 1 | Border arrows mark out-of-frame predictions.",
                    font=font(16), fill=INK)
    final.save(output / "final_state.png", dpi=(200, 200))

    # Oracle is a supplementary export and does not determine selection.
    oracle = trajectory_panel(arrays, states-1, "oracle", 640)
    oracle_canvas = Image.new("RGB", (640, 734), BACKGROUND)
    oracle_canvas.paste(oracle, (0, 94))
    oracle_draw = ImageDraw.Draw(oracle_canvas)
    oracle_draw.text((14, 13), "Oracle best of five | Final state", font=font(23, True), fill=INK)
    oracle_draw.text((14, 52), f"Selected using ground truth | minADE_5 {record['minADE_5']:.6f}", font=font(17), fill=INK)
    oracle_canvas.save(output / "oracle_best5.png", dpi=(200, 200))

    frames = []
    for state in range(states):
        canvas = Image.new("RGB", (3 * side, 756), BACKGROUND)
        frame_draw = ImageDraw.Draw(canvas)
        frame_draw.text((18, 12), first, font=font(27, True), fill=INK)
        frame_draw.text((18, 55), second, font=font(19), fill=INK)
        for column, (label, kind) in enumerate(rows):
            frame_draw.text((column * side + 14, 108), label, font=font(24, True), fill=INK)
            canvas.paste(trajectory_panel(arrays, state, kind, side), (column * side, 150))
        status = "Observed input" if state < known else "Model prediction"
        frame_draw.text((18, 623), f"State {state} / {states-1} | {status}", font=font(26, True), fill=INK)
        frame_draw.text((18, 667), "TEAL: ground truth | ORANGE: prediction 1 | Border arrows mark out-of-frame predictions.", font=font(20), fill=INK)
        frame_draw.text((18, 704), "Rendered images are comparison backgrounds. States follow motion progress.", font=font(18), fill=INK)
        frames.append(canvas)
    gif_frames = [frame.resize((1008, 567), Image.Resampling.LANCZOS) for frame in frames]
    gif_frames[0].save(output / "rollout.gif", save_all=True, append_images=gif_frames[1:],
                       duration=1000, loop=0, optimize=True)
    clean_frames = []
    for frame in gif_frames:
        clean = frame.copy()
        clean_draw = ImageDraw.Draw(clean)
        clean_draw.rectangle((0, 0, 1007, 74), fill=BACKGROUND)
        clean_draw.rectangle((0, 450, 1007, 566), fill=BACKGROUND)
        clean_frames.append(clean)
    for filename, variants in (
        ("rollout_headers_only.gif", clean_frames),
        ("rollout_headers_only_no_comparison.gif",
         [frame.crop((0, 0, 672, 567)) for frame in clean_frames]),
    ):
        variants[0].save(output / filename, save_all=True, append_images=variants[1:],
                         duration=1000, loop=0, disposal=2, optimize=False)
    executable = Path(sys.executable).parent / "ffmpeg"
    ffmpeg = str(executable) if executable.is_file() else shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is required for MP4 presentation exports")
    raw = b"".join(frame.tobytes() * 2 for frame in frames)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", "1344x756", "-framerate", "2", "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-threads", "2", "-movflags", "+faststart", str(output / "rollout.mp4")],
                   input=raw, capture_output=True, check=True)


def save_contact_sheet(cases: list[dict], folder: Path, category: str, mode: str) -> None:
    if not cases:
        return
    width, card_height, columns = 440, 326, 5
    canvas = Image.new("RGB", (columns * width, 86 + math_ceil(len(cases), columns) * card_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    description = "Lowest minADE_1" if category == "top_10" else "Highest minADE_1"
    draw.text((20, 18), f"{LABELS[mode]} | {description} | Prediction 1", font=font(28, True), fill=INK)
    for i, record in enumerate(cases):
        path = folder / f"rank_{i+1:02d}_scene_{record['index']:04d}" / "final_state.png"
        image = Image.open(path).convert("RGB")
        # The right half shows the GT/prediction overlay; retain full square panel.
        image = image.crop((640, 126, 1280, 766)).resize((268, 268), Image.Resampling.LANCZOS)
        x, y = (i % columns) * width, 86 + (i // columns) * card_height
        canvas.paste(image, (x + (width-268)//2, y))
        draw.text((x+18, y+273), f"#{i+1:02d} Scene {record['index']:03d} | {record['minADE_1']:.6f}", font=font(20, True), fill=INK)
        short_object = record['object_id'].rsplit('_', 1)[-1][:8]
        draw.text((x+18, y+301), f"ADE {number(record['ADE_1_pixels'])} px | Object {short_object}", font=font(17), fill=INK)
    canvas.save(folder / "contact_sheet.png", dpi=(200, 200))


def math_ceil(n, d):
    return (n + d - 1) // d


def summary_plots(root: Path, records: dict, summaries: dict, paired: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    colors = ("#008f9e", "#e36521")
    split = json.loads((root / "run.json").read_text())["settings"]["split"]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(plots / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(plots / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(3)
    for mode, offset, color in zip(MODES, (-.18, .18), colors):
        d = summaries[mode]["distributions"]
        axes[0].bar(x + offset, [d[k]["mean"] for k in ("minADE_1", "minADE_5", "stationary_minADE")],
                    width=.34, color=color, label=LABELS[mode])
        axes[1].bar(np.arange(2)+offset, [d[k]["mean"] for k in ("ADE_1_pixels", "FDE_1_pixels")],
                    width=.34, color=color, label=LABELS[mode])
    axes[0].set_xticks(x, ("Prediction 1", "Oracle best of 5", "Stationary baseline"))
    axes[0].set_ylabel("Mean squared error on [-1, 1] coordinates")
    axes[1].set_xticks(np.arange(2), ("Average distance", "Final-state distance"))
    axes[1].set_ylabel("Mean scene error (pixels at model image size)")
    for axis in axes:
        axis.margins(y=.25)
        axis.legend(frameon=False, loc="upper left")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"Child-relative {split} evaluation | Native scoring windows", fontsize=17)
    axes[0].set_title("With poke: states 2+ | Without poke: states 1+", fontsize=11)
    axes[1].set_title("Distance metrics use prediction 1", fontsize=11)
    save(fig, "summary_metrics")
    fig, axis = plt.subplots(figsize=(10, 5))
    for mode, color in zip(MODES, colors):
        values = np.sort([r["minADE_1"] for r in records[mode] if not r["skipped"]])
        axis.step(values, np.arange(1, len(values)+1)/len(values)*100, where="post", color=color,
                  label=LABELS[mode], linewidth=2)
    axis.set_xscale("symlog", linthresh=1e-5)
    axis.set_xlabel("Scene minADE_1 (squared error; lower is better)")
    axis.set_ylabel("Percentage of evaluated scenes")
    axis.set_title(f"Distribution of first-prediction errors | Child-relative {split}")
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    save(fig, "error_distribution")
    if paired:
        fig, axis = plt.subplots(figsize=(7, 6))
        without = np.array([r["without_poke_minADE_1"] for r in paired])
        with_poke = np.array([r["with_poke_minADE_1"] for r in paired])
        axis.scatter(without, with_poke, s=25, alpha=.65, color=colors[0])
        maximum = max(float(without.max()), float(with_poke.max()), 1e-4)
        axis.plot([0, maximum], [0, maximum], "--", color="#75828f", label="Equal error")
        axis.set_xscale("symlog", linthresh=1e-5)
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.set_xlabel("Without poke: minADE_1")
        axis.set_ylabel("With poke: minADE_1")
        axis.set_title("Same future states and valid points\nBelow the diagonal: poke improves prediction 1")
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
        save(fig, "poke_comparison")
    objects = sorted({r["object_id"] for mode in MODES for r in records[mode] if not r["skipped"]})
    fig, axis = plt.subplots(figsize=(10, max(5, len(objects)*.43)))
    for mode, offset, color in zip(MODES, (-.18, .18), colors):
        means = [np.mean([r["minADE_1"] for r in records[mode] if not r["skipped"] and r["object_id"] == obj])
                 for obj in objects]
        axis.barh(np.arange(len(objects))+offset, means, height=.34, color=color, label=LABELS[mode])
    axis.set_yticks(np.arange(len(objects)), [object_label(obj) for obj in objects])
    axis.set_xlabel("Mean scene minADE_1 (native scoring windows)")
    axis.set_ylabel("Object (full IDs in scene_metrics.csv)")
    axis.set_title("First-prediction error by object")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=.2)
    save(fig, "per_object")


def export_mode(root: Path, mode: str, count: int = 10, force_render: bool = False) -> None:
    records = read_records(root, mode)
    expected = len(json.loads((root / "run.json").read_text())["dataset_files"])
    if len(records) != expected or {r["index"] for r in records} != set(range(expected)):
        raise ValueError(f"Cannot rank incomplete evaluation results for {mode}")
    selections = select_cases(records, count)
    write_json(root / mode / "summary.json", aggregate_records(records))
    write_csv(root / mode / "scene_metrics.csv", records)
    write_json(root / mode / "selection.json", selections)
    for category, cases in selections.items():
        folder = root / mode / "visualizations" / category
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "ranking.csv", cases)
        for rank, record in enumerate(cases, 1):
            output = folder / f"rank_{rank:02d}_scene_{record['index']:04d}"
            output.mkdir(exist_ok=True)
            source = root / record["results_file"]
            fingerprint = {"renderer_sha256": file_hash(Path(__file__)), "results_sha256": file_hash(source),
                           "rank": rank, "category": category}
            stamp = output / "render.json"
            required = ("overview.png", "final_state.png", "oracle_best5.png", "rollout.gif", "rollout.mp4",
                        "rollout_headers_only.gif", "rollout_headers_only_no_comparison.gif",
                        "results.npz", "metrics.json")
            cached = (stamp.exists() and json.loads(stamp.read_text()) == fingerprint
                      and all((output / name).is_file() and (output / name).stat().st_size for name in required))
            if force_render or not cached:
                shutil.copy2(source, output / "results.npz")
                write_json(output / "metrics.json", {**record, "rank": rank, "category": category})
                with np.load(source, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                render_case(arrays, record, output, rank, category)
                write_json(stamp, fingerprint)
            print(json.dumps({"event": "case_rendered", "mode": mode, "category": category,
                              "rank": rank, "index": record["index"], "cached": bool(cached and not force_render)}), flush=True)
        save_contact_sheet(cases, folder, category, mode)


def export_presentation(root: Path, count: int = 10, force_render: bool = False) -> None:
    manifest = json.loads((root / "run.json").read_text())
    records = {mode: read_records(root, mode) for mode in MODES}
    expected = len(manifest["dataset_files"])
    for mode in MODES:
        if len(records[mode]) != expected or {r["index"] for r in records[mode]} != set(range(expected)):
            raise ValueError(f"Evaluation is incomplete in {mode}; resume it before exporting")
    summaries = {mode: aggregate_records(records[mode]) for mode in MODES}
    selections = {mode: select_cases(records[mode], count) for mode in MODES}
    for mode in MODES:
        export_mode(root, mode, count, force_render)
    paired = common_horizon_comparison(root, records)
    comparison = {"scenes_compared": len(paired), "first_scored_state": 2,
                  "points": "Intersection of point indices and valid target masks in both modes"}
    if paired:
        for mode in MODES:
            for metric in ("minADE_1", "minADE_5"):
                comparison[f"{mode}_{metric}"] = float(np.mean([r[f"{mode}_{metric}"] for r in paired]))
        gains = np.array([r["poke_improvement_minADE_1"] for r in paired])
        comparison["mean_poke_improvement_minADE_1"] = float(gains.mean())
        comparison["scenes_improved"] = int((gains > 0).sum())
        comparison["scenes_worsened"] = int((gains < 0).sum())
        comparison["scenes_tied"] = int((gains == 0).sum())
        denominator = comparison["without_poke_minADE_1"]
        comparison["relative_improvement_percent"] = (100 * gains.mean() / denominator if denominator > 0 else None)
        with (root / "paired_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    write_json(root / "summary.json", {"checkpoint_step": manifest["checkpoint_step"],
                                      "settings": manifest["settings"], "modes": summaries,
                                      "common_horizon_comparison": comparison,
                                      "rankings": selections})
    summary_plots(root, records, summaries, paired)
    write_report(root, manifest, summaries, selections, comparison)
    manifest["status"] = "complete"
    manifest["presentation_completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["selected_case_count_per_category"] = count
    write_json(root / "run.json", manifest)
    print(json.dumps({"event": "complete", "output": str(root), "modes": summaries,
                      "comparison": comparison}), flush=True)


def write_report(root: Path, manifest: dict, summaries: dict, selections: dict, comparison: dict):
    settings = manifest["settings"]
    pretrained = manifest.get("checkpoint_kind") == "pretrained"
    identity = (f"Checkpoint: **{Path(manifest['source_checkpoint']).name} (original pretrained model)**. No adapters are applied and no checkpoint copy is saved."
                if pretrained else f"Checkpoint step: **{manifest['checkpoint_step']:,}**.")
    lines = [f"# Full {settings['split']} evaluation", "", identity,
             f"Both runs use the same checkpoint, dataset point-selection seed, {settings['ensemble_size']} hypotheses, and {settings['diffusion_steps']} diffusion steps.",
             f"Execution settings: `{json.dumps(manifest['settings'], sort_keys=True)}`.", "",
             f"Devices by run: `{json.dumps(manifest.get('devices_by_mode', {}), sort_keys=True)}`.", "",
             "## Aggregate metrics", "",
             "| Run | Scenes / skipped | minADE_1 | minADE_5 (oracle) | ADE_1 (px) | FDE_1 (px) | Stationary MSE | Beats stationary |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        s, d = summaries[mode], summaries[mode]["distributions"]
        lines.append(f"| {LABELS[mode]} | {s['scenes_evaluated']} / {s['scenes_skipped']} | {s['minADE_1']:.6f} | "
                     f"{s['minADE_5']:.6f} | {d['ADE_1_pixels']['mean']:.2f} | {d['FDE_1_pixels']['mean']:.2f} | "
                     f"{d['stationary_minADE']['mean']:.6f} | {100*s['beats_stationary_fraction']:.1f}% |")
    lines += ["", "Each headline value is the equally weighted mean of scene scores. Full distributions, including median/p90/p95, are in `summary.json`.",
              "The project's minADE is mean **squared Euclidean error in [-1,1] coordinates**; it is not a distance in pixels.",
              "Pixel ADE is mean Euclidean distance; pixel FDE is final-state mean Euclidean distance at the checkpoint's model image resolution.",
              "minADE_1 uses prediction index 0. minADE_5 selects one complete hypothesis using ground truth, never a different hypothesis per point/state.",
              "With five hypotheses, minADE_N equals minADE_5. The stationary baseline repeats the last observed position.",
              "Native windows: with poke scores states 2 onward; without poke scores states 1 onward. Invalid/padded targets are excluded; occluded but geometrically valid targets are included.", "",
              "## Error distribution and coverage", "",
              "| Run | Median minADE_1 | P90 minADE_1 | P95 minADE_1 | Mean RMSE_1 (px) | Conditioned points | Scored point-states | Evaluation time (s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        s, d = summaries[mode], summaries[mode]["distributions"]
        lines.append(f"| {LABELS[mode]} | {d['minADE_1']['median']:.6f} | {d['minADE_1']['p90']:.6f} | "
                     f"{d['minADE_1']['p95']:.6f} | {d['RMSE_1_pixels']['mean']:.2f} | {s['points_conditioned']} | "
                     f"{s['scored_point_states']} | {s['evaluation_seconds']:.1f} |")
    lines += ["", "Evaluation time sums per-scene dataset preparation and sampling time, including warm-up. Checkpoint loading, archive persistence, and visualization export are excluded.",
              f"Each scored scene has {settings['ensemble_size']} saved hypotheses. Native scoring and common-horizon scoring are reported separately.", "",
              "## Comparison on identical future states and points", "",
              "This comparison uses states 2 onward, the intersection of sampled point IDs, and the same valid target mask for both runs."]
    if comparison["scenes_compared"]:
        improvement = comparison["relative_improvement_percent"]
        direction = "reduces" if improvement is not None and improvement >= 0 else "increases"
        lines += [f"Scenes compared: **{comparison['scenes_compared']}**.",
                  f"Mean minADE_1: with poke **{comparison['with_poke_minADE_1']:.6f}**, without poke **{comparison['without_poke_minADE_1']:.6f}**.",
                  f"Using a poke {direction} mean error by **{number(abs(improvement) if improvement is not None else None)}%**.",
                  f"Improved / worsened / tied scenes: **{comparison['scenes_improved']} / {comparison['scenes_worsened']} / {comparison['scenes_tied']}**."]
    for mode in MODES:
        for category, cases in selections[mode].items():
            title = "Top cases" if category == "top_10" else "Failure cases"
            lines += ["", f"## {LABELS[mode]}: {title}", "",
                      "Ranked by minADE_1 only. Failure examples are the highest-error scenes, not a threshold-based failure-rate estimate.", "",
                      "| Rank | Scene | Object | minADE_1 | ADE_1 (px) | FDE_1 (px) | Visualizations |",
                      "|---:|---:|---|---:|---:|---:|---|"]
            for rank, r in enumerate(cases, 1):
                target = f"{mode}/visualizations/{category}/rank_{rank:02d}_scene_{r['index']:04d}"
                lines.append(f"| {rank} | {r['index']} | {r['object_id']} | {r['minADE_1']:.6f} | "
                             f"{number(r['ADE_1_pixels'])} | {number(r['FDE_1_pixels'])} | [PNG]({target}/overview.png) / [MP4]({target}/rollout.mp4) |")
    lines += ["", "## Reuse the saved results", "",
              "Each selected visualization folder contains its own `results.npz` and `metrics.json`. Every evaluated scene is also archived under each mode's `results/` folder.",
              "The saved arrays contain all five hypotheses, ground truth, masks, reference/final images, all rendered-state RGB images, point and link IDs, state times, observed positions, and metadata.",
              "Rendered future-state images are visualization backgrounds only; they were never supplied as model inputs.",
              "The primary PNG/GIF/MP4 exports show prediction 1, matching the ranking. `oracle_best5.png` is a separately labeled supplementary result.", "",
              "Regenerate all presentation exports without a checkpoint load or model inference:", "", "```bash",
              f"python -m finetuning_pair.evaluate --render-only --output-dir {root}", "```", "",
              "For a different renderer, load a case directly:", "", "```python", "import numpy as np",
              'with np.load("path/to/results.npz", allow_pickle=False) as saved:',
              '    prediction_1 = saved["predictions"][0]  # [states, points, xy], normalized [0,1]',
              '    ground_truth = saved["ground_truth"]', '    valid = saved["valid"]',
              '    backgrounds = saved["rgb_frames"]  # uint8 [states, height, width, RGB]', "```", "",
              "`image` and `final_image` retain exact float32 [3,H,W] arrays in [-1,1]; `rgb_frames` contains display-ready uint8 images.",
              "Coordinates map to image pixel centers using `xy * (image_size - 1)`. Predictions can legitimately fall outside [0,1]; the arrays preserve these values.",
              "`known_states` is 2 with poke and 1 without. The first observed movement is motion progress, not a wall-clock interval.",
              "Checkpoint identity and checksums are stored in `run.json`. Rankings are deterministic, with scene index resolving ties.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
