"""Render MYRIAD point rollouts over their reference and final-state images.

All coordinates are normalized to the displayed square image. Ground truth is
optional so the same renderer works for periodic tests and raw inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch


_COLORS = (
    (239, 83, 80), (66, 165, 245), (102, 187, 106), (255, 202, 40),
    (171, 71, 188), (38, 198, 218), (255, 112, 67), (156, 204, 101),
    (255, 167, 38), (121, 134, 203), (77, 182, 172), (240, 98, 146),
)


def _array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reference_image(image: torch.Tensor | np.ndarray) -> Image.Image:
    pixels = _array(image)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("Reference image must be [3,H,W] in [-1,1]")
    rgb = np.clip((pixels.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _xy(position: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    if not np.isfinite(position).all():
        return None
    x, y = float(position[0]), float(position[1])
    if not (-0.1 <= x <= 1.1 and -0.1 <= y <= 1.1):
        return None
    return round(x * (width - 1)), round(y * (height - 1))


def _draw_rollout(
    image: Image.Image, positions: np.ndarray, valid: np.ndarray,
    link_nodes: np.ndarray, frame: int, *, observed_states: int,
    alpha: int = 225,
) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for point in range(positions.shape[1]):
        rgb = _COLORS[point % len(_COLORS)]
        color = (*rgb, alpha)
        previous = None
        for time_index in range(frame + 1):
            current = _xy(positions[time_index, point], width, height) if valid[time_index, point] else None
            if previous is not None and current is not None:
                draw.line((previous, current), fill=color, width=2)
            previous = current
        current = _xy(positions[frame, point], width, height) if valid[frame, point] else None
        if current is None:
            continue
        radius = 4 if frame < observed_states else 3
        draw.ellipse((current[0]-radius, current[1]-radius, current[0]+radius, current[1]+radius),
                     fill=color, outline=(255, 255, 255, 235) if frame < observed_states else None)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_hypotheses(
    image: Image.Image, predictions: np.ndarray, link_nodes: np.ndarray,
    *, ground_truth: np.ndarray | None, ground_truth_valid: np.ndarray | None,
    best_index: int | None,
) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for hypothesis in range(predictions.shape[0]):
        for point in range(predictions.shape[2]):
            color = _COLORS[point % len(_COLORS)]
            alpha = 175 if hypothesis == best_index else 55
            coords = [_xy(predictions[hypothesis, t, point], width, height)
                      for t in range(predictions.shape[1])]
            for first, second in zip(coords, coords[1:]):
                if first is not None and second is not None:
                    draw.line((first, second), fill=(*color, alpha), width=3 if hypothesis == best_index else 1)
    if ground_truth is not None:
        assert ground_truth_valid is not None
        for point in range(ground_truth.shape[1]):
            for t in range(1, ground_truth.shape[0]):
                if not (ground_truth_valid[t-1, point] and ground_truth_valid[t, point]):
                    continue
                first = _xy(ground_truth[t-1, point], width, height)
                second = _xy(ground_truth[t, point], width, height)
                if first is not None and second is not None:
                    draw.line((first, second), fill=(20, 25, 30, 245), width=5)
                    draw.line((first, second), fill=(255, 255, 255, 230), width=2)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_rollout(
    image: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    link_nodes: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    *,
    observed_states: int,
    ground_truth: torch.Tensor | np.ndarray | None = None,
    ground_truth_valid: torch.Tensor | np.ndarray | None = None,
    final_image: torch.Tensor | np.ndarray | None = None,
    best_index: int | None = None,
    title: str = "MYRIAD child-relative rollout",
    metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save a state-by-state contact sheet, animated GIF and ensemble overlay."""
    reference = _reference_image(image)
    final_reference = _reference_image(final_image) if final_image is not None else None
    if final_reference is not None and final_reference.size != reference.size:
        raise ValueError("Final-state image must match the reference image size")
    pred = _array(predictions).astype(np.float32)
    links = _array(link_nodes).astype(np.int64)
    if pred.ndim != 4 or pred.shape[-1] != 2 or links.shape != (pred.shape[2],):
        raise ValueError("Expected predictions [ensemble,states,points,2] and one link per point")
    if not 1 <= observed_states < pred.shape[1]:
        raise ValueError("observed_states must leave at least one predicted state")
    if best_index is not None and not 0 <= best_index < pred.shape[0]:
        raise ValueError("best_index is outside the ensemble")
    truth = None if ground_truth is None else _array(ground_truth).astype(np.float32)
    truth_valid = None if ground_truth_valid is None else _array(ground_truth_valid).astype(bool)
    if (truth is None) != (truth_valid is None):
        raise ValueError("Ground truth and its validity mask must be supplied together")
    if truth is not None and (truth.shape != pred.shape[1:] or truth_valid.shape != truth.shape[:2]):
        raise ValueError("Ground truth shape differs from the predictions")

    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    if truth is not None:
        rows.append(("Ground truth", truth, truth_valid))
    for index in range(min(2 if truth is None else 1, pred.shape[0])):
        rows.append((f"Sample {index + 1}", pred[index], np.ones(pred.shape[1:3], dtype=bool)))
    if best_index is not None:
        rows.append((f"Oracle BEST-{min(5, pred.shape[0])} (sample {best_index + 1})",
                     pred[best_index], np.ones(pred.shape[1:3], dtype=bool)))

    from training_from_scratch.paths import artifact_path
    output = artifact_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    width, height = reference.size
    heading, label_height = 36, 26
    sheet = Image.new("RGB", (width * pred.shape[1], heading + len(rows) * (height + label_height)), "#10151d")
    sheet_draw = ImageDraw.Draw(sheet)
    score = "" if not metrics or "minADE_5" not in metrics else f" | BEST-5 {metrics['minADE_5']:.5f}"
    sheet_draw.text((10, 10), f"{title}{score} | observed states: {observed_states}", fill="white")
    frames: list[Image.Image] = []
    for row_index, (label, positions, valid) in enumerate(rows):
        top = heading + row_index * (height + label_height)
        sheet_draw.text((10, top + 5), label, fill="white")
        for frame in range(pred.shape[1]):
            background = final_reference if final_reference is not None and frame == pred.shape[1] - 1 else reference
            panel = _draw_rollout(background, positions, valid, links, frame, observed_states=observed_states)
            sheet.paste(panel, (frame * width, top + label_height))
            status = "ground truth" if label == "Ground truth" else (
                "observed" if frame < observed_states else "predicted"
            )
            if final_reference is not None and frame == pred.shape[1] - 1:
                status += " on frozen-parent initial reference"
            sheet_draw.text((frame * width + 10, top + label_height + 8),
                            f"state {frame} | {status}",
                            fill="white", stroke_width=1, stroke_fill="black")
    sheet_path = output / "overview.png"
    sheet.save(sheet_path)

    for frame in range(pred.shape[1]):
        canvas = Image.new("RGB", (width * len(rows), height + label_height), "#10151d")
        draw = ImageDraw.Draw(canvas)
        for row_index, (label, positions, valid) in enumerate(rows):
            background = final_reference if final_reference is not None and frame == pred.shape[1] - 1 else reference
            panel = _draw_rollout(background, positions, valid, links, frame, observed_states=observed_states)
            canvas.paste(panel, (row_index * width, label_height))
            draw.text((row_index * width + 8, 6), f"{label} | state {frame}", fill="white")
        frames.append(canvas)
    gif_path = output / "rollout.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=750, loop=0, optimize=True)

    ensemble = _render_hypotheses(reference, pred, links, ground_truth=truth,
                                  ground_truth_valid=truth_valid, best_index=best_index)
    ensemble_path = output / "ensemble.png"
    ensemble.save(ensemble_path)
    artifacts = {"overview": str(sheet_path), "animation": str(gif_path), "ensemble": str(ensemble_path)}
    if final_reference is not None:
        selected = pred[best_index if best_index is not None else 0]
        final_panel = _draw_rollout(
            final_reference, selected, np.ones(pred.shape[1:3], dtype=bool), links,
            pred.shape[1] - 1, observed_states=observed_states,
        )
        final_path = output / "final_state.png"
        final_panel.save(final_path)
        artifacts["final_state"] = str(final_path)
    return artifacts
