"""Fixed-subset test rollouts with metrics and visual artifacts.

Use ``test2`` for periodic previews and reserve ``test1`` for final testing.
Nothing in this module affects checkpoint selection or the training RNG stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from finetuning_pair.sampling import prepare_sampling_cache
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.paths import artifact_path
from finetuning_pair.validation import (
    _autocast_context, _sampler_attention_context, choose_scene_indices,
    paper_style_mse,
)
from finetuning_pair.visualizer import render_rollout


@dataclass(frozen=True)
class TestConfig:
    num_scenes: int = 8
    ensemble_size: int = 5
    diffusion_steps: int = 16
    seed: int = 42
    poke_mode: str = "both"
    visible_only: bool = False
    amp_dtype: str = "auto"

    def __post_init__(self) -> None:
        for name in ("num_scenes", "ensemble_size", "diffusion_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ensemble_size < 5:
            raise ValueError("Periodic testing needs at least five hypotheses for BEST-5")
        if self.poke_mode not in ("with", "without", "both"):
            raise ValueError("poke_mode must be with, without, or both")
        if self.amp_dtype not in ("auto", "none", "bf16"):
            raise ValueError("amp_dtype must be auto, none, or bf16")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.visible_only, bool):
            raise TypeError("visible_only must be a bool")


def _json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _to_cpu_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def sample_rollout(
    model: nn.Module,
    image: torch.Tensor,
    known_positions: torch.Tensor,
    times: torch.Tensor,
    *,
    ensemble_size: int,
    diffusion_steps: int,
    seed: int,
    amp_dtype: str = "auto",
) -> torch.Tensor:
    """Sample [E,T,N,2] from a reference image and one or two known states.

    ``known_positions`` is [observed_states,N,2] in [0,1]. With one state,
    MYRIAD predicts the first displacement and feeds its own motion back.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be [3,H,W]")
    if known_positions.ndim != 3 or known_positions.shape[-1] != 2 or known_positions.shape[0] not in (1, 2):
        raise ValueError("known_positions must be [1 or 2,N,2]")
    if times.ndim != 1 or len(times) <= known_positions.shape[0]:
        raise ValueError("times must contain at least one future state")
    if ensemble_size < 1 or diffusion_steps < 1:
        raise ValueError("ensemble_size and diffusion_steps must be positive")
    device = next(model.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices), _sampler_attention_context():
            torch.manual_seed(seed)
            image = image.to(device=device, dtype=torch.float32).unsqueeze(0)
            known = known_positions.to(device=device, dtype=torch.float32)
            given_pos = known.reshape(1, -1, 2).repeat(ensemble_size, 1, 1)
            ts = times.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(ensemble_size, 1)
            with _autocast_context(device, amp_dtype):
                embedding = model.embed_image(image)
                d_img = {
                    key: value.repeat_interleave(ensemble_size, dim=0) if isinstance(value, torch.Tensor) else value
                    for key, value in embedding.items()
                }
                prepare_sampling_cache(model)
                predicted = model.predict_simulate(
                    n_traj=known.shape[1], ts=ts, given_pos=given_pos,
                    camera_static=torch.ones(ensemble_size, device=device, dtype=torch.bool),
                    d_img=d_img, verbose=False, d_steps=diffusion_steps,
                )
            expected = (ensemble_size, len(times), known.shape[1], 2)
            if predicted.shape != expected:
                raise ValueError(f"MYRIAD returned {tuple(predicted.shape)}, expected {expected}")
            return predicted
    finally:
        model.train(was_training)


def _scene_inputs(sample: Mapping[str, Any], initial_poke: bool, visible_only: bool):
    if visible_only and sample.get("target_definition") == TARGET_DEFINITION:
        raise ValueError("Relative visibility requires counterfactual renders; use geometric validity")
    known_states = 2 if initial_poke else 1
    positions = sample["trajectories"]
    point_valid = sample["point_valid"]
    position_valid = sample["position_valid"]
    visibility = sample["visibility"]
    if positions.ndim != 3 or positions.shape[0] <= known_states:
        raise ValueError("Test scene has no future trajectory states")
    conditionable = point_valid & position_valid[:known_states].all(dim=0)
    if not bool(conditionable.any()):
        return None
    valid = position_valid[:, conditionable].clone()
    if visible_only:
        valid[known_states:] &= visibility[known_states:, conditionable]
    if not bool(valid[known_states:].any()):
        return None
    return (
        positions[:known_states, conditionable],
        positions[:, conditionable],
        valid,
        sample["point_link_nodes"][conditionable],
    )


def _link_errors(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                 links: torch.Tensor) -> dict[str, float]:
    result = {}
    for link in links.unique():
        selected = links == link
        if bool(valid[:, selected].any()):
            result[str(int(link))] = float(paper_style_mse(
                predicted[:, selected].unsqueeze(0), target[:, selected], valid[:, selected]
            )[0].item())
    return result


def run_test_subset(
    model: nn.Module,
    dataset: Dataset,
    config: TestConfig,
    output_root: str | Path,
    *,
    step: int,
) -> dict[str, Any]:
    """Evaluate a fixed subset and save predictions, PNGs, GIFs and JSON."""
    if getattr(dataset, "training", False):
        raise ValueError("Periodic tests require a dataset with training=False")
    if step < 0:
        raise ValueError("step must be nonnegative")
    split = dataset.scene_dir.parent.name
    step_dir = artifact_path(output_root) / split / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    indices = choose_scene_indices(len(dataset), config.num_scenes, config.seed)
    modes = ("with", "without") if config.poke_mode == "both" else (config.poke_mode,)
    scores: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    scene_records = []
    started = time.perf_counter()
    for index in indices:
        sample = dataset[index]
        scene_dir = step_dir / f"scene_{index:04d}_{Path(sample['scene_path']).stem[:8]}"
        scene_modes = {}
        for mode in modes:
            initial_poke = mode == "with"
            inputs = _scene_inputs(sample, initial_poke, config.visible_only)
            if inputs is None:
                scene_modes[mode] = {"skipped": True}
                continue
            known, target, valid, links = inputs
            predicted = sample_rollout(
                model, sample["x"], known, sample["state_times"],
                ensemble_size=config.ensemble_size, diffusion_steps=config.diffusion_steps,
                seed=config.seed + index, amp_dtype=config.amp_dtype,
            )
            device = predicted.device
            known_states = known.shape[0]
            target_device = target.to(device)
            valid_device = valid[known_states:].to(device)
            errors = paper_style_mse(
                predicted[:, known_states:], target_device[known_states:], valid_device,
            )
            first_five = errors[:5]
            best_index = int(first_five.argmin().item())
            metrics = {
                "minADE_1": float(errors[0].item()),
                "minADE_5": float(first_five.min().item()),
                "minADE_N": float(errors.min().item()),
                "best_of_5_index": best_index,
                "hypothesis_errors": [float(value) for value in errors.tolist()],
                "first_scored_state": known_states,
                "points_conditioned": int(known.shape[1]),
                "per_link_best_5": _link_errors(
                    predicted[best_index, known_states:], target_device[known_states:],
                    valid_device, links.to(device),
                ),
            }
            mode_dir = scene_dir / ("with_poke" if initial_poke else "no_poke")
            mode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mode_dir / "predictions.npz",
                predictions=_to_cpu_array(predicted), ground_truth=_to_cpu_array(target),
                valid=_to_cpu_array(valid), link_nodes=_to_cpu_array(links),
                state_times=_to_cpu_array(sample["state_times"]),
                pair_labels=_to_cpu_array(sample["pair_labels"]),
                joint_index=np.array(sample["joint_index"]),
                target_definition=np.array(TARGET_DEFINITION),
                cache_key=np.array(sample["cache_key"]),
            )
            artifacts = render_rollout(
                sample["x"], predicted, links, mode_dir,
                observed_states=known_states, ground_truth=target,
                ground_truth_valid=valid, final_image=sample.get("final_x"),
                best_index=best_index,
                title=f"{split} pair {sample['pair_labels'].tolist()} | example {index} | {'with poke' if initial_poke else 'no poke'}",
                metrics=metrics,
            )
            record = {**metrics, "skipped": False, "artifacts": artifacts}
            _json_file(mode_dir / "metrics.json", record)
            scene_modes[mode] = record
            scores[mode].append(metrics)
        scene_records.append({
            "index": index, "sample_id": sample["sample_id"],
            "object_id": sample["object_id"], "pair_labels": sample["pair_labels"].tolist(),
            "joint_index": sample["joint_index"], "target_definition": TARGET_DEFINITION, "modes": scene_modes,
        })
    summaries = {}
    for mode in modes:
        if not scores[mode]:
            raise ValueError(f"No scoreable scenes in {mode} poke mode")
        summaries[mode] = {
            "minADE_1": math.fsum(row["minADE_1"] for row in scores[mode]) / len(scores[mode]),
            "minADE_5": math.fsum(row["minADE_5"] for row in scores[mode]) / len(scores[mode]),
            "minADE_N": math.fsum(row["minADE_N"] for row in scores[mode]) / len(scores[mode]),
            "scenes_evaluated": len(scores[mode]),
            "scenes_skipped": len(indices) - len(scores[mode]),
            "first_scored_state": 2 if mode == "with" else 1,
        }
    summary = {
        "checkpoint_step": step, "split": split, "target_definition": TARGET_DEFINITION,
        "settings": config.__dict__, "scene_indices": indices,
        "modes": summaries, "scenes": scene_records,
        "total_seconds": time.perf_counter() - started,
    }
    _json_file(step_dir / "metrics.json", summary)
    return {"step": step, "split": split, "path": str(step_dir),
            "modes": summaries, "total_seconds": summary["total_seconds"]}
