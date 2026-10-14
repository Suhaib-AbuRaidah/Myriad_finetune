"""Small, repeatable trajectory validation for articulated MYRIAD fine-tuning.

Validation reports only minADE_1: the first sampled rollout's masked mean
squared Euclidean position error, averaged across pair examples. No oracle
selection among hypotheses is used. This retains the MYRIAD paper's squared
error formula (https://arxiv.org/html/2604.09527#S4.SS2). The dataset stores positions in [0, 1], while the paper's
evaluator scores positions in [-1, 1]; squared errors are scaled accordingly.

By default, one rendered transition supplies the paper-style motion hint and
only later states are scored. Without that hint, only reference-frame positions
are supplied and all subsequent states are scored. The validation split uses
the dataset's fixed, image-space FPS points and a seeded subset of scenes.
Its four transitions and ground-truth
link transforms differ from OWM, so these values are for tracking adaptation,
not numerical comparison with the published benchmark.

The training CLI controls the cadence, subset size, sampler settings and poke
mode. Evaluation is inference-only and restores the training mode and the
process's torch random state afterwards.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.data import Dataset
from finetuning_pair.sampling import prepare_sampling_cache
from finetuning_pair.geometry import TARGET_DEFINITION


@dataclass(frozen=True)
class ValidationConfig:
    num_scenes: int = 16
    ensemble_size: int = 1
    diffusion_steps: int = 16
    seed: int = 42
    visible_only: bool = False
    amp_dtype: str = "auto"  # auto (CUDA bf16 if supported), none, or bf16
    initial_poke: bool = True

    def __post_init__(self) -> None:
        for name in ("num_scenes", "ensemble_size", "diffusion_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.visible_only, bool):
            raise TypeError("visible_only must be a bool")
        if not isinstance(self.initial_poke, bool):
            raise TypeError("initial_poke must be a bool")
        if self.amp_dtype not in ("auto", "none", "bf16"):
            raise ValueError("amp_dtype must be auto, none, or bf16")


def choose_scene_indices(dataset_size: int, num_scenes: int, seed: int) -> list[int]:
    """Pick a fixed subset independent of DataLoader order and training RNG."""
    if dataset_size < 1 or num_scenes < 1:
        raise ValueError("dataset_size and num_scenes must be positive")
    count = min(dataset_size, num_scenes)
    return sorted(np.random.default_rng(seed).choice(dataset_size, count, replace=False).tolist())


def paper_style_mse(
    predictions: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Return paper-scale masked squared-distance means for [0, 1] inputs, [E].

    ``predictions`` is [E,T,N,2], ``target`` is [T,N,2], and ``valid`` is
    [T,N]. The minimum is taken only *after* averaging time and tracks, so
    different links cannot select different ensemble members.
    """
    if predictions.ndim != 4 or target.ndim != 3 or valid.ndim != 2:
        raise ValueError("Expected predictions [E,T,N,2], target [T,N,2], mask [T,N]")
    if predictions.shape[0] < 1 or predictions.shape[1:] != target.shape:
        raise ValueError("Prediction and target shapes do not agree")
    if target.shape[-1] != 2 or valid.shape != target.shape[:2]:
        raise ValueError("Expected 2D points and a matching time/point mask")
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError("The metric needs at least one valid position")
    if predictions.device != target.device or valid.device != target.device:
        raise ValueError("Predictions, target and mask must be on the same device")
    if not torch.isfinite(predictions).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Metric inputs contain nonfinite valid positions")
    # Float32 accumulation avoids bf16 rounding when sampling with autocast.
    # MYRIAD's published evaluator uses [-1, 1] image coordinates. A position
    # difference there is twice the difference in our [0, 1] dataset coordinates.
    difference = 2.0 * (predictions.float() - target.float().unsqueeze(0))
    squared_distance = difference.square().sum(dim=-1)
    return (squared_distance * valid.unsqueeze(0)).sum(dim=(1, 2)) / valid.sum()


def _autocast_context(device: torch.device, setting: str):
    if setting == "none":
        return nullcontext()
    if device.type == "cuda":
        if setting == "bf16" or torch.cuda.is_bf16_supported(including_emulation=False):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("CUDA bf16 autocast was requested but is unsupported")
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif setting == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _scene_tensors(
    sample: Mapping[str, Any], device: torch.device,
    visible_only: bool, initial_poke: bool = True,
):
    if visible_only and sample.get("target_definition") == TARGET_DEFINITION:
        raise ValueError("Relative visibility requires counterfactual renders; use geometric validity")
    trajectory = sample["trajectories"]
    position_valid = sample["position_valid"]
    visibility = sample["visibility"]
    point_valid = sample["point_valid"]
    times = sample["state_times"]
    image = sample["x"]
    known_states = 2 if initial_poke else 1
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2 or trajectory.shape[0] <= known_states:
        raise ValueError("Validation requires observed and future states of [N,2] tracks")
    states, tracks, _ = trajectory.shape
    if (position_valid.shape != (states, tracks) or visibility.shape != (states, tracks)
            or point_valid.shape != (tracks,) or times.shape != (states,)
            or image.ndim != 3 or image.shape[0] != 3):
        raise ValueError("Validation sample shapes do not match the dataset contract")
    if not bool(sample["camera_static"]):
        raise ValueError("MYRIAD articulated validation expects a static camera")
    if not torch.isfinite(times).all() or not torch.isfinite(trajectory[position_valid]).all():
        raise ValueError("Nonfinite validation times or valid positions")
    # Only positions actually supplied to the model may determine which points
    # are conditionable. In no-poke mode, state 1 is a prediction target.
    conditionable = point_valid & position_valid[:known_states].all(dim=0)
    if not bool(conditionable.any()):
        return None
    target = trajectory[known_states:, conditionable].to(device=device, dtype=torch.float32)
    valid = position_valid[known_states:, conditionable]
    if visible_only:
        valid = valid & visibility[known_states:, conditionable]
    if not bool(valid.any()):
        return None
    return (
        image.unsqueeze(0).to(device=device, dtype=torch.float32),
        trajectory[:known_states, conditionable].to(device=device, dtype=torch.float32),
        times.to(device=device, dtype=torch.float32),
        target,
        valid.to(device=device),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validation_flex_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    scale: float | None, block_mask: Any,
) -> torch.Tensor:
    """Keep MYRIAD's FlexAttention and BlockMask with RTX 4070-safe tiles."""
    return flex_attention(
        q, k, v, scale=scale, block_mask=block_mask,
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 1},
    )


_validation_flex_attention_compiled = torch.compile(_validation_flex_attention, dynamic=False)


def _sampler_attention_context():
    """Use the original sparse attention path with a smaller compiled tile."""
    original_module = sys.modules.get("myriad.model")
    if original_module is None or not hasattr(original_module, "flex_attention_compiled"):
        return nullcontext()
    return patch.object(original_module, "flex_attention_compiled", _validation_flex_attention_compiled)


def evaluate_validation(
    model: nn.Module,
    dataset: Dataset,
    config: ValidationConfig,
    *,
    device: torch.device | str | None = None,
    scene_indices: Sequence[int] | None = None,
    scene_callback: Callable[[int, Mapping[str, Any], torch.Tensor | None, torch.Tensor | None], None] | None = None,
) -> dict[str, float | int]:
    """Sample a bounded validation subset and report only first-rollout error.

    Each scene is evaluated separately to avoid padding invalid trajectories
    into MYRIAD's autoregressive inference. A single DINO embedding is shared
    across its ensemble. Sampling RNG is forked, making repeated evaluations
    comparable without changing the training RNG stream.
    Explicit ``scene_indices`` retain the original scene IDs and RNG seeds for
    resumed evaluations. The callback receives the exact scored predictions
    before they are released, or two ``None`` values for an unscoreable scene.
    """
    if not isinstance(config, ValidationConfig):
        raise TypeError("config must be a ValidationConfig")
    if getattr(dataset, "training", False):
        raise ValueError("Validation requires a dataset with training=False")
    model_device = next(model.parameters()).device
    device = model_device if device is None else torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device != model_device:
        raise ValueError("Model and validation device differ")
    indices = (choose_scene_indices(len(dataset), config.num_scenes, config.seed)
               if scene_indices is None else list(scene_indices))
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(dataset) for i in indices)):
        raise ValueError("scene_indices must contain unique valid dataset indices")
    was_training = model.training
    scores_1 = []
    sampled_hypotheses = 0
    used_points = 0
    skipped = 0
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    _synchronize(device)
    start = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices), _sampler_attention_context():
            for index in indices:
                sample = dataset[index]
                prepared = _scene_tensors(
                    sample, device, config.visible_only, config.initial_poke,
                )
                if prepared is None:
                    skipped += 1
                    if scene_callback is not None:
                        scene_callback(index, sample, None, None)
                    continue
                image, known, times, target, valid = prepared
                tracks = known.shape[1]
                # The no-poke path supplies only state 0. MYRIAD then samples
                # state 1 and autoregressively uses its own predicted motion.
                given_pos = known.reshape(1, -1, 2).repeat(config.ensemble_size, 1, 1)
                ts = times.unsqueeze(0).repeat(config.ensemble_size, 1)
                torch.manual_seed(config.seed + index)
                with _autocast_context(device, config.amp_dtype):
                    embedding = model.embed_image(image)
                    d_img = {
                        key: value.repeat_interleave(config.ensemble_size, dim=0)
                        if isinstance(value, torch.Tensor) else value
                        for key, value in embedding.items()
                    }
                    prepare_sampling_cache(model)
                    predicted = model.predict_simulate(
                        n_traj=tracks, ts=ts, given_pos=given_pos,
                        camera_static=torch.ones(config.ensemble_size, device=device, dtype=torch.bool),
                        d_img=d_img, verbose=False, d_steps=config.diffusion_steps,
                    )
                expected = (config.ensemble_size, len(times), tracks, 2)
                if predicted.shape != expected:
                    raise ValueError(f"MYRIAD returned {tuple(predicted.shape)}, expected {expected}")
                errors = paper_style_mse(predicted[:, known.shape[0]:], target, valid)
                scores_1.append(float(errors[0].item()))
                used_points += tracks
                sampled_hypotheses += config.ensemble_size
                if scene_callback is not None:
                    scene_callback(index, sample, predicted, errors)
    finally:
        model.train(was_training)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    if not scores_1:
        raise ValueError("Selected validation scenes have no scoreable future positions")
    result = {
        "minADE_1": math.fsum(scores_1) / len(scores_1),
        "ensemble_size": config.ensemble_size,
        "diffusion_steps": config.diffusion_steps,
        "initial_poke": config.initial_poke,
        "first_scored_state": 2 if config.initial_poke else 1,
        "scenes_evaluated": len(scores_1),
        "scenes_skipped": skipped,
        "points_conditioned": used_points,
        "hypotheses_sampled": sampled_hypotheses,
        "sampling_seconds": elapsed,
        "hypotheses_per_second": sampled_hypotheses / elapsed if elapsed > 0 else float("inf"),
    }
    if hasattr(dataset, "examples"):
        result.update(target_definition=TARGET_DEFINITION, pairs_evaluated=len(scores_1),
                      pairs_skipped=skipped, metric_unit="selected child pair")
    return result
