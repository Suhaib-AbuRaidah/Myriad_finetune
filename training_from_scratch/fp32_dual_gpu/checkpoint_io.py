"""Restore old FP16 master weights or resume the separate FP32 trainer."""
from __future__ import annotations

import torch

from finetuning_pair.checkpoint_io import _validate_shapes, restore_rng_state
from training_from_scratch.checkpoint_io import (
    _read_checkpoint, inspect_checkpoint, make_encoder_reference,
    resolve_encoder_reference, save_training_checkpoint,
)

IMPLEMENTATION = "cached_fp32_device_adamw_v1"
CRITICAL_FIELDS = (
    "initialization", "architecture", "target_definition", "pair_selection", "data_root",
    "batch_size", "lr", "gradient_accumulation_steps", "weight_decay", "warmup_steps", "max_steps",
    "points_per_link", "image_size", "num_transitions", "time_scale", "max_train_scenes",
    "exclude_occluded_loss", "seed", "overlay_alpha", "freeze_dino", "dino_identity", "dino_precision",
)


def check_resume_config(saved, current):
    differences = [key for key in CRITICAL_FIELDS if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError("Resume changes training/data/schedule fields: " + ", ".join(differences))
    if saved.get("world_size", 1) != 1:
        raise ValueError("This trainer resumes single-process checkpoints only")
    if saved.get("training_implementation") not in (None, IMPLEMENTATION):
        raise ValueError("Unsupported training implementation")
    if saved.get("training_implementation") == IMPLEMENTATION:
        for key in ("precision", "amp", "feature_cache_dtype", "grad_clip"):
            if saved.get(key) != current.get(key):
                raise ValueError(f"Resume changes {key}")


def resume_checkpoint(path, model, optimizer, scheduler, *, restore_rng=True):
    payload = _read_checkpoint(path)
    if payload["kind"] != "training":
        raise ValueError("Resume requires latest.pt; best.pt contains inference weights only")
    if payload["optimizer"] is None or payload["scheduler"] is None:
        raise ValueError("Checkpoint lacks optimizer or scheduler state")
    if payload["train_config"].get("precision") == "fp16" and "master_parameters" not in payload["optimizer"]:
        raise ValueError("FP16 conversion requires the saved FP32 master parameters")
    saved = {key: value for key, value in payload["model"].items()
             if not key.startswith("model.image_embedder.")}
    _validate_shapes(saved, model.state_dict(), "Cached FP32 motion model")
    # load_state_dict copies values into FP32 destinations without changing dtype.
    model.load_state_dict(saved, strict=True)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("FP32 model placement was lost during checkpoint restoration")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    saved_names = payload["train_config"].get("optimizer_parameter_names")
    if saved_names is not None and saved_names != names:
        raise ValueError("Optimizer parameter order changed since the checkpoint")
    # Legacy master_parameters overwrite rounded FP16 values. New FP32 saves
    # use their exact model values, with no duplicate parameter storage.
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    model.train(model.training)
    return payload["step"]
