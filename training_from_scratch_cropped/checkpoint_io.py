"""Resumable training checkpoints and smaller inference snapshots.

Frozen-DINO runs reference the original encoder archive in compact mode.
Only the isolated cropped checkpoint format is accepted.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

import torch

from finetuning_pair.checkpoint_io import (
    _atomic_save, _cpu_copy, _json_object, _validate_shapes,
    _validate_step, capture_rng_state, restore_rng_state,
    _base_reference,
)
from finetuning_pair.geometry import TARGET_DEFINITION
from .paths import artifact_path
from training_from_scratch.features import encoder_identity

FORMAT = "myriad_cropped_all_pairs_fp32"
VERSION = 2


def make_encoder_reference(path):
    """Record the local archive identity, without reading multi-GB tensor data."""
    return {**asdict(_base_reference(path)), "identity": encoder_identity(path)}


def resolve_encoder_reference(reference, override=None):
    """Validate the original archive or an explicitly relocated copy.

    Archive size and tensor metadata must match. The original path also keeps
    its modification-time check. This is the same inexpensive archive identity
    scheme used by the repository, not a checksum of every weight byte.
    """
    actual = asdict(_base_reference(override or reference["path"]))
    for key in ("size_bytes", "metadata_sha256"):
        if actual[key] != reference[key]:
            raise ValueError("DINO checkpoint differs from the saved encoder reference")
    if actual["path"] == reference["path"] and actual["mtime_ns"] != reference["mtime_ns"]:
        raise ValueError("DINO checkpoint was modified after this run was saved")
    return {**actual, "identity": reference["identity"]}


def save_training_checkpoint(path, model, optimizer, scheduler, *, step: int,
                             train_config: Mapping[str, Any], metrics=None, data_state=None, scaler=None,
                             checkpoint_storage="compact", weights_only=False):
    model.assert_training_policy()
    _validate_step(step)
    config = _json_object(train_config, "train_config")
    if config.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("Scratch checkpoints require child-relative targets")
    if checkpoint_storage not in ("compact", "full"):
        raise ValueError("checkpoint_storage must be compact or full")
    state = model.state_dict()
    reference = None
    if checkpoint_storage == "compact" and model.freeze_dino:
        reference = getattr(model, "frozen_encoder_reference", None)
        if reference is None or reference["identity"] != config.get("dino_identity"):
            raise ValueError("Compact checkpoints require the frozen encoder's path and identity")
        reference = resolve_encoder_reference(reference)
        state = {name: value for name, value in state.items() if not name.startswith("model.image_embedder.")}
    _atomic_save({
        "format": FORMAT, "version": VERSION, "step": step,
        "kind": "weights" if weights_only else "training", "frozen_encoder": reference,
        "train_config": config, "metrics": _json_object(metrics or {}, "metrics"),
        # torch.save streams CUDA storages through CPU one at a time. Avoid
        # materializing a second multi-GB checkpoint in host RAM first.
        "model": state,
        "optimizer": optimizer.state_dict() if not weights_only else None,
        "scaler": scaler.state_dict() if scaler is not None and not weights_only else None,
        "scheduler": _cpu_copy(scheduler.state_dict()) if scheduler is not None and not weights_only else None,
        "rng": capture_rng_state() if not weights_only else None,
        "data_state": _cpu_copy(data_state) if not weights_only else None,
    }, artifact_path(path))


def _read_checkpoint(path):
    # File-like streams allow checkpoint regression tests to round-trip the
    # complete format in memory without leaving generated checkpoint files.
    is_path = isinstance(path, (str, Path))
    payload = torch.load(Path(path) if is_path else path, map_location="cpu",
                         weights_only=True, mmap=is_path)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") not in (1, VERSION):
        raise ValueError("Expected a MYRIAD scratch-training checkpoint")
    if payload["train_config"].get("target_definition") != TARGET_DEFINITION:
        raise ValueError("Incompatible target definition")
    _validate_step(payload["step"])
    payload.setdefault("kind", "training")
    payload.setdefault("frozen_encoder", None)
    if payload["kind"] not in ("training", "weights"):
        raise ValueError("Unknown checkpoint kind")
    if payload["frozen_encoder"] is not None:
        config = payload["train_config"]
        if not config.get("freeze_dino") or payload["frozen_encoder"]["identity"] != config.get("dino_identity"):
            raise ValueError("Frozen encoder reference differs from the training configuration")
    return payload


def inspect_checkpoint(path):
    payload = _read_checkpoint(path)
    return {key: payload[key] for key in (
        "format", "version", "kind", "frozen_encoder", "step", "train_config", "metrics", "data_state",
    )}


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
