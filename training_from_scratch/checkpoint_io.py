"""Resumable training checkpoints and smaller inference snapshots.

Frozen-DINO runs reference the original encoder archive in compact mode.
Version-1 self-contained checkpoints remain readable.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

import torch

from finetuning_pair.checkpoint_io import (
    _atomic_save, _cpu_copy, _json_object, _tensor_state, _validate_shapes,
    _validate_step, capture_rng_state, restore_rng_state,
    _base_reference,
)
from finetuning_pair.geometry import TARGET_DEFINITION
from training_from_scratch.paths import artifact_path
from training_from_scratch.features import encoder_identity, load_frozen_encoder

FORMAT = "myriad_pair_full_scratch"
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
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
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


def _restore_model(payload, model, *, dino_checkpoint=None):
    saved = _tensor_state(payload["model"], "Full model")
    expected = model.state_dict()
    reference = payload["frozen_encoder"]
    if reference is not None:
        if not model.freeze_dino:
            raise ValueError("Compact frozen-DINO weights require a frozen-DINO model")
        reference = resolve_encoder_reference(reference, dino_checkpoint)
        expected = {name: value for name, value in expected.items() if not name.startswith("model.image_embedder.")}
    _validate_shapes(saved, expected, "Full model" if reference is None else "Motion model")
    if reference is not None:
        # Restore the reference precision as well, including when inference
        # is constructed without the optimizer or training CLI.
        dtype = {"fp32": torch.float32, "fp16": torch.float16}[payload["train_config"].get("dino_precision", "fp32")]
        model.image_embedder.to(dtype=dtype)
        load_frozen_encoder(model.image_embedder, reference["path"])
        model.frozen_encoder_reference = reference
    # The original FM head casts its weights to the activation dtype. Restore
    # that dtype before loading Adam state so resumed precision is preserved.
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    for name, value in saved.items():
        tensor = tensors[name]
        if tensor.dtype != value.dtype:
            tensor.data = tensor.data.to(dtype=value.dtype)
    result = model.load_state_dict(saved, strict=reference is None)
    if reference is not None:
        missing = set(model.state_dict()) - set(expected)
        if result.unexpected_keys or set(result.missing_keys) != missing:
            raise RuntimeError("Unexpected keys while restoring a compact checkpoint")
    model.assert_training_policy()
    model.train(model.training)  # Clear transient timestep embeddings.


def load_model_weights(path, model, *, dino_checkpoint=None):
    payload = _read_checkpoint(path)
    _restore_model(payload, model, dino_checkpoint=dino_checkpoint)
    return payload["step"]


def resume_training_checkpoint(path, model, optimizer, scheduler, *, restore_rng=True, scaler=None, dino_checkpoint=None):
    payload = _read_checkpoint(path)
    if payload["kind"] != "training":
        raise ValueError("This checkpoint contains inference weights only; resume from latest.pt")
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("Scheduler presence differs from saved run")
    _restore_model(payload, model, dino_checkpoint=dino_checkpoint)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise ValueError("FP16 resume requires the saved loss scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload["step"]
