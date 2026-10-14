"""Pretrained initialization and resumable MYRIAD LoRA checkpoints.

The supplied ``myriad_openset.pt`` is a legacy training checkpoint containing
OmegaConf objects as well as model weights. ``load_pretrained_weights`` reads
that archive through a restricted unpickler that permits only its known tensor
and container constructors; OmegaConf metadata is discarded. Model parameters
are loaded strictly into the *unwrapped* MYRIAD model before LoRA injection::

    reference = load_pretrained_weights(bare_model, "myriad_openset.pt")
    adapted = FineTuningModel(bare_model, LoRAConfig(rank=8))
    optimizer = torch.optim.AdamW(adapted.optimizer_groups(
        lora_lr=1e-4, other_lr=3e-5,
    ))

    save_training_checkpoint(path, adapted, optimizer, scheduler,
                             step=step, base_checkpoint=reference,
                             train_config=run_config)
    # On resume: create a fresh bare model, load the same base checkpoint,
    # wrap with the saved LoRA config, then create optimizer/scheduler and:
    step = resume_training_checkpoint(path, adapted, optimizer, scheduler,
                                      base_checkpoint=reference)

The new checkpoint stores all trainable tensors and persistent non-encoder
buffers. Original frozen transformer weights and DINO weights are supplied by
the referenced base checkpoint, so their multi-gigabyte copy is not repeated.
It also stores optimizer, scheduler, RNG and optional sampler state. Loading a
fine-tuning checkpoint requires the same LoRA configuration and base reference.
Use ``load_finetuned_weights`` for evaluation without restoring training state.

The base reference checks path, file size, modification time and the archive's
pickle metadata digest. This is a cheap identity check, not a hash of all
weight bytes. For a reproducible data stream after resume, the future trainer
must also restore its sampler position and avoid untracked prefetched workers.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import types
import typing
import zipfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.paths import artifact_path
from finetuning_pair.model_adapter import FineTuningModel, LoRAConfig


_FORMAT = "myriad_articulated_lora"
_VERSION = 1


@dataclass(frozen=True)
class BaseCheckpointReference:
    path: str
    size_bytes: int
    mtime_ns: int
    metadata_sha256: str


class _IgnoredOmega:
    """Inert holder for training metadata in the legacy archive."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __setstate__(self, state: Any) -> None:
        pass


class _RestrictedLegacyUnpickler(pickle.Unpickler):
    """Allow only the globals actually needed by the supplied legacy file."""

    _ALLOWED = {
        ("collections", "OrderedDict"): OrderedDict,
        ("collections", "defaultdict"): defaultdict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch", "FloatStorage"): torch.FloatStorage,
        ("torch", "BFloat16Storage"): torch.BFloat16Storage,
        ("typing", "Any"): typing.Any,
        ("__builtin__", "list"): list,
        ("__builtin__", "dict"): dict,
        ("__builtin__", "long"): int,
    }
    _IGNORED = {
        ("omegaconf.listconfig", "ListConfig"),
        ("omegaconf.base", "ContainerMetadata"),
        ("omegaconf.nodes", "AnyNode"),
        ("omegaconf.base", "Metadata"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._IGNORED:
            return _IgnoredOmega
        try:
            return self._ALLOWED[module, name]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Unsupported global in pretrained checkpoint: {module}.{name}"
            ) from error


_restricted_pickle = types.ModuleType("myriad_restricted_legacy_pickle")
_restricted_pickle.Unpickler = _RestrictedLegacyUnpickler


def _base_reference(path: str | Path) -> BaseCheckpointReference:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"Expected a PyTorch ZIP checkpoint: {source}")
    with zipfile.ZipFile(source) as archive:
        metadata_members = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(metadata_members) != 1:
            raise ValueError(f"Expected one data.pkl metadata member in {source}")
        metadata_sha256 = hashlib.sha256(archive.read(metadata_members[0])).hexdigest()
    stat = source.stat()
    return BaseCheckpointReference(str(source), stat.st_size, stat.st_mtime_ns, metadata_sha256)


def _tensor_state(value: Any, name: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty tensor state dict")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()):
        raise ValueError(f"{name} must contain only string keys and tensor values")
    return value


def _validate_shapes(
    saved: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor], name: str
) -> None:
    missing = set(expected) - set(saved)
    extra = set(saved) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{name} keys do not match: {len(missing)} missing, {len(extra)} extra; "
            f"examples: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )
    for key, value in saved.items():
        if value.shape != expected[key].shape:
            raise ValueError(
                f"{name} shape mismatch for {key}: {tuple(value.shape)} "
                f"versus {tuple(expected[key].shape)}"
            )


def load_pretrained_weights(
    bare_model: nn.Module, path: str | Path
) -> BaseCheckpointReference:
    """Strictly initialize a bare MYRIAD model from its original checkpoint.

    Only call this on a local pretrained archive you intend to use. The
    restricted legacy loader refuses unfamiliar pickle globals. ``mmap=True``
    avoids materializing the saved optimizer states; PyTorch still reads model
    tensors as they are copied into the actual model.
    """
    if not isinstance(bare_model, nn.Module) or isinstance(bare_model, FineTuningModel):
        raise TypeError("Load pretrained weights into an unwrapped nn.Module")
    reference = _base_reference(path)
    try:
        payload = torch.load(
            reference.path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
            pickle_module=_restricted_pickle,
        )
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError(f"Cannot load restricted pretrained checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("Pretrained checkpoint must contain a 'model' state dict")
    state = _tensor_state(payload["model"], "Pretrained model")
    _validate_shapes(state, bare_model.state_dict(), "Pretrained model")
    bare_model.load_state_dict(state, strict=True)
    return reference


def _persistent_adaptation_keys(model: FineTuningModel) -> set[str]:
    state = model.state_dict()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name for name, _ in model.named_buffers()}
    return trainable | {
        name for name in buffers if name in state and not name.startswith("model.image_embedder.")
    }


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    return value


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise ValueError(f"{name} keys must be strings")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON-compatible finite values") from error
    return result


def capture_rng_state(
    generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    """Capture main-process Python, NumPy, Torch and named loader generators."""
    numpy_state = np.random.get_state()
    generator_states = {}
    for name, generator in (generators or {}).items():
        if not isinstance(name, str) or not isinstance(generator, torch.Generator):
            raise TypeError("generators must map names to torch.Generator objects")
        generator_states[name] = generator.get_state().cpu().clone()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(),
            int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]),
        ),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "generators": generator_states,
    }


def _check_rng_targets(
    state: Mapping[str, Any], generators: Mapping[str, torch.Generator] | None
) -> None:
    expected = set(state["generators"])
    supplied = set(generators or {})
    if expected != supplied:
        raise ValueError(f"Loader generator names differ: saved={sorted(expected)}, got={sorted(supplied)}")
    saved_cuda = len(state["torch_cuda"])
    current_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_cuda != current_cuda:
        raise ValueError(f"CUDA device count differs: saved={saved_cuda}, got={current_cuda}")


def restore_rng_state(
    state: Mapping[str, Any], generators: Mapping[str, torch.Generator] | None = None
) -> None:
    """Restore main-process generators captured by ``capture_rng_state``."""
    _check_rng_targets(state, generators)
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name].cpu())


def _validate_step(step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative count of completed optimizer updates")


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_training_checkpoint(
    path: str | Path,
    model: FineTuningModel,
    optimizer: Optimizer,
    scheduler: Any | None,
    *,
    step: int,
    base_checkpoint: BaseCheckpointReference,
    train_config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    data_state: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    """Atomically save adapters, ordinary trained modules and resume state."""
    if not isinstance(model, FineTuningModel) or not isinstance(optimizer, Optimizer):
        raise TypeError("Expected a FineTuningModel and initialized optimizer")
    if not isinstance(base_checkpoint, BaseCheckpointReference):
        raise TypeError("base_checkpoint must be returned by load_pretrained_weights")
    _validate_step(step)
    model.assert_training_policy()
    destination = artifact_path(path)
    if destination == Path(base_checkpoint.path):
        raise ValueError("Refusing to overwrite the pretrained base checkpoint")
    config = _json_object(train_config, "train_config")
    if config.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("Pair checkpoints require the frozen-parent relative target definition")
    values = _json_object(metrics or {}, "metrics")
    if data_state is not None and not isinstance(data_state, Mapping):
        raise TypeError("data_state must be a mapping")
    state = model.state_dict()
    keys = _persistent_adaptation_keys(model)
    payload = {
        "format": _FORMAT,
        "version": _VERSION,
        "step": step,
        "base_checkpoint": asdict(base_checkpoint),
        "lora_config": model.lora_config.to_dict(),
        "target_modules": list(model.target_modules),
        "train_config": config,
        "metrics": values,
        "model": {name: _cpu_copy(state[name]) for name in sorted(keys)},
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "scheduler": _cpu_copy(scheduler.state_dict()) if scheduler is not None else None,
        "rng": capture_rng_state(generators),
        "data_state": _cpu_copy(data_state) if data_state is not None else None,
    }
    _atomic_save(payload, destination)


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError("Expected a MYRIAD fine-tuning checkpoint")
    if payload.get("version") != _VERSION:
        raise ValueError(f"Unsupported fine-tuning checkpoint version: {payload.get('version')}")
    if payload.get("train_config", {}).get("target_definition") != TARGET_DEFINITION:
        raise ValueError("Incompatible checkpoint: expected child-only frozen-parent relative targets")
    _validate_step(payload.get("step"))
    return payload


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read lightweight metadata without copying checkpoint tensor storage."""
    payload = _read_checkpoint(path)
    return {
        key: payload[key] for key in (
            "format", "version", "step", "base_checkpoint", "lora_config",
            "target_modules", "train_config", "metrics", "data_state",
        )
    }


def _validate_adaptation(
    payload: Mapping[str, Any], model: FineTuningModel,
    base_checkpoint: BaseCheckpointReference,
) -> Mapping[str, torch.Tensor]:
    model.assert_training_policy()
    if payload["base_checkpoint"] != asdict(base_checkpoint):
        raise ValueError("Pretrained base checkpoint identity does not match the saved reference")
    if payload["lora_config"] != model.lora_config.to_dict():
        raise ValueError("LoRA rank, alpha or dropout differs from saved configuration")
    if payload["target_modules"] != list(model.target_modules):
        raise ValueError("Adapted projection names differ from the saved model")
    saved = _tensor_state(payload["model"], "Fine-tuned model")
    expected_state = model.state_dict()
    expected = {name: expected_state[name] for name in _persistent_adaptation_keys(model)}
    _validate_shapes(saved, expected, "Fine-tuned model")
    for name, tensor in saved.items():
        if tensor.dtype != expected[name].dtype and not (tensor.is_floating_point() and expected[name].is_floating_point()):
            raise ValueError(f"Incompatible checkpoint dtype for {name}")
    return saved


def _restore_model_tensors(model: FineTuningModel, saved: Mapping[str, torch.Tensor]) -> None:
    # RFHeadDistributionOutput converts its head to the activation dtype during
    # training. Preserve the saved tensor dtype before optimizer restoration;
    # otherwise a bf16 head may silently resume with fp32 weights/moments.
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, tensor in saved.items():
        if name in parameters and parameters[name].dtype != tensor.dtype:
            parameters[name].data = parameters[name].data.to(dtype=tensor.dtype)
        elif name in buffers and buffers[name].dtype != tensor.dtype:
            buffers[name].data = buffers[name].data.to(dtype=tensor.dtype)
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint tensors: {result.unexpected_keys}")
    expected_missing = set(model.state_dict()) - set(saved)
    if set(result.missing_keys) != expected_missing:
        raise RuntimeError("Unexpected missing tensors while restoring fine-tuned model")
    model.assert_training_policy()


def load_finetuned_weights(
    path: str | Path, model: FineTuningModel, *,
    base_checkpoint: BaseCheckpointReference,
) -> int:
    """Load only fine-tuned model tensors for validation or inference."""
    payload = _read_checkpoint(path)
    saved = _validate_adaptation(payload, model, base_checkpoint)
    _restore_model_tensors(model, saved)
    return payload["step"]


def resume_training_checkpoint(
    path: str | Path,
    model: FineTuningModel,
    optimizer: Optimizer,
    scheduler: Any | None,
    *,
    base_checkpoint: BaseCheckpointReference,
    generators: Mapping[str, torch.Generator] | None = None,
    restore_rng: bool = True,
) -> int:
    """Restore model, optimizer, scheduler and RNG; return completed step."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an initialized torch Optimizer")
    payload = _read_checkpoint(path)
    saved = _validate_adaptation(payload, model, base_checkpoint)
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("Scheduler presence differs from the saved run")
    if restore_rng:
        _check_rng_targets(payload["rng"], generators)
    _restore_model_tensors(model, saved)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload["rng"], generators)
    return payload["step"]
