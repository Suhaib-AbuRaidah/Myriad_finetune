"""Frozen DINO initialization and content-addressed image-feature caches."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from finetuning_pair.checkpoint_io import _base_reference, _restricted_pickle, _validate_shapes, _atomic_save
from training_from_scratch.paths import artifact_path


def encoder_identity(checkpoint):
    reference = asdict(_base_reference(checkpoint))
    return hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()


def load_frozen_encoder(encoder, checkpoint):
    """Read only encoder tensors from the local legacy checkpoint via mmap."""
    payload = torch.load(Path(checkpoint), map_location="cpu", mmap=True,
                         weights_only=False, pickle_module=_restricted_pickle)
    prefix = "image_embedder."
    state = {name[len(prefix):]: value for name, value in payload["model"].items()
             if name.startswith(prefix)}
    _validate_shapes(state, encoder.state_dict(), "Frozen DINO")
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval()


class FeatureCache:
    def __init__(self, root, *, identity, image_size, dtype, feature_dtype=None, require_cached=False):
        self.root = artifact_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = json.dumps({"version": 1, "encoder": identity,
                                     "image_size": image_size, "dtype": str(dtype),
                                     "feature_dtype": str(feature_dtype or dtype)}, sort_keys=True)
        self.hits = self.misses = 0
        self.require_cached = require_cached

    def path(self, image):
        # RGB already includes pair-specific role overlays, crop and resize.
        # FPS changes points but not this image; every visit can reuse features.
        digest = hashlib.sha256(self.namespace.encode())
        digest.update(str(tuple(image.shape)).encode())
        digest.update(image.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
        return self.root / f"{digest.hexdigest()}.pt"

    def read(self, image):
        path = self.path(image)
        if not path.exists():
            self.misses += 1
            if self.require_cached:
                raise FileNotFoundError(f"Missing DINO feature {path}; run training_from_scratch.cache_features first")
            return None
        self.hits += 1
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("namespace") != self.namespace:
            raise ValueError(f"Incompatible feature cache: {path}")
        return saved["embedding"]

    def write(self, image, embedding):
        _atomic_save({"namespace": self.namespace, "embedding": {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in embedding.items()}}, self.path(image))
