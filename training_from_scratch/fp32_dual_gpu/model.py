"""FP32 motion model using the existing read-only FP16 DINO feature cache."""
from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from training_from_scratch.features import FeatureCache
from training_from_scratch.model import ScratchModel
from training_from_scratch.train import training_loss as original_training_loss


class CachedEncoderSpec(nn.Module):
    """Keep architecture metadata without allocating unused DINO weights."""

    def __init__(self, *, image_size=384, embed_dim=1024):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

    def forward(self, image):
        raise RuntimeError("This trainer requires precomputed DINO features")


def build_cached_myriad(*, image_size=384):
    from myriad.model import MyriadStepByStep_Large

    encoder = CachedEncoderSpec(image_size=image_size)
    with patch("myriad.model.DinoFeatureExtractor", return_value=encoder):
        return MyriadStepByStep_Large()


class CachedFP32Model(ScratchModel):
    def __init__(self, model, *, gradient_checkpointing=False):
        super().__init__(model, gradient_checkpointing=gradient_checkpointing, freeze_dino=True)

    def place(self, device):
        # Bypass the old placement helper: the encoder has no tensors to move.
        self.precision = "fp32"
        self.transformer.to(device=device, dtype=torch.float32)
        self.distribution_head.to(device=device, dtype=torch.float32)
        return self

    def autocast(self, device=None):
        return torch.autocast(device_type=(device or self.training_device).type, enabled=False)

    def configure_cache(self, root, *, identity, feature_dtype="fp16"):
        self.feature_cache = FeatureCache(
            root, identity=identity, image_size=self.image_embedder.image_size,
            dtype=torch.float32,
            feature_dtype={"fp16": torch.float16, "fp32": torch.float32}[feature_dtype],
            require_cached=True,
        )

    def embed_image(self, image):
        if self.feature_cache is None:
            raise RuntimeError("Configure the DINO feature cache before training or evaluation")
        rows = [self.feature_cache.read(single) for single in image.split(1)]
        length = rows[0]["L"]
        for row in rows:
            if row["L"] != length:
                raise ValueError("Cached images have different token counts")
            if not all(bool(torch.isfinite(row[key]).all()) for key in ("feat", "pos")):
                raise FloatingPointError("Nonfinite cached DINO features or positions")
        # Preserve the cache namespace and files; widen their stored values only
        # when loading them into the FP32 training model.
        return {"feat": torch.cat([row["feat"] for row in rows]).to(self.training_device, torch.float32),
                "pos": torch.cat([row["pos"] for row in rows]).to(self.training_device, torch.float32),
                "L": length}


def training_loss(model, batch, *, loss_denominator=None):
    # The RF loss runs after model.forward, so disable autocast around the
    # complete loss as well as inside the model, including ambient AMP contexts.
    with model.autocast():
        return original_training_loss(model, batch, amp_dtype="none", loss_denominator=loss_denominator)
