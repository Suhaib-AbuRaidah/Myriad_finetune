"""Offline construction and end-to-end training policy for MYRIAD Large."""
from __future__ import annotations

from functools import wraps
from contextlib import nullcontext
from typing import Any, Iterator
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training_from_scratch.sampling import prepare_sampling_cache, sampler_attention_context


def build_myriad(*, encoder_only: bool = False) -> nn.Module:
    """Use the original Large architecture, initializing DINO from config only.

    MYRIAD's original constructor calls AutoModel.from_pretrained even when
    constructing a new model. Replace that call only during construction so
    no weights are downloaded or loaded for either encoder or motion model.
    """
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
    from myriad.model import MyriadStepByStep_Large

    config = DINOv3ViTConfig(
        image_size=384, patch_size=16, hidden_size=1024,
        intermediate_size=4096, num_hidden_layers=24,
        num_attention_heads=16, num_register_tokens=4,
    )

    def local_encoder(name: str, *args: Any, **kwargs: Any) -> nn.Module:
        if name != "facebook/dinov3-vitl16-pretrain-lvd1689m" or args or kwargs:
            raise ValueError(f"Unexpected encoder construction: {name}")
        encoder = DINOv3ViTModel(config)
        # Retain the same state-dict layer names used elsewhere in this repo
        # with Transformers 5.17, without registering layers twice.
        inner = encoder.model
        encoder.layer = inner.layer
        del encoder.model
        object.__setattr__(encoder, "model", inner)
        return encoder

    with patch.object(AutoModel, "from_pretrained", side_effect=local_encoder):
        if encoder_only:
            from myriad.dinov3 import DinoFeatureExtractor
            return DinoFeatureExtractor(image_size=384, model_version="dinov3_vitl16", requires_grad=False)
        return MyriadStepByStep_Large()


def _checkpoint_forward(block: nn.Module) -> None:
    original = block.forward

    @wraps(original)
    def forward(*args: Any, **kwargs: Any):
        if block.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)

    # Patching forward preserves all original parameter and buffer names.
    block.forward = forward


def _fp32_motion_residual(module: nn.Module, args):
    """Accumulate motion-block skip connections beyond FP16's 65504 limit.

    Autocast still runs the blocks' Linear layers and attention in FP16.
    Their outputs add to this FP32 stream before the next normalization.
    """
    if args and args[0].dtype == torch.float16:
        return (args[0].float(), *args[1:])


class ScratchModel(nn.Module):
    """Train encoder, projections, motion blocks and flow head together.

    Keep MYRIAD's fixed RoPE frequencies and flow-input scales fixed. These
    are mathematical constants, rather than learned model weights.
    """

    def __init__(self, model: nn.Module, *, gradient_checkpointing: bool = False, freeze_dino: bool = False):
        super().__init__()
        self.model = model
        self.freeze_dino = freeze_dino
        self.precision = "fp32"
        self.feature_cache = None
        self.model.train_image_feature_extractor = not freeze_dino
        self._residual_precision_hook = self.transformer.mid_level.register_forward_pre_hook(_fp32_motion_residual)
        if freeze_dino:
            self.image_embedder.requires_grad_(False)
        # Respect the architecture's fixed constants; all learned weights
        # are already requires_grad=True in the original constructor.
        self.assert_training_policy()
        if gradient_checkpointing:
            if not freeze_dino:
                encoder = self.image_embedder.model.model
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for block in self.transformer.mid_level:
                _checkpoint_forward(block)
        self.train()

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def image_embedder(self):
        return self.model.image_embedder

    @property
    def distribution_head(self):
        return self.model.distribution_head

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dino:
            self.image_embedder.eval()
        cache = getattr(getattr(self.distribution_head, "head", None), "_timestep_emb_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        return self

    def forward(self, *args: Any, **kwargs: Any):
        with self.autocast():
            return self.model(*args, **kwargs)

    @property
    def training_device(self):
        return next(self.transformer.parameters()).device

    def autocast(self, device=None):
        device = device or self.training_device
        return (torch.autocast(device_type=device.type, dtype=torch.float16)
                if self.precision == "fp16" else nullcontext())

    def place(self, device, *, precision="fp32", dino_device=None, dino_precision=None):
        """Place components separately without first staging DINO on GPU 0."""
        self.precision = precision
        dtype = torch.float16 if precision == "fp16" else torch.float32
        # 1e5 flow-input scales overflow FP16. Fixed numerical constants stay
        # FP32; every learned parameter still uses the requested storage dtype.
        scales = self.distribution_head.head.in_proj_scales.detach().float().clone()
        self.transformer.to(device=device, dtype=dtype)
        self.distribution_head.to(device=device, dtype=dtype)
        self.distribution_head.head.in_proj_scales.data = scales.to(device=device)
        dino_precision = dino_precision or ("fp32" if self.freeze_dino else precision)
        self.image_embedder.to(device=dino_device or device,
                               dtype=torch.float16 if dino_precision == "fp16" else torch.float32)
        if precision == "fp16":
            output = self.distribution_head

            def distribution(cond, steps=None, cfg_scale=None):
                from myriad.model import RFHeadDistribution
                return RFHeadDistribution(output.head, cond.to(dtype),
                                          steps=output.steps if steps is None else steps,
                                          cfg_scale=output.cfg_scale if cfg_scale is None else cfg_scale)

            # Original forward calls head.to(cond.dtype), which would also
            # downcast the fixed 1e5 scales to inf. Preserve those constants.
            output.forward = distribution
        return self

    def embed_image(self, image):
        encoder_device = next(self.image_embedder.parameters()).device

        def compute(single):
            # Explicitly disable the training GPU's outer FP16 autocast when
            # frozen pretrained DINO needs FP32 range, even on another GPU.
            encoder_dtype = next(self.image_embedder.parameters()).dtype
            with torch.autocast(device_type=encoder_device.type, dtype=torch.float16,
                                enabled=encoder_dtype == torch.float16):
                row = self.model.embed_image(single.to(encoder_device))
            row["feat"] = row["feat"].to(next(self.transformer.parameters()).dtype)
            return row

        if self.feature_cache is None:
            embedding = compute(image)
        else:
            rows = []
            for single in image.split(1):
                row = self.feature_cache.read(single)
                if row is None:
                    row = compute(single)
                    self.feature_cache.write(single, row)
                rows.append(row)
            embedding = {key: torch.cat([row[key].to(self.training_device) for row in rows])
                         if torch.is_tensor(rows[0][key]) else rows[0][key] for key in rows[0]}
        return {key: value.to(self.training_device) if torch.is_tensor(value) else value
                for key, value in embedding.items()}

    @torch.no_grad()
    def predict_simulate(self, *args: Any, **kwargs: Any):
        with self.autocast(), sampler_attention_context():
            prepare_sampling_cache(self)
            return self.model.predict_simulate(*args, **kwargs)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def assert_training_policy(self) -> None:
        fixed = {"transformer.pos_emb.freqs", "transformer.pos_emb.t_freqs",
                 "distribution_head.head.in_proj_scales"}
        if self.model.train_image_feature_extractor == self.freeze_dino:
            raise ValueError("Unexpected image-encoder gradient policy")
        for name, parameter in self.model.named_parameters():
            frozen = name in fixed or (self.freeze_dino and name.startswith("image_embedder."))
            if parameter.requires_grad == frozen:
                raise ValueError(f"Unexpected trainability for {name}")
            if "lora_" in name:
                raise ValueError("Scratch models must not contain LoRA adapters")

    def parameter_report(self) -> dict[str, Any]:
        groups = {}
        for name, parameter in self.model.named_parameters():
            group = name.split(".")[0]
            counts = groups.setdefault(group, {"total": 0, "trainable": 0})
            counts["total"] += parameter.numel()
            counts["trainable"] += parameter.numel() if parameter.requires_grad else 0
        trainable = sum(c["trainable"] for c in groups.values())
        return {
            "initialization": "random_motion_pretrained_frozen_dino" if self.freeze_dino else "random_all_learned_weights",
            "precision": self.precision,
            "total_parameters": sum(c["total"] for c in groups.values()),
            "trainable_parameters": trainable,
            "groups": groups,
            "parameter_storage_GiB": sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**3,
            # FP32 parameters + gradients + two Adam moments, before activations.
            "estimated_fp32_adam_training_GiB": trainable * 16 / 1024**3,
        }
