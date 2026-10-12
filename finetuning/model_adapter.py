"""Block-only LoRA and the MYRIAD fine-tuning parameter policy.

This module needs only PyTorch; it does not import/download DINO or load a large
checkpoint. Construct MYRIAD and load its pretrained weights FIRST, then wrap it::

    adapted = FineTuningModel(model, LoRAConfig(rank=8))
    adapted.train()  # auxiliary modules train; the image encoder stays in eval
    optimizer = torch.optim.AdamW(adapted.optimizer_groups(
        lora_lr=1e-4, other_lr=3e-5,
    ))

Only transformer.mid_level blocks are adapted: their fused in_proj (QKV + FFN)
and out_proj (attention + FFN) receive LoRA. Original block weights, normalization
parameters and attention scales are frozen. Motion embedding, cross_proj,
scale_proj, the transformer's output MLP, and the Flow Matching head train
normally. Fixed RoPE frequencies and the FM scale cascade stay fixed.

Use the wrapper's train/eval and embed_image methods throughout training. Its
registered child is ``model``; state_dict keys consequently start with ``model.``
and adapted projections contain ``base_layer`` keys. Resume checkpoints must be
loaded into an identically configured wrapper, rather than a bare MYRIAD model.
Checkpoint initialization/resume will be handled by the separate checkpoint module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    """The training CLI will expose rank, alpha and dropout as arguments.

    None alpha means alpha=rank, giving scaling alpha/rank=1. A is initialized
    normally and B to zero, so installing adapters preserves the base outputs.
    """

    rank: int
    alpha: float | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if self.alpha is not None and (not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")

    @property
    def scaling(self) -> float:
        return (self.rank if self.alpha is None else self.alpha) / self.rank

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoRALinear(nn.Module):
    """A frozen linear projection plus (alpha/rank) * B(A(dropout(x)))."""

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base layer")
        if not base_layer.weight.is_floating_point():
            raise ValueError("LoRALinear requires floating-point base weights")
        if config.rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError("LoRA rank must not exceed the projection's smaller dimension")
        self.base_layer = base_layer
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(config.rank, self.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(self.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base_layer.requires_grad_(False)
        self.train(base_layer.training)

    @property
    def weight(self) -> nn.Parameter:
        """Original weight, exposed for compatible linear-layer inspection."""
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base_layer(x) + update * self.config.scaling

    def delta_weight(self) -> torch.Tensor:
        """Dense adapter update for verification or a future inference export."""
        return (self.lora_B @ self.lora_A) * self.config.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.config.rank}, scaling={self.config.scaling}, "
            f"dropout={self.config.dropout}"
        )


# These are deliberately fixed nn.Parameters in the original MYRIAD source.
# Explicit names also work when a torch.hub inference loader has frozen every
# parameter: we restore ordinary learning without unfreezing these constants.
_FIXED_PARAMETER_NAMES = frozenset({
    "transformer.pos_emb.freqs",
    "transformer.pos_emb.t_freqs",
    "distribution_head.head.in_proj_scales",
})
_BLOCK_PREFIX = "transformer.mid_level."


class FineTuningModel(nn.Module):
    """Wrap a pretrained MYRIAD model and enforce the agreed training policy.

    Default expected_depth=24 ensures that all MYRIAD Large blocks are covered.
    A smaller expected_depth can be supplied for inexpensive structural tests.
    The wrapper forwards MYRIAD's image embedding and public prediction methods.
    """

    def __init__(
        self, model: nn.Module, config: LoRAConfig, *, expected_depth: int = 24
    ):
        super().__init__()
        targets = self._validate_model(model, config, expected_depth)
        original_mode = model.training
        self.model = model
        self.lora_config = config
        self.expected_depth = expected_depth
        self.target_modules = tuple(name for _, _, name in targets)

        # Restore learnable auxiliary parameters even for an inference-frozen
        # model; never wrap an entire transformer container in a freezing helper.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name not in _FIXED_PARAMETER_NAMES)
        model.train_image_feature_extractor = False
        model.image_embedder.requires_grad_(False)
        model.transformer.mid_level.requires_grad_(False)
        for block, attr, _ in targets:
            setattr(block, attr, LoRALinear(getattr(block, attr), config))
        self.train(original_mode)
        self.assert_training_policy()

    @staticmethod
    def _validate_model(
        model: nn.Module, config: LoRAConfig, expected_depth: int
    ) -> list[tuple[nn.Module, str, str]]:
        if not isinstance(config, LoRAConfig):
            raise TypeError("config must be a LoRAConfig")
        if isinstance(expected_depth, bool) or not isinstance(expected_depth, int) or expected_depth < 1:
            raise ValueError("expected_depth must be a positive integer")
        for attr in ("image_embedder", "transformer", "distribution_head"):
            if not isinstance(getattr(model, attr, None), nn.Module):
                raise ValueError(f"MYRIAD model must expose an nn.Module {attr}")
        if not hasattr(model, "train_image_feature_extractor"):
            raise ValueError("MYRIAD model must expose train_image_feature_extractor")
        blocks = getattr(model.transformer, "mid_level", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != expected_depth:
            raise ValueError(f"Expected {expected_depth} transformer.mid_level blocks")
        if any(isinstance(module, LoRALinear) for module in model.modules()):
            raise ValueError("Model already contains LoRA adapters; refusing double injection")
        targets = []
        # Preflight every projection before changing trainability or modules.
        for i, block in enumerate(blocks):
            for attr in ("in_proj", "out_proj"):
                layer = getattr(block, attr, None)
                name = f"{_BLOCK_PREFIX}{i}.{attr}"
                if not isinstance(layer, nn.Linear):
                    raise ValueError(f"{name} must be an nn.Linear")
                if not layer.weight.is_floating_point():
                    raise ValueError(f"{name} must have floating-point weights")
                if config.rank > min(layer.in_features, layer.out_features):
                    raise ValueError(f"LoRA rank {config.rank} exceeds a dimension of {name}")
                targets.append((block, attr, name))
        return targets

    @property
    def transformer(self) -> nn.Module:
        return self.model.transformer

    @property
    def image_embedder(self) -> nn.Module:
        return self.model.image_embedder

    @property
    def distribution_head(self) -> nn.Module:
        return self.model.distribution_head

    def train(self, mode: bool = True) -> FineTuningModel:
        super().train(mode)
        self.model.image_embedder.eval()
        # Validation sampling caches contain derived head embeddings. Invalidate
        # them after mode changes so updated head weights are used next time.
        head = getattr(self.distribution_head, "head", None)
        cache = getattr(head, "_timestep_emb_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # No no_grad context here: gradients must pass through frozen blocks
        # to their LoRA adapters and the normally trained upstream modules.
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def embed_image(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.embed_image(*args, **kwargs)

    @torch.no_grad()
    def predict_parallel(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.predict_parallel(*args, **kwargs)

    @torch.no_grad()
    def predict_simulate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.predict_simulate(*args, **kwargs)

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        for block in self.transformer.mid_level:
            for attr in ("in_proj", "out_proj"):
                layer = getattr(block, attr)
                yield layer.lora_A
                yield layer.lora_B

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def optimizer_groups(
        self,
        *,
        lora_lr: float,
        other_lr: float,
        weight_decay: float = 0.01,
        lora_weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Separate learning rates, excluding every frozen parameter."""
        for name, value in (("lora_lr", lora_lr), ("other_lr", other_lr)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("weight_decay", weight_decay), ("lora_weight_decay", lora_weight_decay)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.assert_training_policy()
        adapters = list(self.lora_parameters())
        adapter_ids = {id(p) for p in adapters}
        ordinary = [p for p in self.trainable_parameters() if id(p) not in adapter_ids]
        return [
            {"params": adapters, "lr": lora_lr, "weight_decay": lora_weight_decay, "name": "lora"},
            {"params": ordinary, "lr": other_lr, "weight_decay": weight_decay, "name": "ordinary"},
        ]

    def assert_training_policy(self) -> None:
        """Cheap structural checks; no data loading or optimization is involved."""
        adapter_ids = {id(p) for p in self.lora_parameters()}
        for name, parameter in self.model.named_parameters():
            frozen = (
                name.startswith("image_embedder.")
                or (name.startswith(_BLOCK_PREFIX) and id(parameter) not in adapter_ids)
                or name in _FIXED_PARAMETER_NAMES
            )
            if parameter.requires_grad == frozen:
                raise RuntimeError(f"Unexpected requires_grad={parameter.requires_grad} for {name}")
        if self.model.train_image_feature_extractor:
            raise RuntimeError("DINO image feature extraction must disable gradients")
        if any(module.training for module in self.image_embedder.modules()):
            raise RuntimeError("DINO image encoder must remain in evaluation mode")
        for block in self.transformer.mid_level:
            for attr in ("in_proj", "out_proj"):
                layer = getattr(block, attr)
                if not isinstance(layer, LoRALinear) or layer.config != self.lora_config:
                    raise RuntimeError("Every block projection must use the same LoRA configuration")

    def parameter_report(self) -> dict[str, Any]:
        """Counts and target names for training logs and checkpoint metadata."""
        adapter_ids = {id(p) for p in self.lora_parameters()}
        groups = {}
        for name, parameter in self.model.named_parameters():
            if id(parameter) in adapter_ids:
                group = "lora_adapters"
            elif name.startswith("image_embedder."):
                group = "image_encoder"
            elif name.startswith(_BLOCK_PREFIX):
                group = "original_transformer_blocks"
            elif name.startswith("distribution_head."):
                group = "flow_matching_head"
            else:
                group = "auxiliary_modules"
            counts = groups.setdefault(group, {"total": 0, "trainable": 0})
            counts["total"] += parameter.numel()
            if parameter.requires_grad:
                counts["trainable"] += parameter.numel()
        return {
            "lora_config": self.lora_config.to_dict(),
            "num_blocks": len(self.transformer.mid_level),
            "num_adapted_projections": len(self.target_modules),
            "target_modules": list(self.target_modules),
            "total_parameters": sum(c["total"] for c in groups.values()),
            "trainable_parameters": sum(c["trainable"] for c in groups.values()),
            "groups": groups,
        }
