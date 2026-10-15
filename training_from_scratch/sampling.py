"""Sampler attention that works without compiling a FlexAttention kernel."""
from __future__ import annotations

from contextlib import nullcontext
import sys
from unittest.mock import patch

from torch.nn.attention.flex_attention import create_mask

from finetuning_pair.sampling import prepare_sampling_cache


def dense_sampler_mask(mask_mod, B, H, Q_LEN, KV_LEN, device="cuda", **kwargs):
    """Evaluate MYRIAD's exact token predicate as a boolean SDPA mask.

    BlockMask.to_dense() describes block occupancy and is not an equivalent
    token mask. create_mask evaluates the original causal predicate directly,
    including the image-prefix rows that must attend to nothing.
    """
    return create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=device)


def sampler_attention_context():
    """Route sampler prefill through the existing PyTorch SDPA branch.

    Incremental decoding already supplies boolean masks. Patching only mask
    construction keeps attention scaling, KV caching and the original model
    code intact. Nested validation/model contexts restore their caller's state.
    """
    original_module = sys.modules.get("myriad.model")
    if original_module is None:
        return nullcontext()
    return patch.object(original_module, "create_block_mask", dense_sampler_mask)
