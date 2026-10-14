"""Precision handling for MYRIAD's transient autoregressive caches."""
from __future__ import annotations
import torch


def prepare_sampling_cache(model):
    """Match K/V buffers to the attention projections before sampler prefill.

    The pretrained implementation initializes these nonpersistent buffers as
    BF16 regardless of the run's precision. Float32 inference must use float32
    caches; otherwise assigning projected keys silently rounds them to BF16.
    Empty changed-precision caches let the original grow_kv_cache allocate the
    required capacity without casting an old rollout's contents.
    """
    transformer = getattr(model, 'transformer', None)
    layers = getattr(transformer, 'mid_level', ())
    for layer in layers:
        cache = getattr(layer, 'kv_cache', None)
        if cache is None:
            continue
        weight = layer.in_proj.weight
        device = weight.device
        dtype = (torch.get_autocast_dtype(device.type)
                 if torch.is_autocast_enabled(device.type) else weight.dtype)
        for name in ('k', 'v'):
            buffer = getattr(cache, name)
            if buffer.dtype != dtype or buffer.device != device:
                shape = list(buffer.shape)
                shape[2] = 0
                setattr(cache, name, torch.empty(shape, dtype=dtype, device=device))
