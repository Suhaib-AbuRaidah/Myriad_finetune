"""Persistent data-parallel replicas launched with torchrun.

MYRIAD computes its distribution loss after forward and changes the FM head's
activation dtype at runtime. Explicit gradient synchronization preserves that
existing path without imposing fixed reducer buckets on the pretrained model.
Each rank sums its valid-token loss divided by the global valid-token count;
gradient SUM therefore equals the original loss on the combined GPU batch.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Iterable
import torch
from torch import nn
import torch.distributed as dist


@dataclass(frozen=True)
class ParallelContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @classmethod
    def from_environment(cls):
        return cls(rank=int(os.environ.get('RANK', '0')),
                   world_size=int(os.environ.get('WORLD_SIZE', '1')),
                   local_rank=int(os.environ.get('LOCAL_RANK', '0')))

    @property
    def enabled(self): return self.world_size > 1
    @property
    def primary(self): return self.rank == 0

    def initialize(self, device_setting, resolve_device):
        if self.enabled:
            if device_setting not in ('auto', 'cuda', 'cpu'):
                raise ValueError('With torchrun use --device cuda/auto or cpu; LOCAL_RANK selects each GPU')
            device = resolve_device(device_setting)
            if device.type == 'cuda':
                if self.local_rank >= torch.cuda.device_count():
                    raise ValueError('Not enough visible CUDA devices for this torchrun launch')
                device = torch.device('cuda', self.local_rank)
                torch.cuda.set_device(device)
            dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                    timeout=timedelta(hours=2),
                                    device_id=device if device.type == 'cuda' else None)
        else:
            device = resolve_device(device_setting)
            if device.type == 'cuda':
                if device.index is None:
                    device = torch.device('cuda', self.local_rank)
                torch.cuda.set_device(device)
        return device

    def barrier(self):
        if self.enabled: dist.barrier()

    def sum(self, value):
        result = value.detach().clone()
        if self.enabled: dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def primary_flag(self, value, device):
        if not self.enabled: return value
        tensor = torch.tensor(int(value), device=device)
        dist.broadcast(tensor, src=0)
        return bool(tensor.item())

    def local_indices(self, global_indices):
        """Shard the shared cursor; an empty final shard gets a zero-loss dummy."""
        if not global_indices: raise ValueError('Global training batch must not be empty')
        selected = global_indices[self.rank::self.world_size]
        return (selected, False) if selected else ([global_indices[0]], True)

    def synchronize_initial_parameters(self, parameters: Iterable[nn.Parameter]):
        if self.enabled:
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)

    def synchronize_gradients(self, parameters: Iterable[nn.Parameter], bucket_bytes=16*1024*1024):
        if not self.enabled: return
        parameters = list(parameters)
        if not parameters: return
        # Preserve None for parameters unused on every rank: AdamW must not
        # apply weight decay to globally unused parameters.
        used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32,
                            device=parameters[0].device)
        dist.all_reduce(used, op=dist.ReduceOp.SUM)
        groups = {}
        for parameter, count in zip(parameters, used.tolist()):
            if count:
                groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
        for group in groups.values():
            bucket, size = [], 0
            for parameter in group:
                needed = parameter.numel() * parameter.element_size()
                if bucket and size + needed > bucket_bytes:
                    self._reduce_bucket(bucket)
                    bucket, size = [], 0
                bucket.append(parameter)
                size += needed
            if bucket: self._reduce_bucket(bucket)

    @staticmethod
    def _reduce_bucket(parameters):
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in parameters:
            gradient = flat[offset:offset+parameter.numel()].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
            offset += parameter.numel()

    def checkpoint_data_state(self, cursor_state):
        from finetuning_pair.checkpoint_io import capture_rng_state
        if not self.enabled: return cursor_state
        rank_rng = capture_rng_state()
        states = [None] * self.world_size
        dist.all_gather_object(states, rank_rng)
        return dict(cursor_state, world_size=self.world_size, rank_rng_states=states)

    def restore_rank_rng(self, data_state):
        from finetuning_pair.checkpoint_io import restore_rng_state
        if not self.enabled: return
        if data_state.get('world_size') != self.world_size or len(data_state.get('rank_rng_states', [])) != self.world_size:
            raise ValueError('Resume requires RNG states for the same number of ranks')
        restore_rng_state(data_state['rank_rng_states'][self.rank])

    def close(self):
        if self.enabled and dist.is_initialized(): dist.destroy_process_group()
