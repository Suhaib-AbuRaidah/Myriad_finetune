"""Tiny two-GPU collective check; does not load or train the pretrained model.

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    -m finetuning_pair.verification.check_cuda_parallel
"""
import json
from pathlib import Path
import torch
from torch import nn
from finetuning_pair.parallel import ParallelContext
from finetuning_pair.train import _resolve_device

ctx = ParallelContext.from_environment()
device = ctx.initialize('cuda', _resolve_device)
try:
    weight = nn.Parameter(torch.full((2,), float(ctx.rank+1),device=device))
    unused = nn.Parameter(torch.ones((),device=device))
    ctx.synchronize_initial_parameters([weight,unused])
    torch.testing.assert_close(weight,torch.ones_like(weight))
    total = ctx.sum(torch.tensor(1 if ctx.rank==0 else 3,device=device))
    loss = (ctx.rank+1)*weight.sum()/total
    loss.backward()
    ctx.synchronize_gradients([weight,unused])
    torch.testing.assert_close(weight.grad,torch.full_like(weight,.75))
    assert unused.grad is None
    torch.manual_seed(300+ctx.rank)
    states = ctx.checkpoint_data_state(dict(epoch=0,offset=2,dataset_size=10))
    expected = torch.rand(4,device=device)
    torch.manual_seed(999)
    ctx.restore_rank_rng(states)
    torch.testing.assert_close(torch.rand(4,device=device),expected)
    info = dict(rank=ctx.rank,device=str(device),gpu=torch.cuda.get_device_name(device),
                native_bf16=torch.cuda.is_bf16_supported(including_emulation=False),
                gradient_sum_correct=True,rank_cuda_rng_resume=True)
    print(json.dumps(info),flush=True)
    if ctx.primary:
        Path('finetuning_pair/verification/cuda_parallel_checks.json').write_text(json.dumps(info,indent=2)+'\n')
    ctx.barrier()
finally:
    ctx.close()
