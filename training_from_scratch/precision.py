"""FP16 model storage with FP32 master weights and Adam state."""
from __future__ import annotations

import torch
from torch import nn


class MasterAdamW(torch.optim.AdamW):
    """Use ordinary AdamW on FP32 copies; round model weights after updates.

    GradScaler unscales the FP32 master gradients, avoiding its prohibition
    against unscaling FP16 leaf gradients. Adam's small updates accumulate in
    FP32 rather than disappearing when rounded to FP16 every step.
    """

    def __init__(self, parameters, *, device=None, **kwargs):
        self.model_parameters = list(parameters)
        masters = [nn.Parameter(p.detach().to(device=device or p.device, dtype=torch.float32, copy=True))
                   for p in self.model_parameters]
        super().__init__(masters, foreach=False, **kwargs)

    def optimizer_parameters(self):
        return [p for group in self.param_groups for p in group["params"]]

    def capture_model_gradients(self):
        for model, master in zip(self.model_parameters, self.optimizer_parameters()):
            master.grad = (model.grad.detach().to(device=master.device, dtype=torch.float32, copy=True)
                           if model.grad is not None else None)
            model.grad = None

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for parameter in self.model_parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for model, master in zip(self.model_parameters, self.optimizer_parameters()):
            model.copy_(master)
        return loss

    def state_dict(self):
        return {**super().state_dict(),
                "master_parameters": [p.detach() for p in self.optimizer_parameters()]}

    def load_state_dict(self, state_dict):
        saved = dict(state_dict)
        masters = saved.pop("master_parameters")
        current = self.optimizer_parameters()
        if len(masters) != len(current) or any(a.shape != b.shape for a, b in zip(masters, current)):
            raise ValueError("FP32 master parameter layout differs from checkpoint")
        super().load_state_dict(saved)
        with torch.no_grad():
            for master, value, model in zip(current, masters, self.model_parameters):
                master.copy_(value)
                model.copy_(master)
