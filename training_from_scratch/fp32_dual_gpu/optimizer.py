"""FP32 model parameters on one device, FP32 AdamW work on another."""
from __future__ import annotations

import torch

from training_from_scratch.precision import MasterAdamW


class DeviceAdamW(MasterAdamW):
    def __init__(self, parameters, *, device, **kwargs):
        parameters = list(parameters)
        if not parameters or any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("DeviceAdamW requires FP32 model parameters")
        devices = {p.device for p in parameters}
        if len(devices) != 1:
            raise ValueError("Keep the complete model on one device")
        self.model_device = next(iter(devices))
        self.optimizer_device = torch.device(device)
        super().__init__(parameters, device=self.optimizer_device, **kwargs)

    def _synchronize(self):
        for device in {self.model_device, self.optimizer_device}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def capture_model_gradients(self):
        super().capture_model_gradients()
        self._synchronize()

    @torch.no_grad()
    def step(self, closure=None):
        result = super().step(closure)
        # Complete Adam and all weight copies before the next forward pass.
        self._synchronize()
        return result

    def state_dict(self):
        state = super().state_dict()
        # Unlike FP16 model storage, the FP32 model copy contains every bit of
        # the optimizer weights. Save those weights once, in model.state_dict.
        del state["master_parameters"]
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        if "master_parameters" not in state:
            # The caller restores the FP32 model first, as with ordinary AdamW.
            state["master_parameters"] = [p.detach() for p in self.model_parameters]
        super().load_state_dict(state)
        self._synchronize()
