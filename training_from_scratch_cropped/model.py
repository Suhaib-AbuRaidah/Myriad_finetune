"""Reuse the full FP32 motion model and original flow-matching objective."""
from training_from_scratch.fp32_dual_gpu.model import CachedFP32Model as BaseModel
from training_from_scratch.fp32_dual_gpu.model import build_cached_myriad, training_loss


class CachedFP32Model(BaseModel):
    def configure_cropped_cache(self, cache):
        if not cache.require_cached:
            raise ValueError("Training requires verified precomputed features")
        self.feature_cache = cache
