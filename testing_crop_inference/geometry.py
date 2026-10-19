"""Pixel-center transforms for mask crops; no future observations choose a crop."""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from finetuning_pair.conditioning import blend_roles, crop_geometry
from training_from_scratch.dataset import _image_tensor, _training_targets


@dataclass(frozen=True)
class SquareCrop:
    top: int
    left: int
    side: int
    image_size: int = 384

    @property
    def matrix(self):
        scale = self.image_size / self.side
        return np.array([[scale, 0, scale * (.5-self.left)-.5],
                         [0, scale, scale * (.5-self.top)-.5], [0, 0, 1.]])

    def normalize(self, source_pixels):
        return ((np.asarray(source_pixels)-[self.left, self.top]+.5)
                * self.image_size/self.side-.5)/(self.image_size-1)

    def unnormalize(self, positions):
        return ((np.asarray(positions)*(self.image_size-1)+.5)
                * self.side/self.image_size-.5+[self.left, self.top])


def mask_crop(parent, child, *, margin=.1, image_size=384):
    """Square union box with margin on each side, shifted inside the image.

    The original square center view is the maximum extent. Require all visible
    parent/child mask pixels to be inside it; this experiment must not silently
    remove portions of either mask on non-square source images.
    """
    parent, child = np.asarray(parent, bool), np.asarray(child, bool)
    if parent.shape != child.shape or parent.ndim != 2 or margin < 0 or not np.isfinite(margin):
        raise ValueError("Expected two equal masks and a finite nonnegative margin")
    h, w = parent.shape
    top, left, maximum = crop_geometry(h, w)
    yy, xx = np.nonzero(parent | child)
    if not len(xx):
        raise ValueError("Empty parent/child union")
    if xx.min() < left or xx.max() >= left+maximum or yy.min() < top or yy.max() >= top+maximum:
        raise ValueError("Visible pair extends beyond original center crop")
    extent = max(int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1))
    # Two pixels protect supplied boundary points under half-pixel resizing.
    side = min(maximum, max(extent+4, math.ceil(extent*(1+2*margin))))
    crop_left = int(np.clip(math.floor((xx.min()+xx.max()+1-side)/2), left, left+maximum-side))
    crop_top = int(np.clip(math.floor((yy.min()+yy.max()+1-side)/2), top, top+maximum-side))
    return SquareCrop(crop_top, crop_left, side, image_size)


def resize_mask(mask, crop):
    part = np.ascontiguousarray(mask[crop.top:crop.top+crop.side, crop.left:crop.left+crop.side])
    return F.interpolate(torch.from_numpy(part).float()[None, None],
                         size=(crop.image_size, crop.image_size), mode="nearest")[0, 0].bool()


def crop_sample(sample, source, crop):
    """Rebuild RGB, overlays, masks and targets; retain world/depth references.

    Future validity remains the original-image validity for paired scoring.
    A separate crop_position_valid exposes whether a valid point exits the new
    view. Targets and predictions are never clipped or dropped at crop borders.
    """
    h, w = source["initial_depth"].shape
    original = SquareCrop(*crop_geometry(h, w), crop.image_size)
    result = dict(sample)
    for key in ("trajectories", "actual_trajectories"):
        source_pixels = original.unnormalize(sample[key].numpy())
        result[key] = torch.from_numpy(crop.normalize(source_pixels).astype(np.float32))
    parent_label, child_label = sample["pair_labels"].tolist()
    seg, rgb = source["initial_segmentation"], source["initial_rgb"]
    parent, child = seg == parent_label, seg == child_label
    result["x"] = _image_tensor(blend_roles(rgb, parent, child, sample["overlay_alpha"]),
                                crop.top, crop.left, crop.side, crop.image_size)
    result["original_x"] = _image_tensor(rgb, crop.top, crop.left, crop.side, crop.image_size)
    result["parent_mask"], result["child_mask"] = resize_mask(parent, crop), resize_mask(child, crop)
    p = result["trajectories"]
    result["crop_position_valid"] = sample["position_valid"] & ((p >= 0) & (p <= 1)).all(-1)
    result["camera_intrinsic_model"] = torch.from_numpy(crop.matrix @ source["camera_intrinsic"])
    result["reference_pixels_model"] = torch.from_numpy(
        (crop.normalize(sample["reference_pixels"].numpy())*(crop.image_size-1)).astype(np.float32))
    result.update(_training_targets(result))
    return result


def restore_predictions(predictions, crop, original):
    return original.normalize(crop.unnormalize(predictions))
