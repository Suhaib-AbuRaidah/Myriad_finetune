"""Shared training/inference role overlays and pixel-center crop geometry."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

PARENT_COLOR = np.array([0, 120, 255], dtype=np.float32)
CHILD_COLOR = np.array([255, 145, 0], dtype=np.float32)


def blend_roles(rgb, parent_mask, child_mask, alpha=0.85):
    """Apply the specified alpha convention only to visible role masks.

    Float output avoids quantizing the retained texture before model resizing.
    Disjoint masks are required, since visible segmentation has one role/pixel.
    """
    rgb = np.asarray(rgb)
    parent, child = np.asarray(parent_mask, dtype=bool), np.asarray(child_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or parent.shape != rgb.shape[:2] or child.shape != parent.shape:
        raise ValueError('Expected RGB [H,W,3] and two masks [H,W]')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('overlay alpha must be finite and in [0,1]')
    if np.any(parent & child):
        raise ValueError('Visible parent and child masks must not overlap')
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError('RGB values must be finite and in [0,255]')
    output = rgb.astype(np.float32, copy=True)
    for mask, color in ((parent, PARENT_COLOR), (child, CHILD_COLOR)):
        output[mask] = (1 - alpha) * output[mask] + alpha * color
    return output


def crop_geometry(height, width):
    side = min(height, width)
    return round((height - side) / 2), round((width - side) / 2), side


def normalized_pixels(pixels, height, width, image_size):
    top, left, side = crop_geometry(height, width)
    return ((np.asarray(pixels) - [left, top] + 0.5) * image_size / side - 0.5) / (image_size - 1)


def model_mask(mask, image_size):
    height, width = mask.shape
    top, left, side = crop_geometry(height, width)
    tensor = torch.from_numpy(np.ascontiguousarray(mask[top:top+side, left:left+side])).float()[None, None]
    return F.interpolate(tensor, size=(image_size, image_size), mode='nearest')[0, 0].bool()
