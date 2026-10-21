"""Initial-mask crops using the tested inference pixel-center convention."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from testing_crop_inference.geometry import SquareCrop


def mask_crop(parent, child, *, margin=.1, image_size=384):
    parent, child = np.asarray(parent, bool), np.asarray(child, bool)
    if parent.shape != child.shape or parent.ndim != 2 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Need matching masks and nonnegative finite padding")
    h, w = parent.shape
    y, x = np.nonzero(parent | child)
    if not len(x):
        raise ValueError("Empty initial parent/child union")
    extent = max(int(np.ptp(x))+1, int(np.ptp(y))+1)
    # A square may exceed the short source dimension on rectangular inputs.
    # Pad with edge RGB / false segmentation, preserving every visible pixel.
    side = min(max(h, w), max(8, extent+4, math.ceil(extent*(1+2*margin))))
    left = int(np.clip(math.floor((x.min()+x.max()+1-side)/2), min(0, w-side), max(0, w-side)))
    top = int(np.clip(math.floor((y.min()+y.max()+1-side)/2), min(0, h-side), max(0, h-side)))
    return SquareCrop(top, left, side, image_size)


def square_array(array, crop, *, mask=False):
    h, w = array.shape[:2]
    t, l, s = crop.top, crop.left, crop.side
    pads = [(max(0, -t), max(0, t+s-h)), (max(0, -l), max(0, l+s-w))]
    pads += [(0, 0)]*(array.ndim-2)
    padded = np.pad(array, pads, mode="constant" if mask else "edge")
    t, l = t+pads[0][0], l+pads[1][0]
    return np.ascontiguousarray(padded[t:t+s, l:l+s])


def image_tensor(rgb, top, left, side, size):
    part = square_array(rgb, SquareCrop(top, left, side, size))
    x = torch.from_numpy(part).permute(2, 0, 1)[None].float()/255
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                         antialias=True)[0].mul(2).sub(1).clamp(-1, 1)


def resize_mask(mask, crop):
    part = square_array(mask, crop, mask=True)
    return F.interpolate(torch.from_numpy(part)[None, None].float(),
                         size=(crop.image_size, crop.image_size), mode="nearest")[0, 0].bool()
