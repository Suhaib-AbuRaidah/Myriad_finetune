"""Recommended second-loop prototype; uses only initial geometry and tracks."""
import cv2
import numpy as np

from ..data import Inputs
from .solver import fit

CONFIG = dict(position_strength=128., position_count=64, center_method="mean",
    symmetric=True, noise_reference_px=1.)


def estimate_motion(xyz, uv, valid, k, e, *, parent_points, child_points,
                    parent_normals, child_normals):
    """Return fitted states 2/3/4 using the original prototype's public interface.

    All geometry is WORLD-frame state-zero geometry. Tracks are source-image
    pixels, with shape [5,N,2]. The pivot represents an infinite axis line.
    No true axis, pivot, joint state, or future geometry is accepted.
    """
    xyz, uv, valid = np.asarray(xyz, float), np.asarray(uv, float), np.asarray(valid, bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (5, len(xyz), 2) or valid.shape != (5, len(xyz)):
        raise ValueError("Expected xyz [N,3], uv [5,N,2], valid [5,N]")
    if not np.isfinite(xyz).all() or not np.isfinite(uv[valid]).all():
        raise ValueError("Anchors and valid trajectory coordinates must be finite")
    data = Inputs(xyz, uv, valid, np.asarray(k, float), np.asarray(e, float),
        np.asarray(parent_points, float), np.asarray(child_points, float),
        np.asarray(parent_normals, float), np.asarray(child_normals, float), None)
    fitted = fit(data, CONFIG)
    if fitted is None:
        return None
    p = np.asarray(fitted["params"])
    transforms = []
    for angle in p[6:]:
        rotation = cv2.Rodrigues(p[:3]*angle)[0]
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = p[3:6]-rotation@p[3:6]
        transforms.append(transform)
    return dict(kind="revolute", axis=p[:3], pivot=p[3:6], angles=p[6:], states=[2, 3, 4],
        transforms=np.stack(transforms), fit=fitted)
