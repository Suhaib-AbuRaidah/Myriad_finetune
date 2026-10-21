"""Frozen recommended prototype; production fitting is deliberately untouched."""
import cv2
import numpy as np

from .data import Inputs
from .solver import fit

# Strength 256 improves development mean by 0.119 degrees but creates a 61.53
# degree error in a GT-track control; 128 keeps its maximum below 0.300 degrees.
CONFIG = dict(mode="ambiguous_mixture", strength=128., quadratic=True, starts=True,
    normal_method="paired", orientation_count=64, position_strength=4.)


def estimate_motion(xyz, uv, valid, k, e, *, parent_points, child_points,
                    parent_normals, child_normals):
    """Fit an assumed revolute joint using initial geometry and five-state tracks.

    xyz: [N,3] initial world anchors. uv: [5,N,2] source-image pixels.
    valid: [5,N] original correspondence mask. k: [3,3], e: [4,4] world-to-camera.
    Part points/normals are [M,3] WORLD arrays at the observed initial state only.
    Neither future geometry nor GT articulation parameters are accepted.
    Returned transforms correspond to states 2, 3, 4; pivot has a gauge and is
    intended to represent an infinite axis line, not a unique anatomical point.
    """
    xyz, uv = np.asarray(xyz, float), np.asarray(uv, float)
    valid = np.asarray(valid, bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (5, len(xyz), 2) or valid.shape != (5, len(xyz)):
        raise ValueError("Expected xyz [N,3], uv [5,N,2], valid [5,N]")
    if not np.isfinite(xyz).all() or not np.isfinite(uv[valid]).all():
        raise ValueError("Anchors and valid trajectory coordinates must be finite")
    data = Inputs(xyz, uv, valid, np.asarray(k, float), np.asarray(e, float),
        np.asarray(parent_points, float), np.asarray(child_points, float),
        np.asarray(parent_normals, float), np.asarray(child_normals, float), None)
    fitted = fit(data, CONFIG)
    if fitted is None: return None
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
