"""Ground-truth relative trajectories; all operations precede projection."""
from __future__ import annotations
import numpy as np

TARGET_DEFINITION = 'child_relative_frozen_initial_parent_camera_v1'


def relative_camera_points(local_points, parent_world, child_world, camera_extrinsic):
    """Return [states,points,4] in the fixed initial camera frame.

    Dataset T_world_link maps link-local coordinates to world. E=T_camera_world
    maps world into the fixed +Z camera, hence G_link(t)=E @ T_world_link(t).
    A child surface point is fixed in its child-local frame. Applying G_c(t)
    is equivalent to G_c(t) @ inv(G_c(0)) @ X_camera(0).
    """
    parent_world = np.asarray(parent_world, dtype=np.float64)
    child_world = np.asarray(child_world, dtype=np.float64)
    e = np.asarray(camera_extrinsic, dtype=np.float64)
    local = np.asarray(local_points, dtype=np.float64)
    if parent_world.shape != child_world.shape or parent_world.ndim != 3 or parent_world.shape[1:] != (4, 4):
        raise ValueError('Parent/child transforms must be [states,4,4]')
    if e.shape != (4, 4) or local.ndim != 2 or local.shape[1] != 3:
        raise ValueError('Expected camera [4,4] and local points [N,3]')
    local_h = np.column_stack((local, np.ones(len(local))))
    gp, gc = e @ parent_world, e @ child_world
    actual_camera = np.einsum('sij,nj->sni', gc, local_h)
    frozen_parent = gp[0] @ np.linalg.inv(gp)
    relative = np.einsum('sij,snj->sni', frozen_parent, actual_camera)
    return relative, actual_camera


def project_camera_points(camera_h, intrinsic):
    camera_h = np.asarray(camera_h, dtype=np.float64)
    camera = camera_h[..., :3] / camera_h[..., 3:4]
    z = camera[..., 2]
    projected = camera @ np.asarray(intrinsic, dtype=np.float64).T
    front = np.isfinite(projected).all(-1) & np.isfinite(z) & (z > 1e-8)
    denominator = np.where(front, projected[..., 2], 1.0)
    pixels = projected[..., :2] / denominator[..., None]
    return np.where(front[..., None], pixels, 0.0), z, front
