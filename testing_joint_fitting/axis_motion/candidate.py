"""Frozen development candidate; no production replacement is installed."""
from __future__ import annotations

import cv2
import numpy as np

from testing_joint_fitting.geometric import contact_evidence
from .common import FitInput,lengths
from .refinement import fit_variant

VARIANT="frame_length_contact"


def estimate_revolute_motion(xyz,trajectories_source_pixels,valid,camera_intrinsic,camera_extrinsic,
                            *,initial_depth,initial_segmentation,pair_labels,image_size=384):
    """Assume a revolute joint; return world transforms for states 1--4.

    Only initial RGB-D, calibration, pair identities, predicted tracks and
    correspondence validity are accepted. Pixel coordinates refer to the
    original camera image, not [0,1] normalized or resized model coordinates.
    The parent stays at its initial pose, matching this checkpoint's target.
    Validity must be provided by tracking in deployment; the offline benchmark
    retains its existing GT geometric-validity masks for fair comparisons.
    The axis point is gauge-dependent along its line, not an identified pivot.
    """
    xyz=np.asarray(xyz,float);uv=np.asarray(trajectories_source_pixels,float);valid=np.asarray(valid,bool)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or uv.shape!=(5,len(xyz),2) or valid.shape!=(5,len(xyz)):
        raise ValueError("Expected world points [N,3], tracks [5,N,2], validity [5,N]")
    if image_size<2 or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Need finite points and valid trajectory observations")
    k,e=np.asarray(camera_intrinsic,float),np.asarray(camera_extrinsic,float)
    contact=contact_evidence(dict(depth=np.asarray(initial_depth),segmentation=np.asarray(initial_segmentation),
                                  pair_labels=pair_labels),k,e)
    data=FitInput(xyz,uv,valid,k,e,contact,min(np.asarray(initial_depth).shape)/image_size)
    fit=fit_variant(data,VARIANT)
    if fit is None:return None
    axis,pivot=fit["params"][:3],fit["params"][3:6]
    transforms=[]
    for angle in fit["params"][6:]:
        rotation=cv2.Rodrigues(axis*angle)[0]
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=pivot-rotation@pivot
        transforms.append(transform)
    motion=lengths(uv)
    return dict(assumed_kind="revolute",axis_world=axis,axis_point_world=pivot,
                states=fit["states"],transforms_world=np.stack(transforms),fit=fit,
                median_path_model_pixels=float(np.median(motion["path"])/data.source_per_model_pixel),
                median_extent_model_pixels=float(np.median(motion["extent"])/data.source_per_model_pixel))
