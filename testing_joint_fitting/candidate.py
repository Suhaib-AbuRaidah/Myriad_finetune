"""Isolated contact-constrained candidate. Not installed in training/evaluation."""
from __future__ import annotations

import cv2
import numpy as np

from .conditioning import fit_conditioned


def estimate_motion(xyz, trajectories_source_pixels, valid, camera_intrinsic,
                    camera_extrinsic, *, initial_depth, initial_segmentation, pair_labels):
    """Return world-space transforms at states 2--4 and uncalibrated diagnostics.

    Inputs: initial world points [N,3], one trajectory [5,N,2] in source-image
    pixels, a correspondence mask [5,N], initial RGB-D geometry and pair labels.
    No GT joint parameters or future depth/poses are accepted. The current
    benchmark supplies its existing GT geometric-validity masks; a deployment
    caller must supply correspondence validity through its own tracking system.
    The parent is frozen at its initial pose, matching this Myriad checkpoint.
    """
    context=dict(depth=initial_depth,segmentation=initial_segmentation,pair_labels=pair_labels)
    result=fit_conditioned(np.asarray(xyz),np.asarray(trajectories_source_pixels),np.asarray(valid,bool),
        np.asarray(camera_intrinsic),np.asarray(camera_extrinsic),variant="contact_restart",context=context)
    if result is None:
        return None
    kind=result["kind"];parameters=result["fits"][kind]["params"]
    matrices=[]
    for state in range(3):
        transform=np.eye(4)
        if kind=="revolute":
            r=cv2.Rodrigues(parameters[:3]*parameters[6+state])[0]
            transform[:3,:3]=r
            transform[:3,3]=parameters[3:6]-r@parameters[3:6]
        elif kind=="prismatic":
            transform[:3,3]=parameters[:3]*parameters[3+state]
        matrices.append(transform)
    return dict(fit=result,kind=kind,states=np.array([2,3,4]),transforms_world=np.stack(matrices))
