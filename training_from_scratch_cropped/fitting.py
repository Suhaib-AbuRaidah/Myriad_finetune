"""Contact-restart estimator boundary: predicted tracks and INITIAL RGB-D only."""
import numpy as np
from testing_joint_fitting.candidate import estimate_motion


def fit_predictions(xyz, source_pixels, intrinsic, extrinsic, *, initial_depth,
                    initial_segmentation, pair_labels):
    # Correspondences come from the initial selected point identities and finite
    # predictions. Future GT validity/poses/labels do not enter this estimator.
    valid = np.isfinite(source_pixels).all(-1)
    if len(xyz) < 4 or valid[2:5].sum() < 8:
        return None
    result = estimate_motion(xyz, source_pixels, valid, intrinsic, extrinsic,
        initial_depth=initial_depth, initial_segmentation=initial_segmentation,
        pair_labels=pair_labels)
    if result is None:
        return None
    fitted = result["fit"]
    diag = fitted["diagnostics"]
    diag["contact_fallback"] = ("uncertainty_scaled_soft_axis_line_prior" if "contact" in diag
                                 else "trajectory_only_revolute_fit_no_visible_contact_evidence")
    # The revolute candidate remains present regardless of selected type.
    return fitted
