"""Contact-center experiments that leave the frozen orientation fitter intact."""
import numpy as np

from finetuning_pair.analyze_joints import fit_joint
from ..candidate import CONFIG as ORIENTATION_CONFIG
from ..geometry import mesh_contact, direction_options
from ..solver import fit as orientation_fit, solve


def position_contact(data, config):
    contact = mesh_contact(data.parent, data.child, config.get("position_count", 32),
        config.get("symmetric", False), data.parent_normals, data.child_normals)
    if contact is None:
        return None
    center_method = config.get("center_method", "median")
    if center_method == "mean":
        contact["center"] = contact["midpoints"].mean(0)
    elif center_method != "median":
        raise ValueError(f"Unknown center method: {center_method}")
    if config.get("isotropic", False):
        contact["whitening"] = np.eye(3)/contact["uncertainty_floor"]
    return contact


def fit(data, config, seed=None):
    """Fit from initial geometry and predicted tracks; accepts no scoring truth.

    Increasing position_strength softly pulls the *line* toward the contact
    center, without penalizing displacement of the pivot along that same line.
    The orientation prior, pixel loss and optimizer remain the frozen fitter's.
    """
    if config.get("baseline"):
        return orientation_fit(data, ORIENTATION_CONFIG, seed)
    if seed is None:
        base = fit_joint(data.xyz, data.uv[2:5], data.valid[2:5], data.k, data.e)
        seed = None if base is None else base["fits"].get("revolute")
    if seed is None:
        return None
    contact = position_contact(data, config)
    if contact is None:
        return orientation_fit(data, ORIENTATION_CONFIG, seed)
    orientation = mesh_contact(data.parent, data.child, 64, False,
        data.parent_normals, data.child_normals, "paired")
    if orientation is None or "normal_spectrum" not in orientation:
        return orientation_fit(data, ORIENTATION_CONFIG, seed)
    position_strength = config["position_strength"]
    sigma = max(.25, float(seed["median_px"])/np.sqrt(2*np.log(2)))
    noise_reference = config.get("noise_reference_px", 0.)
    if noise_reference:
        position_strength *= sigma/np.hypot(sigma, noise_reference)
    candidates = [solve(data, seed, contact, direction, 128., True, True,
        position_strength)
        for direction in direction_options(orientation, "ambiguous_mixture")]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    result = min(candidates, key=lambda c: c["diagnostics"]["objective"])
    result["diagnostics"].update(contact_center=contact["center"],
        contact_points=contact["points"], center_method=config.get("center_method", "median"),
        nominal_position_strength=config["position_strength"], noise_reference_px=noise_reference,
        axis_center_distance=float(np.linalg.norm(np.cross(
            result["params"][3:6]-contact["center"], result["params"][:3]))))
    return result


VARIANTS = {"orientation_baseline": dict(baseline=True)}
for strength in (4., 8., 16., 32., 64., 128.):
    VARIANTS[f"median32_soft{strength:g}"] = dict(position_strength=strength)
for count in (32, 64, 128, 256):
    for strength in (16., 64.):
        VARIANTS[f"mean{count}_soft{strength:g}"] = dict(position_strength=strength,
            position_count=count, center_method="mean")
for strength in (16., 64.):
    VARIANTS[f"symmetric_mean64_soft{strength:g}"] = dict(position_strength=strength,
        position_count=64, center_method="mean", symmetric=True)
for count, symmetric in ((32, False), (64, False), (64, True)):
    for strength in (32., 64., 128., 256.):
        name = f"{'symmetric_' if symmetric else ''}mean{count}_adaptive{strength:g}"
        VARIANTS[name] = dict(position_strength=strength, position_count=count,
            center_method="mean", symmetric=symmetric, noise_reference_px=1.)
