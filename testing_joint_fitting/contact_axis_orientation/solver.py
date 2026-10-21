"""Jointly fit axis, pivot and angles using soft, sign-invariant geometry priors."""
import math
import cv2
import numpy as np
from scipy.optimize import least_squares

from finetuning.joint_parame_estim.run import unit, project
from finetuning_pair.analyze_joints import fit_joint
from testing_joint_fitting.geometric import refine_revolute, tangent_frame, huber_cost
from .geometry import mesh_contact, direction_options


def solve(data, initial, contact, direction=None, strength=0., starts=False, quadratic=False,
          position_strength=1.):
    """Same minimal coordinates/soft_l1 objective as the previous position fitter.

    All starts/candidates share the original seed's residual scale. Orientation
    residual is sigma*strength*cross(axis,direction), or sigma*strength*dot for
    perpendicularity. Thus direction signs have no effect on the objective.
    """
    if initial is None:
        return None
    xyz, uv, valid, k, e = data.xyz, data.uv[2:5], data.valid[2:5], data.k, data.e
    rows, cols = np.where(valid)
    if len(rows) < 8: return initial
    p = np.asarray(initial["params"], float)
    anchor = xyz.mean(0)
    scale = max(float(np.sqrt(np.mean(np.sum((xyz-anchor)**2, axis=1)))), 1e-5)
    axis0 = unit(p[:3]); frame = tangent_frame(axis0)
    sigma = max(.25, float(initial["median_px"])/math.sqrt(2*math.log(2)))

    def unpack(x):
        rotation = cv2.Rodrigues(frame@x[:2])[0]
        axis = rotation@axis0
        pivot = anchor+scale*(rotation@frame)@x[2:4]
        return axis, pivot, x[4:]

    def points_for(x):
        axis, pivot, angles = unpack(x)
        relative = xyz-pivot
        cross = np.cross(axis, relative)
        return (xyz[None]+np.sin(angles)[:, None, None]*cross
                +(1-np.cos(angles))[:, None, None]*np.cross(axis, cross))

    def components(x):
        points = points_for(x)
        image = (np.stack([project(a, k, e) for a in points])-uv)[rows, cols]
        axis, pivot, _ = unpack(x)
        direction_w = contact["whitening"]@axis
        offset_w = contact["whitening"]@(pivot-contact["center"])
        position = position_strength*sigma*(offset_w-direction_w*(direction_w@offset_w)/(direction_w@direction_w))
        if direction is None or strength == 0:
            orientation = np.empty(0)
        else:
            target, relation, confidence = direction
            residual = (np.cross(axis, target) if relation == "align" else
                target@axis if relation == "subspace" else np.atleast_1d(axis@target))
            orientation = sigma*strength*np.sqrt(confidence)*residual
        depth = (points@e[:3, :3].T+e[:3, 3])[rows, cols, 2]
        return image, position, orientation, depth

    def errors(x):
        image, position, orientation, depth = components(x)
        return np.r_[image.ravel(), position, orientation, np.minimum(depth/scale, 0)*100]

    def mixed_loss(z):
        # Quadratic directional prior must not be treated as a pixel outlier.
        # Its squared vector norm is also independent of the world basis.
        root = np.sqrt(1+z)
        rho = np.stack([2*(root-1), 1/root, -.5/root**3])
        begin = 2*len(rows)+3
        end = begin+(3 if direction is not None and direction[1] in ("align", "subspace") else 1)
        if direction is not None and strength:
            rho[0, begin:end] = z[begin:end]
            rho[1, begin:end] = 1
            rho[2, begin:end] = 0
        return rho

    x0 = np.r_[np.zeros(2), frame.T@(p[3:6]-anchor)/scale, p[6:]]
    x_contact = x0.copy(); x_contact[2:4] = frame.T@(contact["center"]-anchor)/scale
    seeds = [x0, x_contact]
    if starts and direction is not None and direction[1] in ("align", "subspace"):
        target = (unit(direction[0]) if direction[1] == "align" else
            np.linalg.eigh(direction[0].T@direction[0])[1][:, 0])
        if target@axis0 < 0: target = -target
        cross = np.cross(axis0, target)
        angle = np.arctan2(np.linalg.norm(cross), np.clip(axis0@target, -1., 1.))
        rotation_vector = unit(cross)*angle if np.linalg.norm(cross)>1e-12 else np.zeros(3)
        x = x0.copy(); x[:2] = frame.T@rotation_vector
        rotation = cv2.Rodrigues(frame@x[:2])[0]
        x[2:4] = (rotation@frame).T@(contact["center"]-anchor)/scale
        seeds.append(x)
    results = []
    for seed in seeds:
        fitted = least_squares(errors, seed, loss=mixed_loss if quadratic else "soft_l1", f_scale=8., x_scale="jac",
            max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        image, position, orientation, depth = components(fitted.x)
        if np.isfinite(fitted.x).all() and np.all(depth > 0):
            results.append((fitted.cost, fitted, image, position, orientation))
    if not results: return None
    cost, fitted, image, position, orientation = min(results, key=lambda item: item[0])
    axis, pivot, angles = unpack(fitted.x)
    norms = np.linalg.norm(image, axis=1)
    return dict(params=np.r_[axis, pivot, angles], rmse_px=float(np.sqrt(np.mean(norms**2))),
        median_px=float(np.median(norms)), sse=huber_cost(norms), n=len(norms),
        diagnostics=dict(objective=float(cost), sigma=sigma, evaluations=fitted.nfev,
            converged=bool(fitted.success), position_prior_norm=float(np.linalg.norm(position)),
            orientation_prior_norm=float(np.linalg.norm(orientation)), strength=strength,
            position_strength=position_strength,
            direction=None if direction is None else direction[0],
            relation=None if direction is None else direction[1],
            confidence=None if direction is None else direction[2]))


VARIANTS = {
    "original": dict(baseline="original"),
    "rgbd_position": dict(baseline="rgbd"),
    "mesh_position": dict(baseline="mesh"),
}
for mode in ("major", "middle", "normal", "perp_major", "shape", "mixture", "all"):
    for strength in (4., 16.):
        VARIANTS[f"{mode}_{strength:g}"] = dict(mode=mode, strength=strength)
for mode in ("major", "shape", "mixture"):
    for strength in (4., 16.):
        VARIANTS[f"quad_{mode}_{strength:g}"] = dict(mode=mode, strength=strength, quadratic=True, starts=True)
VARIANTS["mesh_solver_zero"] = dict(mode="major", strength=0.)
for mode in ("normal_major", "normal_confident", "normal_shape", "normal_field"):
    for strength in (4., 16., 64.):
        VARIANTS[f"surface_{mode}_{strength:g}"] = dict(mode=mode, strength=strength,
            quadratic=True, starts=True)
for method, count in (("balanced", 32), ("balanced", 64), ("paired", 64)):
    for mode in ("normal_major", "normal_field"):
        name = f"{method}{count}_{mode}_64"
        VARIANTS[name] = dict(mode=mode, strength=64., quadratic=True, starts=True,
            normal_method=method, orientation_count=count)
        for position in (2., 4.):
            VARIANTS[f"{name}_position{position:g}"] = dict(VARIANTS[name], position_strength=position)
for mode in ("fused", "gated", "projected"):
    VARIANTS[mode+"_64"] = dict(mode=mode, strength=64., quadratic=True, starts=True,
        normal_method="paired", orientation_count=64, position_strength=4.)
VARIANTS["ambiguous_mixture_64"] = dict(mode="ambiguous_mixture", strength=64., quadratic=True,
    starts=True, normal_method="paired", orientation_count=64, position_strength=4.)
VARIANTS["multiscale_64"] = dict(mode="normal_field", strength=64., quadratic=True,
    starts=True, normal_method="paired", orientation_count=64, position_strength=4., multiscale=True)
for count in (32, 128):
    VARIANTS[f"paired{count}_normal_field_64_position4"] = dict(mode="normal_field", strength=64.,
        quadratic=True, starts=True, normal_method="paired", orientation_count=count, position_strength=4.)
VARIANTS["ambiguous_mixture_count128_64"] = dict(VARIANTS["ambiguous_mixture_64"], orientation_count=128)
for strength in (32., 128., 256.):
    VARIANTS[f"ambiguous_mixture_strength{strength:g}"] = dict(VARIANTS["ambiguous_mixture_64"], strength=strength)


def fit(data, config, seed=None):
    if seed is None:
        base = fit_joint(data.xyz, data.uv[2:5], data.valid[2:5], data.k, data.e)
        seed = None if base is None else base["fits"].get("revolute")
    if seed is None: return None
    baseline = config.get("baseline")
    if baseline == "original": return seed
    contact = data.rgbd_contact if baseline == "rgbd" else mesh_contact(
        data.parent, data.child, config.get("count", 32), config.get("symmetric", False),
        data.parent_normals, data.child_normals)
    if contact is None: return seed
    if baseline in ("rgbd", "mesh"):
        return refine_revolute(seed, data.xyz, data.uv[2:5], data.valid[2:5], data.k, data.e,
            contact=contact, contact_start=True)
    orientation_contact = mesh_contact(data.parent, data.child, config.get("orientation_count", config.get("count", 32)),
        config.get("symmetric", False), data.parent_normals, data.child_normals,
        config.get("normal_method", "deduplicated"))
    if "normal_spectrum" not in orientation_contact and config["mode"] not in (
            "major", "middle", "normal", "perp_major", "shape", "mixture", "all"):
        result = refine_revolute(seed, data.xyz, data.uv[2:5], data.valid[2:5], data.k, data.e,
            contact=contact, contact_start=True)
        result["diagnostics"] = dict(fallback="mesh position only: insufficient valid contact normals")
        return result
    if config.get("multiscale"):
        moment = np.zeros((3, 3)); total = 0.
        for count in (16, 32, 64, 128):
            patch = mesh_contact(data.parent, data.child, count, False, data.parent_normals,
                data.child_normals, config.get("normal_method", "paired"))
            if "normal_spectrum" not in patch: continue
            confidence = patch["normal_concentration"]**2
            axis = patch["normal_axes"][:, 2]
            moment += confidence*np.outer(axis, axis); total += confidence
        if total>1e-8:
            values, axes = np.linalg.eigh(moment/total)
            orientation_contact.update(normal_spectrum=values, normal_axes=axes,
                normal_concentration=float((values[2]-values[1])/max(values[2], 1e-15)))
    candidates = [solve(data, seed, contact, direction, config["strength"], config.get("starts", False),
            config.get("quadratic", False), config.get("position_strength", 1.))
        for direction in direction_options(orientation_contact, config["mode"])]
    candidates = [c for c in candidates if c is not None]
    return min(candidates, key=lambda c: c["diagnostics"]["objective"]) if candidates else None
