"""Contact hypotheses from state-zero mesh samples; no articulation truth."""
import numpy as np
from scipy.spatial import cKDTree


def mesh_contact(parent, child, count=32, symmetric=False, parent_normals=None, child_normals=None,
                 normal_method="deduplicated"):
    if count < 8: raise ValueError("Contact patches need at least eight samples")
    parent, child = np.asarray(parent, float), np.asarray(child, float)
    parent_finite, child_finite = np.isfinite(parent).all(1), np.isfinite(child).all(1)
    parent, child = parent[parent_finite], child[child_finite]
    if parent_normals is not None: parent_normals = np.asarray(parent_normals)[parent_finite]
    if child_normals is not None: child_normals = np.asarray(child_normals)[child_finite]
    if min(len(parent), len(child)) < 8:
        return None
    distance, index = cKDTree(parent).query(child)
    chosen = np.argsort(distance, kind="stable")[:min(count, len(child))]
    midpoints = .5 * (child[chosen] + parent[index[chosen]])
    selected_parent, selected_child = index[chosen], chosen
    gaps = distance[chosen]
    if symmetric:
        reverse, reverse_index = cKDTree(child).query(parent)
        chosen_reverse = np.argsort(reverse, kind="stable")[:min(count, len(parent))]
        midpoints = np.concatenate([midpoints, .5*(parent[chosen_reverse]+child[reverse_index[chosen_reverse]])])
        gaps = np.r_[gaps, reverse[chosen_reverse]]
        selected_parent = np.r_[selected_parent, chosen_reverse]
        selected_child = np.r_[selected_child, reverse_index[chosen_reverse]]
    center = np.median(midpoints, axis=0)
    raw_covariance = np.cov((midpoints-center).T)
    values, axes = np.linalg.eigh(raw_covariance)
    values = np.maximum(values, 0.)
    cloud_scale = min(np.sqrt(np.mean(np.sum((p-p.mean(0))**2, axis=1))) for p in (parent, child))
    floor = max(.5*cloud_scale, float(np.median(gaps)), 1e-5)
    covariance = raw_covariance + floor**2*np.eye(3)
    whitening = (axes / np.sqrt(values+floor**2)[None]).T
    largest = max(values[2], 1e-15)
    result = dict(center=center, covariance=covariance, whitening=whitening,
        gap=float(np.min(gaps)), uncertainty_floor=floor, points=len(midpoints),
        eigenvalues=values, axes=axes, cloud_scale=cloud_scale,
        linearity=float((values[2]-values[1])/largest),
        planarity=float((values[1]-values[0])/largest),
        midpoints=midpoints)
    if parent_normals is not None and child_normals is not None:
        # Normal signs are irrelevant to the second moment. Deduplicate so a
        # sparse part's single nearest sample cannot dominate by repetition.
        normals = np.r_[parent_normals[np.unique(selected_parent)], child_normals[np.unique(selected_child)]]
        normals = normals[np.isfinite(normals).all(1) & (np.linalg.norm(normals, axis=1)>1e-8)]
        if len(normals)<8: return result
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        moment = normals.T@normals / len(normals)
        if normal_method in ("balanced", "paired"):
            # Keep one observation from each surface per nearest pair. Each
            # part has equal total mass, independent of mesh sampling density.
            pn, cn = parent_normals[selected_parent], child_normals[selected_child]
            finite = (np.isfinite(pn).all(1) & np.isfinite(cn).all(1)
                & (np.linalg.norm(pn, axis=1)>1e-8) & (np.linalg.norm(cn, axis=1)>1e-8))
            pn, cn = pn[finite], cn[finite]
            pn = pn/np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
            cn = cn/np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
            weights = np.ones(len(pn)) if normal_method == "balanced" else np.abs(np.sum(pn*cn, axis=1))**8
            if weights.sum()>1e-8:
                moment = ((pn.T*weights)@pn+(cn.T*weights)@cn)/(2*weights.sum())
        spectrum, basis = np.linalg.eigh(moment)
        result.update(normal_spectrum=spectrum, normal_axes=basis,
            normal_concentration=float((spectrum[2]-spectrum[1])/max(spectrum[2], 1e-15)),
            normal_cylindricity=float((spectrum[1]-spectrum[0])/max(spectrum[2], 1e-15)))
    return result


def direction_options(contact, mode):
    """Each option describes one sign-invariant quadratic axis prior."""
    axes = contact["axes"]
    line, plane = contact["linearity"], contact["planarity"]
    if mode == "normal_major": return [(contact["normal_axes"][:, 2], "align", 1.)]
    if mode == "normal_confident":
        return [(contact["normal_axes"][:, 2], "align", contact["normal_concentration"])]
    if mode == "normal_shape":
        index = 2 if contact["normal_concentration"] >= contact["normal_cylindricity"] else 0
        confidence = max(contact["normal_concentration"], contact["normal_cylindricity"])
        return [(contact["normal_axes"][:, index], "align", confidence)]
    if mode == "normal_field":
        values = contact["normal_spectrum"]
        weights = np.sqrt(np.maximum(0., (values[2]-values)/max(values[2], 1e-15)))
        return [(weights[:, None]*contact["normal_axes"].T, "subspace", 1.)]
    if mode == "ambiguous_mixture":
        if contact["normal_concentration"] >= .2: return direction_options(contact, "normal_field")
        return [(contact["normal_axes"][:, i], "align", 1.) for i in (2, 1, 0)]
    if mode in ("fused", "gated", "projected"):
        values = contact["normal_spectrum"]
        matrix = contact["normal_axes"]@np.diag(1-values/max(values[2], 1e-15))@contact["normal_axes"].T
        confidence = contact["normal_concentration"]
        index = 2 if line >= plane else 0
        target = axes[:, index]
        shape_confidence = max(line, plane)
        if mode == "gated":
            return [(target, "align", shape_confidence)] if confidence < .2 else direction_options(contact, "normal_field")
        if mode == "projected":
            if confidence >= .2: return direction_options(contact, "normal_field")
            normal_null = contact["normal_axes"][:, 0]
            target = target-normal_null*(normal_null@target)
            if np.linalg.norm(target) < .1: return direction_options(contact, "normal_field")
            return [(target/np.linalg.norm(target), "align", shape_confidence)]
        matrix += (1-confidence)*shape_confidence*(np.eye(3)-np.outer(target, target))
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Remove the axis-independent constant so the best-supported direction
        # has zero residual, including fully degenerate geometry.
        eigenvalues = np.maximum(eigenvalues-eigenvalues[0], 0.)
        return [(np.sqrt(eigenvalues)[:, None]*eigenvectors.T, "subspace", 1.)]
    if mode == "major": return [(axes[:, 2], "align", 1.)]
    if mode == "middle": return [(axes[:, 1], "align", 1.)]
    if mode == "normal": return [(axes[:, 0], "align", 1.)]
    if mode == "perp_major": return [(axes[:, 2], "perpendicular", 1.)]
    if mode == "shape":
        return [(axes[:, 2], "align", line)] if line >= plane else [(axes[:, 0], "align", plane)]
    if mode == "mixture":
        # The normal and the longest tangent represent physically distinct hinge geometries.
        return [(axes[:, 2], "align", 1.), (axes[:, 0], "align", 1.)]
    if mode == "all": return [(axes[:, i], "align", 1.) for i in (0, 1, 2)]
    raise ValueError(mode)
