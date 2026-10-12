"""Estimate articulated joints from saved MYRIAD 2D tracks and reference RGB-D.

Only predicted 2D coordinates enter the reprojection targets; saved dataset
position_valid masks select correspondences. Joint types, axes, and origins
are used only for scoring. Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from finetuning.dataset import ArticulatedDataset, _backproject


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATES = np.array([2, 3, 4])  # both modes have model-generated predictions here


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def project(points, k, e):
    camera = points @ e[:3, :3].T + e[:3, 3]
    z = np.maximum(camera[:, 2], 1e-5)
    return (camera @ k.T)[:, :2] / z[:, None]


def from_normalized(tracks, height, width, image_size=384):
    side = min(height, width)
    offset = np.array([round((width-side)/2), round((height-side)/2)])
    return ((tracks * (image_size-1) + .5) * side / image_size - .5) + offset


def estimate_pose(world, image, valid, k, e):
    """Reference-world to current-world rigid pose from calibrated PnP."""
    ok = valid & np.isfinite(image).all(1)
    if ok.sum() < 4:
        return np.eye(3), np.zeros(3), False
    xyz = np.ascontiguousarray(world[ok], dtype=np.float64)
    uv = np.ascontiguousarray(image[ok], dtype=np.float64)
    # Reference camera extrinsic is a close starting pose for short motions.
    r0, _ = cv2.Rodrigues(e[:3, :3])
    try:
        success, rv, tv = cv2.solvePnP(xyz, uv, k, None, r0, e[:3, 3].copy(), True,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return np.eye(3), np.zeros(3), False
        rv, tv = cv2.solvePnPRefineLM(xyz, uv, k, None, rv, tv)
        rc, _ = cv2.Rodrigues(rv)
        rw = e[:3, :3].T @ rc
        tw = e[:3, :3].T @ (np.asarray(tv).reshape(3) - e[:3, 3])
        if not np.isfinite(rw).all() or not np.isfinite(tw).all():
            return np.eye(3), np.zeros(3), False
        return rw, tw, True
    except cv2.error:
        return np.eye(3), np.zeros(3), False


def initial_axis(child_pose, parent_pose, anchor):
    rotations, translations, axes, angles = [], [], [], []
    for (rc, tc, goodc), (rp, tp, goodp) in zip(child_pose, parent_pose):
        if not (goodc and goodp):
            continue
        r = rp.T @ rc
        t = rp.T @ (tc-tp)
        rv, _ = cv2.Rodrigues(r)
        angle = np.linalg.norm(rv)
        if angle > .005 and angle < 2.8:
            rotations.append(r)
            translations.append(t)
            axes.append(rv[:, 0] / angle)
            angles.append(angle)
    if not axes:
        return np.array([0., 0., 1.]), anchor.copy(), np.zeros(len(child_pose))
    direction = axes[0]
    signed = np.array([a if a @ direction >= 0 else -a for a in axes])
    weights = np.asarray(angles)
    axis = unit(np.average(signed, axis=0, weights=weights))
    angles_out = []
    matrix, rhs = [], []
    for (rc, tc, goodc), (rp, tp, goodp) in zip(child_pose, parent_pose):
        if goodc and goodp:
            r = rp.T @ rc
            t = rp.T @ (tc-tp)
            rv, _ = cv2.Rodrigues(r)
            angles_out.append(float(rv[:, 0] @ axis))
            matrix.append(np.eye(3)-r)
            rhs.append(t)
        else:
            angles_out.append(0.)
    a = np.concatenate(matrix)
    b = np.concatenate(rhs)
    # Pick the axis point closest to the observed part centroid (axial gauge).
    reg = 1e-3
    a = np.vstack([a, reg*axis])
    b = np.r_[b, reg*(axis @ anchor)]
    pivot = np.linalg.lstsq(a, b, rcond=1e-7)[0]
    if not np.isfinite(pivot).all() or np.linalg.norm(pivot-anchor) > 5:
        pivot = anchor.copy()
    return axis, pivot, np.asarray(angles_out)


def modeled_points(kind, params, xyz, parent_poses):
    u = unit(params[:3])
    if kind == 'revolute':
        q = params[3:6]
        angles = params[6:]
        relative = xyz-q
        cross = np.cross(u, relative)
        cross2 = np.cross(u, cross)
        local = [q+relative+math.sin(a)*cross+(1-math.cos(a))*cross2 for a in angles]
    elif kind == 'prismatic':
        local = [xyz + d*u for d in params[3:]]
    else:
        local = [xyz for _ in parent_poses]
    return np.asarray([p @ r.T + t for p, (r, t, _) in zip(local, parent_poses)])


def child_poses_from_joint(kind, params, parent_poses):
    """Compose the fitted joint with its parent; these poses feed descendants."""
    direction = unit(params[:3]) if kind != 'fixed' else None
    poses = []
    for state, (parent_r, parent_t, _) in enumerate(parent_poses):
        if kind == 'revolute':
            joint_r, _ = cv2.Rodrigues(direction * params[6 + state])
            pivot = params[3:6]
            joint_t = pivot - joint_r @ pivot
        elif kind == 'prismatic':
            joint_r = np.eye(3)
            joint_t = params[3 + state] * direction
        else:
            joint_r, joint_t = np.eye(3), np.zeros(3)
        poses.append((parent_r @ joint_r, parent_r @ joint_t + parent_t, True))
    return poses


def fit_model(kind, xyz, image, mask, parent_poses, k, e, init, max_iter=18):
    """Small damped, finite-difference robust reprojection least-squares fit."""
    rows, cols = np.where(mask)
    if len(rows) < 8:
        return None
    def errors(p):
        point3 = modeled_points(kind, p, xyz, parent_poses)
        reproj = np.asarray([project(point3[s], k, e) for s in range(len(rows_by_state))])
        return (reproj[rows, cols] - image[rows, cols]).reshape(-1)
    rows_by_state = parent_poses
    p = init.astype(float).copy()
    # Point-level Huber weights; 8 px reduces influence of stray predicted tracks.
    def loss(err):
        norms = np.linalg.norm(err.reshape(-1, 2), axis=1)
        return float(np.sum(np.where(norms < 8, .5*norms**2, 8*norms-32)))
    damping = 1e-2
    for _ in range(max_iter):
        err = errors(p)
        if not np.isfinite(err).all():
            return None
        w = np.repeat(np.minimum(1, 8/np.maximum(np.linalg.norm(err.reshape(-1, 2), axis=1), 1e-8)), 2)
        jac = np.empty((len(err), len(p)))
        for j in range(len(p)):
            h = 1e-4 * max(1, abs(p[j]))
            perturbed = p.copy()
            perturbed[j] += h
            jac[:, j] = (errors(perturbed)-err)/h
        if not np.isfinite(jac).all():
            return None
        jtw = jac.T*w
        a = jtw @ jac
        g = jtw @ err
        accepted = False
        old = loss(err)
        for _ in range(6):
            step = np.linalg.lstsq(a+damping*np.diag(np.diag(a)+1), -g, rcond=1e-8)[0]
            candidate = p + step
            candidate[:3] = unit(candidate[:3])
            if kind == 'revolute':
                # Remove the unobservable component along the rotation axis.
                anchor = xyz.mean(axis=0)
                candidate[3:6] -= candidate[:3]*((candidate[3:6]-anchor) @ candidate[:3])
            new_err = errors(candidate)
            if np.isfinite(new_err).all() and loss(new_err) < old:
                p = candidate
                damping = max(damping/3, 1e-6)
                accepted = True
                break
            damping *= 10
        if not accepted or np.linalg.norm(step) < 1e-5:
            break
    err = errors(p).reshape(-1, 2)
    norms = np.linalg.norm(err, axis=1)
    # Trim gross outliers once and refit, as in robust trajectory fitting.
    if kind != 'fixed' and len(norms) >= 12:
        threshold = max(12, float(np.quantile(norms, .8)))
        keep = norms <= threshold
        if keep.sum() >= 8 and keep.sum() < len(norms):
            trimmed = np.zeros_like(mask)
            trimmed[rows[keep], cols[keep]] = True
            refined = fit_model(kind, xyz, image, trimmed, parent_poses, k, e, p, max_iter=8) if max_iter > 8 else None
            if refined is not None:
                p = refined['params']
                err = errors(p).reshape(-1, 2)
                norms = np.linalg.norm(err, axis=1)
    return {'params': p, 'rmse_px': float(np.sqrt(np.mean(norms**2))),
            'median_px': float(np.median(norms)), 'n': len(norms),
            'sse': float(np.sum(np.where(norms < 8, norms**2, 16*norms-64)))}


def select_type(fits, motion_px):
    """Select the smallest robust reprojection error, with a stationary-motion guard."""
    fixed = fits['fixed']
    scores = {}
    for kind, result in fits.items():
        if result is None:
            continue
        scores[kind] = result['sse']
    if motion_px < 3 or fixed['median_px'] < 3:
        return 'fixed', scores
    return min(scores, key=scores.get), scores


def axis_error(est, gt):
    return math.degrees(math.acos(np.clip(abs(unit(est) @ unit(gt)), 0, 1)))


def axis_distance(p, axis, origin):
    """Distance from GT origin to estimated line, invariant to axial pivot shift."""
    return float(np.linalg.norm(np.cross(origin-p, unit(axis))))


def scene_data(ds, index):
    sample = ds[index]
    path = Path(sample['scene_path'])
    with np.load(path) as a:
        source = {k: a[k].copy() for k in (
            'rgb', 'depth', 'camera_intrinsic', 'camera_extrinsic', 'joint_indices',
            'joint_types', 'joint_parent_link_indices', 'joint_child_link_indices',
            'joint_origins_world', 'joint_axes_world', 'link_indices')}
    pix = sample['reference_pixels'].numpy().astype(float)
    x = np.clip(np.rint(pix[:, 0]).astype(int), 0, source['depth'].shape[2]-1)
    y = np.clip(np.rint(pix[:, 1]).astype(int), 0, source['depth'].shape[1]-1)
    depth = source['depth'][0, y, x].astype(float)
    source['world'] = _backproject(pix, depth, source['camera_intrinsic'], source['camera_extrinsic'])
    source['point_link_nodes'] = sample['point_link_nodes'].numpy()
    source['point_valid'] = sample['point_valid'].numpy()
    source['path'] = path
    return source


def evaluate_scene(source, result_path, mode, index):
    with np.load(result_path) as a:
        selected = a['point_indices']
        nodes = a['link_nodes']
        predictions = a['predictions'][0].astype(float)
        validity = a['position_valid'].astype(bool)
        known = int(a['known_states'])
    if not np.array_equal(nodes, source['point_link_nodes'][selected]):
        raise ValueError(f'Point ordering differs: {result_path}')
    if known != (2 if mode == 'with_poke' else 1):
        raise ValueError(f'Unexpected known_states={known}: {result_path}')
    world = source['world'][selected]
    valid0 = source['point_valid'][selected]
    k, e = source['camera_intrinsic'], source['camera_extrinsic']
    h, w = source['rgb'].shape[1:3]
    pixels = from_normalized(predictions, h, w)[STATES]
    valid = validity[STATES] & valid0[None]
    by_label = {int(label): n for n, label in enumerate(source['link_indices'])}
    if -1 not in by_label:
        raise ValueError(f'Missing identified stationary base in {source["path"]}')
    base_mask = nodes == by_label[-1]
    base_projection = project(world[base_mask], k, e)
    base_errors = np.linalg.norm(pixels[:, base_mask] - base_projection, axis=-1)
    base_drift = float(np.median(base_errors[valid[:, base_mask]])) if valid[:, base_mask].any() else float('nan')
    identity_poses = [(np.eye(3), np.zeros(3), True) for _ in STATES]
    link_poses = {-1: identity_poses}
    records = []
    # Joints are visited in parent-before-child order, regardless of source index.
    pending = set(range(len(source['joint_indices'])))
    while pending:
        ready = [j for j in sorted(pending)
                 if int(source['joint_parent_link_indices'][j]) in link_poses]
        if not ready:
            raise ValueError(f'Joint hierarchy is disconnected or cyclic: {source["path"]}')
        j = ready[0]
        pending.remove(j)
        parent_label = int(source['joint_parent_link_indices'][j])
        child_label = int(source['joint_child_link_indices'][j])
        parent_poses = link_poses[parent_label]
        if child_label not in by_label:
            link_poses[child_label] = parent_poses
            continue
        child_node = by_label[int(child_label)]
        child_mask = nodes == child_node
        if not child_mask.any() or valid[:, child_mask].sum() < 8:
            link_poses[child_label] = parent_poses
            continue
        xyz = world[child_mask]
        uv = pixels[:, child_mask]
        mask = valid[:, child_mask] & np.isfinite(uv).all(-1)
        # Child PnP is only an initializer; its pose never drives descendants.
        child_init = [estimate_pose(xyz, uv[s], mask[s], k, e)
                      for s in range(len(STATES))]
        anchor = xyz.mean(0)
        axis0, pivot0, theta0 = initial_axis(child_init, parent_poses, anchor)
        fixed = fit_model('fixed', xyz, uv, mask, parent_poses, k, e, np.zeros(0), max_iter=0)
        rev = fit_model('revolute', xyz, uv, mask, parent_poses, k, e,
                        np.r_[axis0, pivot0, theta0])
        # Prismatic initialization comes from PnP translations when available.
        translations = []
        for (rc, tc, gc), (rp, tp, gp) in zip(child_init, parent_poses):
            translations.append(rp.T @ (tc-tp) if gc and gp else np.zeros(3))
        travel = np.asarray(translations)
        sv = np.linalg.svd(travel, full_matrices=False)[2][0]
        pris = fit_model('prismatic', xyz, uv, mask, parent_poses, k, e,
                         np.r_[sv, travel @ sv])
        fits = {'fixed': fixed, 'prismatic': pris, 'revolute': rev}
        if fixed is None:
            link_poses[child_label] = parent_poses
            continue
        # Relative 2D motion against the parent-rigid (fixed) hypothesis.
        fixed_xy = np.asarray([project(p, k, e) for p in modeled_points('fixed', np.zeros(0), xyz, parent_poses)])
        motion = float(np.median(np.linalg.norm((uv-fixed_xy)[mask], axis=1)))
        predicted_type, scores = select_type(fits, motion)
        gt_type = {0: 'revolute', 1: 'prismatic', 4: 'fixed'}.get(int(source['joint_types'][j]), 'other')
        chosen = fits[predicted_type]
        link_poses[child_label] = child_poses_from_joint(predicted_type, chosen['params'], parent_poses)
        est_axis = unit(chosen['params'][:3]) if predicted_type != 'fixed' else np.full(3, np.nan)
        est_pivot = chosen['params'][3:6] if predicted_type == 'revolute' else np.full(3, np.nan)
        gt_axis = source['joint_axes_world'][0, j].astype(float)
        gt_origin = source['joint_origins_world'][0, j].astype(float)
        rec = dict(scene_index=index, sample_id=source['path'].stem, mode=mode,
                   joint_index=int(source['joint_indices'][j]), parent_link=int(parent_label),
                   child_link=int(child_label), gt_type=gt_type, predicted_type=predicted_type,
                   n_points=int(child_mask.sum()), n_observations=int(mask.sum()),
                   parent_pose_valid=True, base_drift_px=base_drift, motion_px=motion,
                   fixed_rmse_px=fixed['rmse_px'],
                   prismatic_rmse_px=pris['rmse_px'] if pris else float('nan'),
                   revolute_rmse_px=rev['rmse_px'] if rev else float('nan'),
                   selected_rmse_px=chosen['rmse_px'],
                   axis_x=est_axis[0], axis_y=est_axis[1], axis_z=est_axis[2],
                   pivot_x=est_pivot[0], pivot_y=est_pivot[1], pivot_z=est_pivot[2],
                   gt_axis_x=gt_axis[0], gt_axis_y=gt_axis[1], gt_axis_z=gt_axis[2],
                   gt_origin_x=gt_origin[0], gt_origin_y=gt_origin[1], gt_origin_z=gt_origin[2],
                   axis_error_deg=axis_error(est_axis, gt_axis) if predicted_type == gt_type and gt_type != 'fixed' else float('nan'),
                   pivot_distance=axis_distance(est_pivot, est_axis, gt_origin) if predicted_type == gt_type == 'revolute' else float('nan'),
                   score_fixed=scores.get('fixed', float('nan')),
                   score_prismatic=scores.get('prismatic', float('nan')),
                   score_revolute=scores.get('revolute', float('nan')))
        records.append(rec)
    return records


def write_csv(path, records):
    if not records:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(records, total_gt):
    result = {'scenes': len(set(r['scene_index'] for r in records)),
              'joints_evaluated': len(records), 'joints_total': total_gt,
              'joint_coverage': len(records)/total_gt if total_gt else None,
              'gt_type_counts': {}, 'predicted_type_counts': {}, 'confusion': {}}
    types = ['fixed', 'prismatic', 'revolute']
    for t in types:
        rows = [r for r in records if r['gt_type'] == t]
        result['gt_type_counts'][t] = len(rows)
        result['predicted_type_counts'][t] = sum(r['predicted_type'] == t for r in records)
        result['confusion'][t] = {p: sum(r['predicted_type'] == p for r in rows) for p in types}
    correct = sum(r['gt_type'] == r['predicted_type'] for r in records)
    result['joint_type_accuracy'] = correct/len(records) if records else None
    result['correct_type_fraction_all_gt'] = correct/total_gt if total_gt else None
    for t in types:
        denom = result['gt_type_counts'][t]
        result[f'{t}_recall'] = result['confusion'][t][t]/denom if denom else None
        predicted = result['predicted_type_counts'][t]
        precision = result['confusion'][t][t]/predicted if predicted else None
        recall = result[f'{t}_recall']
        result[f'{t}_precision'] = precision
        result[f'{t}_f1'] = (2*precision*recall/(precision+recall) if precision+recall else 0.0) if precision is not None and recall is not None else None
    supported_recalls = [result[f'{t}_recall'] for t in types if result[f'{t}_recall'] is not None]
    result['balanced_type_accuracy'] = float(np.mean(supported_recalls)) if supported_recalls else None
    for name in ('axis_error_deg', 'pivot_distance', 'selected_rmse_px'):
        values = np.asarray([r[name] for r in records], float)
        values = values[np.isfinite(values)]
        result[name] = {'n': len(values), 'mean': float(np.mean(values)) if len(values) else None,
                        'median': float(np.median(values)) if len(values) else None,
                        'p90': float(np.percentile(values, 90)) if len(values) else None,
                        'p99': float(np.percentile(values, 99)) if len(values) else None}
    result['reprojection_failures_over_50px'] = sum(r['selected_rmse_px'] > 50 for r in records)
    result['reprojection_failures_over_100px'] = sum(r['selected_rmse_px'] > 100 for r in records)
    drift_by_scene = {r['scene_index']: r['base_drift_px'] for r in records}
    drift = np.asarray(list(drift_by_scene.values()), float)
    drift = drift[np.isfinite(drift)]
    result['stationary_base_track_drift_px'] = {
        'n_scenes': len(drift),
        'median': float(np.median(drift)) if len(drift) else None,
        'p90': float(np.percentile(drift, 90)) if len(drift) else None,
    }
    return result


def figures(by_mode, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = OUT/'figures'
    out.mkdir(exist_ok=True)
    colors = {'without_poke': '#c76b25', 'with_poke': '#287e8e'}
    for field, title, xlabel, upper in [
        ('axis_error_deg', 'Revolute axis orientation error', 'Degrees (lower is better)', 90),
        ('pivot_distance', 'Revolute GT origin to estimated axis', 'World units (lower is better)', None),
        ('selected_rmse_px', 'Selected joint model reprojection RMSE', 'Source-image pixels (lower is better)', None)]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values_by_mode = {}
        for mode, records in by_mode.items():
            v = np.asarray([r[field] for r in records], float)
            v = v[np.isfinite(v)]
            values_by_mode[mode] = v
            if len(v):
                xs = np.sort(v)
                ax.step(xs, np.arange(1, len(xs)+1)/len(xs), where='post',
                        label=f'{mode.replace("_", " ")} (n={len(v)}, median={np.median(v):.3g})',
                        color=colors[mode], linewidth=2)
        if upper is not None:
            ax.set_xlim(0, upper)
        elif any(len(v) for v in values_by_mode.values()):
            ax.set_xlim(left=0, right=max(np.percentile(v, 99) for v in values_by_mode.values() if len(v))*1.05)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Fraction of evaluated joints')
        ax.set_title(title)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/f'{field}_cdf.png', dpi=180)
        fig.savefig(out/f'{field}_cdf.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(1, len(by_mode), figsize=(5.3*len(by_mode), 4.4), squeeze=False)
    labels = ['fixed', 'prismatic', 'revolute']
    for ax, (mode, summary) in zip(axes[0], summaries.items()):
        matrix = np.array([[summary['confusion'][gt][pred] for pred in labels] for gt in labels])
        ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(3), labels, rotation=25)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel('Inferred type')
        ax.set_ylabel('Ground-truth type')
        ax.set_title(mode.replace('_', ' '))
        for y in range(3):
            for x in range(3):
                ax.text(x, y, str(matrix[y, x]), ha='center', va='center', color='black')
    fig.tight_layout()
    fig.savefig(out/'joint_type_confusion.png', dpi=180)
    fig.savefig(out/'joint_type_confusion.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mode, records in by_mode.items():
        by_scene = {r['scene_index']: r['base_drift_px'] for r in records}
        values = np.asarray(list(by_scene.values()), float)
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                    color=colors[mode], linewidth=2,
                    label=f'{mode.replace("_", " ")} (median={np.median(values):.2f} px)')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Predicted stationary-base track drift (source-image pixels)')
    ax.set_ylabel('Fraction of scenes')
    ax.set_title('Stationary-base consistency diagnostic')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'stationary_base_drift_cdf.png', dpi=180)
    fig.savefig(out/'stationary_base_drift_cdf.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    modes = list(summaries)
    for ax, field, title, ylabel in zip(axes,
            ('axis_error_deg', 'pivot_distance'),
            ('Mean axis orientation error', 'Mean pivot-to-axis distance'),
            ('Degrees ↓', 'World units ↓')):
        vals = [summaries[m][field]['mean'] or 0 for m in modes]
        bars = ax.bar([f'{m.replace("_", " ")}\n(n={summaries[m][field]["n"]})' for m in modes], vals,
                      color=[colors[m] for m in modes])
        ax.bar_label(bars, fmt='%.3g', padding=3)
        ax.set_ylim(0, max(vals)*1.25 if any(vals) else 1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'mean_joint_errors.png', dpi=180)
    fig.savefig(out/'mean_joint_errors.pdf')
    plt.close(fig)


def example_figures(by_mode):
    """Reference-image overlays for low, typical, and high axis-error cases."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = OUT/'figures'/'examples'
    for mode, records in by_mode.items():
        folder = out/mode
        folder.mkdir(parents=True, exist_ok=True)
        for previous in folder.glob('*'):
            if previous.suffix in ('.png', '.pdf'):
                previous.unlink()
        candidates = [r for r in records if r['gt_type'] == r['predicted_type'] == 'revolute'
                      and np.isfinite(r['axis_error_deg']) and r['selected_rmse_px'] < 30
                      and r['pivot_distance'] < 1]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda r: r['axis_error_deg'])
        for tag, percentile in [('good', 0), ('typical', .5), ('failure', .9)]:
            rec = (min(candidates, key=lambda r: r['axis_error_deg']/45 + r['pivot_distance']/.1)
                   if tag == 'good' else ranked[round(percentile*(len(ranked)-1))])
            scene_path = ROOT/'data/Robotic_arm/test1/scenes'/f'{rec["sample_id"]}.npz'
            with np.load(scene_path) as a:
                image = a['rgb'][0]
                depth = a['depth'][0]
                foreground = a['depth_valid_mask'][0] & (a['link_segmentation'][0] != -2)
                k = a['camera_intrinsic'].astype(float)
                e = a['camera_extrinsic'].astype(float)
            yy, xx = np.nonzero(foreground[::8, ::8])
            samples = np.column_stack((xx*8, yy*8))
            if len(samples) < 2:
                continue
            xyz = _backproject(samples, depth[samples[:, 1], samples[:, 0]], k, e)
            length = .23*np.linalg.norm(xyz.max(0)-xyz.min(0))
            gt_q = np.array([rec[f'gt_origin_{a}'] for a in 'xyz'], float)
            gt_u = unit(np.array([rec[f'gt_axis_{a}'] for a in 'xyz'], float))
            pred_q = np.array([rec[f'pivot_{a}'] for a in 'xyz'], float)
            pred_u = unit(np.array([rec[f'axis_{a}'] for a in 'xyz'], float))
            if pred_u @ gt_u < 0:
                pred_u = -pred_u
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.imshow(image)
            for q, u, color in ((gt_q, gt_u, '#18dc65'), (pred_q, pred_u, '#ff4b3a')):
                line = project(np.array([q-length*u, q+length*u]), k, e)
                center = project(q[None], k, e)[0]
                ax.plot(line[:, 0], line[:, 1], color=color, linewidth=3,
                        solid_capstyle='round')
                ax.scatter([center[0]], [center[1]], s=95, color=color,
                           edgecolors='black', linewidths=1, zorder=5)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.set_xlabel('Source-image x (pixels)')
            ax.set_ylabel('Source-image y (pixels)')
            ax.set_title(f'{mode.replace("_", " ")} · {tag} · scene {int(rec["scene_index"]):04d}, joint {rec["joint_index"]}\n'
                         f'Axis error {rec["axis_error_deg"]:.1f}° · pivot distance {rec["pivot_distance"]:.3g} world units')
            ax.legend(handles=[Line2D([0], [0], color='#18dc65', lw=3, label='Ground truth'),
                               Line2D([0], [0], color='#ff4b3a', lw=3, label='Estimated')],
                      loc='upper right')
            fig.tight_layout()
            fig.savefig(folder/f'{tag}_scene_{int(rec["scene_index"]):04d}_joint_{rec["joint_index"]}.png', dpi=180)
            fig.savefig(folder/f'{tag}_scene_{int(rec["scene_index"]):04d}_joint_{rec["joint_index"]}.pdf')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='First N scenes for debugging')
    parser.add_argument('--modes', nargs='+', default=['without_poke', 'with_poke'],
                        choices=['without_poke', 'with_poke'])
    parser.add_argument('--report-only', action='store_true',
                        help='Rebuild summaries and figures from saved per-joint CSV files')
    args = parser.parse_args()
    if args.report_only:
        previous = json.loads((OUT/'results'/'summary.json').read_text())
        by_mode = {}
        for mode in args.modes:
            with open(OUT/'results'/mode/'per_joint.csv', newline='') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                for field in ('axis_error_deg', 'pivot_distance', 'selected_rmse_px', 'base_drift_px',
                              'axis_x', 'axis_y', 'axis_z', 'pivot_x', 'pivot_y', 'pivot_z',
                              'gt_axis_x', 'gt_axis_y', 'gt_axis_z',
                              'gt_origin_x', 'gt_origin_y', 'gt_origin_z'):
                    row[field] = float(row[field])
            by_mode[mode] = rows
        summaries = {}
        for mode, rows in by_mode.items():
            summary = summarize(rows, previous[mode]['joints_total'])
            for field in ('dataset_split', 'hypothesis_index', 'forecast_states_used', 'mode'):
                summary[field] = previous[mode][field]
            summary['fit_method'] = 'base_outward_forward_kinematics'
            summary['stationary_base_fixed'] = True
            summary['type_selection_score'] = 'unpenalized_robust_reprojection_sse'
            summaries[mode] = summary
            (OUT/'results'/mode/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        figures(by_mode, summaries)
        example_figures(by_mode)
        (OUT/'results'/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summaries, indent=2, allow_nan=False))
        return
    ds = ArticulatedDataset(ROOT/'data/Robotic_arm', split='test1', training=False,
                            points_per_link=8, image_size=384, seed=42)
    count = min(len(ds), args.limit) if args.limit else len(ds)
    runs = ROOT/'finetuning/runs/test1_presentation'
    by_mode = {mode: [] for mode in args.modes}
    total_gt = 0
    for i in range(count):
        source = scene_data(ds, i)
        total_gt += len(source['joint_indices'])
        for mode in args.modes:
            path = runs/mode/'results'/f'scene_{i:04d}_{source["path"].stem}'/'results.npz'
            if not path.exists():
                raise FileNotFoundError(path)
            by_mode[mode].extend(evaluate_scene(source, path, mode, i))
        if (i+1) % 20 == 0 or i+1 == count:
            print(f'{i+1}/{count} scenes', flush=True)
    summaries = {}
    for mode, records in by_mode.items():
        folder = OUT/'results'/mode
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder/'per_joint.csv', records)
        scene_rows = []
        for i in range(count):
            scene_records = [r for r in records if r['scene_index'] == i]
            scene_rows.append({'scene_index': i, 'sample_id': ds.files[i].stem,
                               'joints_evaluated': len(scene_records),
                               'base_drift_px': scene_records[0]['base_drift_px'] if scene_records else float('nan'),
                               'correct_types': sum(r['gt_type'] == r['predicted_type'] for r in scene_records),
                               'mean_axis_error_deg': float(np.mean([r['axis_error_deg'] for r in scene_records if np.isfinite(r['axis_error_deg'])])) if any(np.isfinite(r['axis_error_deg']) for r in scene_records) else float('nan'),
                               'mean_pivot_distance': float(np.mean([r['pivot_distance'] for r in scene_records if np.isfinite(r['pivot_distance'])])) if any(np.isfinite(r['pivot_distance']) for r in scene_records) else float('nan')})
        write_csv(folder/'per_scene.csv', scene_rows)
        summary = summarize(records, total_gt)
        summary['dataset_split'] = 'test1'
        summary['hypothesis_index'] = 0
        summary['forecast_states_used'] = STATES.tolist()
        summary['mode'] = mode
        summary['fit_method'] = 'base_outward_forward_kinematics'
        summary['stationary_base_fixed'] = True
        summary['type_selection_score'] = 'unpenalized_robust_reprojection_sse'
        summaries[mode] = summary
        (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    figures(by_mode, summaries)
    example_figures(by_mode)
    (OUT/'results'/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
