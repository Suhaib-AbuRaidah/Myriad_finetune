"""Fit saved test1 tracks with the unchanged public candidate; no neural inference."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np

from testing_joint_fitting.contact_axis_orientation.candidate import CONFIG, estimate_motion
from testing_joint_fitting.contact_axis_orientation.validation_transfer.evaluate import frozen_sources
from testing_joint_fitting.axis_motion.common import score, summary
from testing_joint_fitting.experiment import write
from training_from_scratch.full_evaluation.metrics import distribution, kinematic_reference

DEFAULT = Path('training_from_scratch_cropped/runs/full_test1_val80_20261021')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proper_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def scene_worker(job):
    archive, records = job
    archive = Path(archive)
    source_path = archive / 'test1' / records[0]['source_archive']
    with np.load(source_path, allow_pickle=False) as a:
        points, normals, labels = (a[k][0] for k in
            ('mesh_points', 'mesh_normals', 'mesh_point_link_labels'))
        source = {k: a[k] for k in ('joint_indices', 'joint_axes_world',
            'joint_origins_world', 'link_transforms_world', 'link_indices')}
    output = []
    for record in records:
        path = archive / 'test1' / record['prediction_file']
        with np.load(path, allow_pickle=False) as a:
            xyz, uv, valid, k, e = (a[key].astype(float) for key in
                ('reference_world', 'predictions_source_pixels', 'valid',
                 'camera_intrinsic', 'camera_extrinsic'))
            uv = uv[0]
            valid = valid.astype(bool)
            assert list(a['pair_labels']) == [record['parent_link'], record['child_link']]
        assert uv.shape == (5, len(xyz), 2)
        masks = [labels == record[key] for key in ('parent_link', 'child_link')]
        start = time.perf_counter()
        exception = None
        try:
            if valid[2:5].sum() < 8:
                raise ValueError('Fewer than eight valid observations in fitting states 2..4')
            result = estimate_motion(xyz, uv, valid, k, e,
                parent_points=points[masks[0]], child_points=points[masks[1]],
                parent_normals=normals[masks[0]], child_normals=normals[masks[1]])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
            result = None
            exception = repr(error)
        elapsed = time.perf_counter() - start

        # Truth is used only here, after the estimator has returned its result.
        jp = int(np.flatnonzero(source['joint_indices'] == record['joint_index'])[0])
        lookup = {int(label): node for node, label in enumerate(source['link_indices'])}
        parent, child = (lookup[record[key]] for key in ('parent_link', 'child_link'))
        t = source['link_transforms_world'].astype(float)
        truth_transforms = np.stack([t[0, parent] @ np.linalg.inv(t[s, parent]) @
            t[s, child] @ np.linalg.inv(t[0, child]) for s in (2, 3, 4)])
        reference = kinematic_reference(source, jp, parent, child)
        case = {key: record[key] for key in ('pair_index', 'scene_index', 'joint_ordinal', 'object_id')}
        case.update(gt_axis=source['joint_axes_world'][0, jp].astype(float),
            gt_pivot=np.array(reference['pivot']) if reference.get('observable') else None,
            gt_rotations=truth_transforms[:, :3, :3])
        row = score(case, None if result is None else result['fit'])
        row.update(joint_index=record['joint_index'], sample_id=record['sample_id'],
            fit_seconds=elapsed, exception=exception, source_archive=record['source_archive'],
            prediction_file=record['prediction_file'], prediction_sha256=digest(path),
            record_file=f"records/pair_{record['pair_index']:05d}.json",
            valid_fit_observations=int(valid[2:5].sum()),
            trajectory_metrics=record['trajectory_metrics'].get('primary'),
            gt_axis=case['gt_axis'], gt_pivot=case['gt_pivot'],
            gt_transforms=truth_transforms, kinematic_reference=reference)
        if result is not None:
            row.update(transforms=result['transforms'], rmse_px=result['fit'].get('rmse_px'),
                fit=result['fit'])
            p = np.asarray(row['params'])
            axis = p[:3] / np.linalg.norm(p[:3])
            gt_axis = case['gt_axis'] / np.linalg.norm(case['gt_axis'])
            axis_error = np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, gt_axis)),
                abs(axis @ gt_axis)))
            rotation_errors, rmse_squared, depths, translation_errors = [], [], [], []
            reconstruction_error = 0.
            for index, state in enumerate((2, 3, 4)):
                transform = result['transforms'][index]
                rotation = cv2.Rodrigues(p[:3] * p[6+index])[0]
                reconstruction_error = max(reconstruction_error,
                    float(np.max(np.abs(transform[:3, :3] - rotation))),
                    float(np.max(np.abs(transform[:3, 3] - (p[3:6] - rotation @ p[3:6])))))
                relative = transform[:3, :3] @ proper_rotation(truth_transforms[index, :3, :3]).T
                angle = np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
                rotation_errors.append(float(angle))
                translation_errors.append(float(np.linalg.norm(transform[:3, 3] - truth_transforms[index, :3, 3])))
                world = xyz @ transform[:3, :3].T + transform[:3, 3]
                camera = world @ e[:3, :3].T + e[:3, 3]
                homogeneous = camera @ k.T
                projected = homogeneous[:, :2] / homogeneous[:, 2:]
                rmse_squared.extend(np.square(projected[valid[state]] - uv[state, valid[state]]).sum(-1))
                depths.extend(camera[valid[state], 2])
            pivot_error = None if case['gt_pivot'] is None else float(np.linalg.norm(
                np.cross(case['gt_pivot'] - p[3:6], axis)))
            row.update(rotation_error_per_state_deg=rotation_errors,
                translation_error_per_state=translation_errors,
                audit=dict(axis_error_difference_deg=abs(float(axis_error)-row['axis_error_deg']),
                    rotation_error_difference_deg=abs(float(np.mean(rotation_errors))-row['rotation_error_deg']),
                    pivot_error_difference=None if pivot_error is None else abs(pivot_error-row['pivot_distance']),
                    reprojection_rmse_difference_px=abs(float(np.sqrt(np.mean(rmse_squared)))-row['rmse_px']),
                    transform_reconstruction_max_difference=reconstruction_error,
                    axis_unit_error=abs(float(np.linalg.norm(p[:3]))-1.),
                    min_scored_depth=float(min(depths))))
        output.append(row)
    return output


def enriched_summary(rows):
    result = summary(rows)
    result.update(reprojection_rmse_pixels=distribution(r.get('rmse_px') for r in rows),
        trajectory_ADE_source_pixels=distribution((r.get('trajectory_metrics') or {}).get('pixel_ADE') for r in rows))
    return result


def generate_report(output, cohort, rows, wall_seconds):
    groups = {}
    for field in ('joint_ordinal', 'object_id', 'scene_index'):
        groups[field] = {str(value): enriched_summary([r for r in rows if r[field] == value])
            for value in sorted({r[field] for r in rows})}
    old_path = Path('testing_joint_fitting/contact_axis_orientation/results/variants/ambiguous_mixture_strength128.json')
    previous = json.loads(old_path.read_text())
    keyed = {(r['scene_index'], r['joint_ordinal'], r['object_id']): r for r in rows}
    matched_old = [r for r in previous['rows'] if (r['scene_index'], r['joint_ordinal'], r['object_id']) in keyed]
    matched_new = [keyed[r['scene_index'], r['joint_ordinal'], r['object_id']] for r in matched_old]
    comparison = dict(n=len(matched_old), old_checkpoint_step=108000, new_checkpoint_step=86000,
        matching='Exact scene index, joint ordinal and object identity; no pair-index matching',
        old=summary(matched_old), cropped=summary(matched_new),
        caveat='Different checkpoints/training pipelines; this comparison does not isolate cropping causally.')
    result = dict(cohort=cohort, overall=enriched_summary(rows), by_joint=groups['joint_ordinal'],
        by_object=groups['object_id'], by_scene=groups['scene_index'], matched_previous=comparison,
        wall_seconds=wall_seconds, cpu_fit_seconds=sum(r['fit_seconds'] for r in rows))
    write(output / 'summary.json', result)
    write(output / 'rows.json', rows)
    write(output / 'outliers.json', sorted(rows, key=lambda r: r['axis_error_with_failure_penalty_deg'], reverse=True)[:50])
    audits = [r['audit'] for r in rows if not r['failed']]
    maxima = {key: max(a[key] for a in audits if a[key] is not None)
        for key in audits[0] if key != 'min_scored_depth'}
    audit = dict(status='passed', successful_fits=len(audits), independently_recomputed_max_differences=maxima,
        min_scored_depth=min(a['min_scored_depth'] for a in audits),
        all_prediction_hashes_recorded=True, frozen_algorithm_sources_unchanged=True)
    assert maxima['axis_error_difference_deg'] < 1e-5, maxima
    assert maxima['rotation_error_difference_deg'] < .001, maxima
    assert maxima['pivot_error_difference'] < 1e-10, maxima
    assert maxima['reprojection_rmse_difference_px'] < 1e-5, maxima
    assert maxima['transform_reconstruction_max_difference'] < 1e-12, maxima
    assert maxima['axis_unit_error'] < 1e-6, maxima
    assert audit['min_scored_depth'] > 0, audit
    write(output / 'audit.json', audit)
    overall = result['overall']
    text = ['# Frozen candidate on cropped-model test1 predictions', '',
        'Checkpoint: cropped `best.pt`, step 86000, SHA256 `'+cohort['checkpoint_sha256']+'`.', '',
        f"All {cohort['scenes']} test1 scenes were considered. {len(rows)} scoreable revolute joints "
        f"across {cohort['scored_scenes']} scenes and {cohort['objects']} objects were fitted; "
        f"{overall['failures']} fitting failures. Fixed joints and unavailable observations are listed below.", '',
        'The unchanged public `contact_axis_orientation/candidate.py` entrypoint was used with strength 128 '
        'and position strength 4. Hypothesis 0 only, source-image pixel coordinates, original source visibility masks, '
        'fitting states 2, 3, 4. Initial mesh points/normals only. No GT axes, pivots, transforms or future mesh '
        'geometry entered estimation. No new inference, hypothesis oracle, parameter tuning or validation fitting.', '',
        '| Metric | Mean | Median | P90 |', '|---|---:|---:|---:|']
    for name, key in [('Axis orientation error (degrees)', 'axis'), ('Pivot-to-axis distance (world units)', 'pivot'),
        ('Rotation error (degrees)', 'rotation'), ('Fit reprojection RMSE (source pixels)', 'reprojection_rmse_pixels'),
        ('Input trajectory ADE (source pixels, states 1–4)', 'trajectory_ADE_source_pixels')]:
        d = overall[key]
        text.append(f"| {name} | {d['mean']:.6f} | {d['median']:.6f} | {d['p90']:.6f} |")
    text += ['', f"Axis error ≤10°: {overall['success_10_deg']*100:.2f}%; ≤15°: {overall['success_15_deg']*100:.2f}%.", '',
        '## By source joint order', '', '| Joint | Count | Axis mean ° | Axis median ° | Pivot mean | Rotation mean ° | Within 15° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ordinal, s in result['by_joint'].items():
        text.append(f"| {ordinal} | {s['n']} | {s['axis']['mean']:.4f} | {s['axis']['median']:.4f} | "
            f"{s['pivot']['mean']:.6f} | {s['rotation']['mean']:.4f} | {s['success_15_deg']*100:.2f}% |")
    text += ['', 'Joint order is the original joint index + 1; excluded fixed joints are not renumbered.', '',
        '## Matched comparison with the previous uncropped-model development evaluation', '',
        f"Exactly {comparison['n']} scene/joint/object identities match the previous 80-scene cohort.", '',
        '| Metric mean | Previous uncropped, step 108000 | Cropped, step 86000 |', '|---|---:|---:|']
    for name, key in [('Axis error °', 'axis'), ('Pivot distance', 'pivot'), ('Rotation error °', 'rotation')]:
        text.append(f"| {name} | {comparison['old'][key]['mean']:.6f} | {comparison['cropped'][key]['mean']:.6f} |")
    text += ['', comparison['caveat'], 'The fitter was originally developed on these test1 objects. '
        'The full test1 result is not a test of generalization to unseen object identities.', '',
        '## Coverage and definitions', '', '```json', json.dumps(cohort, indent=2), '```', '',
        'Axis orientation is sign-invariant, in [0°, 90°]. Pivot distance is the perpendicular distance '
        'from the transform-consistent GT pivot reference to the predicted infinite axis line; it is '
        'not Euclidean distance between arbitrary pivot representatives. Values use dataset world units '
        '(no unverified conversion to centimetres). Rotation error averages SO(3) errors over states 2–4. '
        'Pairs are weighted equally. Failed fits receive 90° in the separately recorded failure-penalized axis metric.', '',
        'Candidate assumes revolute joints; no joint-type classifier is evaluated. Fixed joints are excluded. '
        'Source validity masks may reflect GT visibility, as in the previous fitting protocol.', '',
        f'Elapsed evaluation time: {wall_seconds:.1f}s; summed CPU fitting time: {result["cpu_fit_seconds"]:.1f}s.', '',
        '## Saved artifacts', '',
        '`rows.json` and `pairs/pair_XXXXX.json` retain axis/pivot/angles, state transforms, solver diagnostics, '
        'kinematic scoring references, source-prediction paths and hashes. The parent inference archive retains '
        'all hypotheses, images, masks, cameras, meshes and trajectories for presentation visualizations. '
        '`summary.json` includes every joint, object and scene aggregate; `outliers.json` lists the 50 largest '
        'axis errors. `protocol.json` freezes inputs/configuration/code hashes. `audit.json` records independent '
        'axis, pivot, SO(3), reprojection and transform reconstruction checks.']
    (output / 'REPORT.md').write_text('\n'.join(text)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    archive = args.archive.resolve()
    output = archive / 'candidate_test1'
    manifest = json.loads((archive / 'run.json').read_text())
    assert json.loads((archive / 'audit.json').read_text())['status'] == 'passed'
    records = [json.loads(p.read_text()) for p in sorted((archive / 'test1/records').glob('*.json'))]
    split = manifest['splits']['test1']
    assert len(split['scenes']) == 320 and len(records) == len(split['pairs']) == 881
    assert manifest['checkpoint_step'] == 86000
    excluded, selected = [], []
    for r in records:
        if r['gt_type'] != 'revolute' or not r['scoreable']:
            excluded.append(dict(pair_index=r['pair_index'], scene_index=r['scene_index'],
                joint_index=r['joint_index'], reason='fixed joint' if r['gt_type']=='fixed' else 'no valid future targets'))
        else:
            selected.append(r)
    cohort = dict(scenes=320, source_joints=len(split['inventory']), eligible_predictions=len(records),
        source_kind_counts=dict(Counter(r['kind'] for r in split['inventory'])),
        scored_revolute_pairs=len(selected), scored_scenes=len({r['scene_index'] for r in selected}),
        objects=len({r['object_id'] for r in selected}), fixed_pairs_excluded=sum(r['gt_type']=='fixed' for r in records),
        no_future_targets_excluded=sum(r['gt_type']=='revolute' and not r['scoreable'] for r in records),
        initially_unobservable_joints=sum(not r['eligible'] for r in split['inventory']),
        checkpoint_sha256=manifest['checkpoint_sha256'])
    sources = frozen_sources()
    protocol = dict(archive=str(archive), run_manifest_sha256=digest(archive / 'run.json'),
        record_hashes={p.name: digest(p) for p in sorted((archive / 'test1/records').glob('*.json'))},
        algorithm_sources=sources, runner_sha256=digest(Path(__file__)), config=CONFIG,
        primary_hypothesis=0, fitting_states=[2, 3, 4], initial_geometry_state=0,
        coordinates='source image pixels / world geometry', tuning=False, cohort=cohort)
    if (output / 'protocol.json').exists():
        assert json.loads((output / 'protocol.json').read_text()) == protocol, 'Protocol changed'
    write(output / 'protocol.json', protocol)
    write(output / 'exclusions.json', dict(predicted_pairs=excluded,
        unobservable_source_joints=[r for r in split['inventory'] if not r['eligible']]))
    groups = defaultdict(list)
    for r in selected:
        groups[r['scene_index']].append(r)
    rows = []
    print(json.dumps(dict(event='loaded', **cohort)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(scene_worker, (str(archive), group)) for group in groups.values()]
        for completed, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            rows.extend(batch)
            for row in batch:
                write(output / 'pairs' / f"pair_{row['pair_index']:05d}.json", row)
            if completed % 40 == 0:
                print(json.dumps(dict(event='progress', scenes=completed, pairs=len(rows),
                    seconds=time.perf_counter()-started)), flush=True)
    assert frozen_sources() == sources, 'Fitter changed during evaluation'
    assert len(rows) == len(selected) == len({r['pair_index'] for r in rows})
    rows.sort(key=lambda r: r['pair_index'])
    result = generate_report(output, cohort, rows, time.perf_counter()-started)
    print(json.dumps(dict(event='complete', summary=result['overall'], output=str(output))), flush=True)


if __name__ == '__main__':
    main()
