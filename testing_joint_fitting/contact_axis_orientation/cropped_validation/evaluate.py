"""Fit saved validation tracks with the unchanged public candidate; no neural inference."""
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
    source_path = archive / 'val' / records[0]['source_archive']
    with np.load(source_path, allow_pickle=False) as a:
        points, normals, labels = (a[k][0] for k in
            ('mesh_points', 'mesh_normals', 'mesh_point_link_labels'))
        source = {k: a[k] for k in ('joint_indices', 'joint_axes_world',
            'joint_origins_world', 'link_transforms_world', 'link_indices')}
    output = []
    for record in records:
        path = archive / 'val' / record['prediction_file']
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


def previous_comparison(archive, manifest, rows):
    """Match the old 50-scene validation run by filename, never local scene index."""
    old_root = Path('training_from_scratch/runs/val50_contact_axis_transfer_20261021')
    old_manifest = json.loads((old_root / 'run.json').read_text())
    previous = json.loads(Path('testing_joint_fitting/contact_axis_orientation/validation_transfer/'
        'results/variants/frozen_candidate.json').read_text())
    keyed = {(Path(r['source_archive']).stem, r['joint_ordinal'], r['object_id']): r for r in rows}
    old_rows, new_rows = [], []
    for r in previous['rows']:
        filename = Path(old_manifest['scenes'][r['scene_index']]['path']).stem
        match = keyed.get((filename, r['joint_ordinal'], r['object_id']))
        if match is not None:
            old_rows.append(r)
            new_rows.append(match)
    return dict(n=len(old_rows), old=summary(old_rows), cropped=summary(new_rows),
        old_checkpoint_step=old_manifest['checkpoint_step'],
        cropped_checkpoint_step=manifest['checkpoint_step'],
        matching='Exact original scene filename, joint ordinal, object ID; no local index matching',
        caveat='Checkpoints and training pipelines differ; this does not isolate the effect of cropping.')


def verify(output, archive, protocol, rows):
    assert frozen_sources() == protocol['algorithm_sources'], 'Fitter changed'
    assert digest(Path(__file__)) == protocol['runner_sha256']
    assert digest(archive / 'run.json') == protocol['run_manifest_sha256']
    for name, expected in protocol['record_hashes'].items():
        assert digest(archive / 'val/records' / name) == expected
    for r in rows:
        assert digest(archive / 'val' / r['prediction_file']) == r['prediction_sha256']
        saved = json.loads((output / 'pairs' / f"pair_{r['pair_index']:05d}.json").read_text())
        # Convert numpy arrays before comparing the saved and in-memory records.
        from training_from_scratch.full_evaluation.metrics import clean
        assert saved == clean(r)
    audits = [r['audit'] for r in rows if not r['failed']]
    assert audits, 'No successful fits'
    maxima = {k: max((a[k] for a in audits if a[k] is not None), default=None)
        for k in audits[0] if k != 'min_scored_depth'}
    limits = dict(axis_error_difference_deg=1e-5, rotation_error_difference_deg=.001,
        pivot_error_difference=1e-10, reprojection_rmse_difference_px=1e-5,
        transform_reconstruction_max_difference=1e-12, axis_unit_error=1e-6)
    for key, limit in limits.items():
        assert maxima[key] is None or maxima[key] < limit, (key, maxima[key])
    depth = min(a['min_scored_depth'] for a in audits)
    assert depth > 0
    result = dict(status='passed', successful_fits=len(audits),
        independently_recomputed_max_differences=maxima, min_scored_depth=depth,
        prediction_hashes_reverified=len(rows), persisted_pair_records_verified=len(rows),
        inference_record_hashes_reverified=len(protocol['record_hashes']),
        runner_and_algorithm_hashes_reverified=True,
        insufficient_fitting_observations=sum(r['failed'] and r['valid_fit_observations'] < 8 for r in rows),
        other_fit_failures=sum(r['failed'] and r['valid_fit_observations'] >= 8 for r in rows))
    write(output / 'audit.json', result)
    return result


def fmt(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def report(output, archive, manifest, cohort, rows, elapsed, audit):
    aggregate = enriched_summary(rows)
    grouped = {field: {str(value): enriched_summary([r for r in rows if r[field] == value])
        for value in sorted({r[field] for r in rows})}
        for field in ('joint_ordinal', 'object_id', 'scene_index')}
    test = json.loads((archive / 'candidate_test1/summary.json').read_text())
    comparison = previous_comparison(archive, manifest, rows)
    result = dict(cohort=cohort, overall=aggregate, by_joint=grouped['joint_ordinal'],
        by_object=grouped['object_id'], by_scene=grouped['scene_index'],
        same_checkpoint_test1=test['overall'], matched_previous_validation=comparison,
        wall_seconds=elapsed, cpu_fit_seconds=sum(r['fit_seconds'] for r in rows))
    write(output / 'summary.json', result)
    write(output / 'rows.json', rows)
    write(output / 'outliers.json', sorted(rows, key=lambda r: r['axis_error_with_failure_penalty_deg'], reverse=True)[:50])
    failed = [r for r in rows if r['failed']]
    lines = ['# Frozen candidate on cropped-model validation predictions', '',
        f"All {cohort['scenes']} archived validation scenes were considered: {len(rows)} scoreable revolute pairs, "
        f"{len(rows)-len(failed)} successful fits, {len(failed)} unavailable fits, {cohort['objects']} object identities.", '',
        f"Checkpoint: cropped `best.pt`, step {manifest['checkpoint_step']}, SHA256 `{manifest['checkpoint_sha256']}`.", '',
        'Unchanged `testing_joint_fitting/contact_axis_orientation/candidate.py`; the configuration and dependency '
        'hashes match the preceding test1 run. Strength 128, position strength 4, hypothesis 0 only, fitting states '
        '2/3/4, original source-image pixel coordinates and correspondence masks, initial mesh points and normals. '
        'GT articulation enters scoring only. No new inference, fitter tuning, hypothesis oracle or improvement loop.', '',
        '| Metric | Mean | Median | P90 | N |', '|---|---:|---:|---:|---:|']
    for label, key in [('Axis orientation error °', 'axis'), ('Pivot-to-axis distance (world units)', 'pivot'),
        ('Rotation error °', 'rotation'), ('Fit reprojection RMSE (source pixels)', 'reprojection_rmse_pixels'),
        ('Input trajectory ADE (source pixels, states 1–4)', 'trajectory_ADE_source_pixels')]:
        d = aggregate[key]
        lines.append(f"| {label} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['p90'])} | {d['n']} |")
    lines += ['', f"Axis error ≤10°: {100*aggregate['success_10_deg']:.2f}%; "
        f"≤15°: {100*aggregate['success_15_deg']:.2f}%. Denominator: all {len(rows)} considered pairs, including unavailable fits.",
        f"Mean axis error with 90° assigned to unavailable fits: {fmt(aggregate['axis_with_failure_penalty']['mean'])}°.", '',
        '## By source joint order', '',
        '| Joint | Considered | Fitted | Axis mean ° | Axis median ° | Pivot mean | Rotation mean ° | Within 15° |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ordinal, s in result['by_joint'].items():
        lines.append(f"| {ordinal} | {s['n']} | {s['n']-s['failures']} | {fmt(s['axis']['mean'],4)} | "
            f"{fmt(s['axis']['median'],4)} | {fmt(s['pivot']['mean'])} | {fmt(s['rotation']['mean'],4)} | {100*s['success_15_deg']:.2f}% |")
    lines += ['', 'Joint order is source joint index + 1, without renumbering after fixed-joint exclusions.', '',
        '## Same cropped checkpoint: test1 versus validation', '',
        '| Metric mean | Test1 | Validation |', '|---|---:|---:|']
    for label, key in [('Axis error °', 'axis'), ('Failure-inclusive axis error °', 'axis_with_failure_penalty'),
        ('Pivot distance', 'pivot'), ('Rotation error °', 'rotation')]:
        lines.append(f"| {label} | {fmt(test['overall'][key]['mean'])} | {fmt(aggregate[key]['mean'])} |")
    lines += ['', f"Validation/test1 object-ID overlap: {cohort['test1_object_overlap']}. "
        'The validation objects differ from the test1 objects used during fitter development. '
        'Raw split comparisons also reflect different object/joint distributions.', '',
        '## Matched previous uncropped validation evaluation', '',
        f"{comparison['n']} pairs matched by original scene filename, source joint order and object identity. "
        'Local scene indices differ between the old 50-scene subset and this 80-scene archive.', '',
        '| Metric mean | Uncropped, step 108000 | Cropped, step 86000 |', '|---|---:|---:|']
    for label, key in [('Axis error °', 'axis'), ('Pivot distance', 'pivot'), ('Rotation error °', 'rotation')]:
        lines.append(f"| {label} | {fmt(comparison['old'][key]['mean'])} | {fmt(comparison['cropped'][key]['mean'])} |")
    lines += ['', comparison['caveat'], '', '## Per-object results', '',
        '| Object ID | Considered | Fitted | Mean axis ° | Median axis ° | Mean pivot | Mean rotation ° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for obj, s in result['by_object'].items():
        lines.append(f"| {obj} | {s['n']} | {s['n']-s['failures']} | {fmt(s['axis']['mean'],4)} | "
            f"{fmt(s['axis']['median'],4)} | {fmt(s['pivot']['mean'])} | {fmt(s['rotation']['mean'],4)} |")
    lines += ['', '## Coverage and definitions', '', '```json', json.dumps(cohort, indent=2), '```', '',
        f"Insufficient-observation failures: {audit['insufficient_fitting_observations']}; other fitting failures: "
        f"{audit['other_fit_failures']}. `exclusions.json` lists fixed/unobservable joints and pairs lacking future targets. "
        'Each failed pair remains in the coverage denominator and failure-inclusive axis metric. '
        'Distribution means/medians use successful fits with finite metric references.', '',
        'Axis error is sign-invariant, 0–90°. Pivot error is the perpendicular distance from the '
        'transform-consistent GT pivot to the fitted infinite axis line, in dataset world units. '
        'Rotation error averages SO(3) errors over states 2–4. Each joint pair has equal weight. '
        'Candidate assumes revolute joints; joint-type classification is not evaluated. '
        'The original correspondence mask can reflect GT visibility, as in the test1 evaluation.', '',
        '## Saved artifacts and verification', '',
        '`rows.json` and `pairs/pair_XXXXX.json` retain fitted axes, pivots, angles, transforms, solver diagnostics, '
        'scoring references, and original prediction paths/hashes. Paths are relative to the archive `val/` directory. '
        'Images, masks, all hypotheses, trajectories, calibration, and meshes remain in the inference archive. '
        '`summary.json` holds per-joint/object/scene metrics; `outliers.json` contains the largest axis errors. '
        '`protocol.json` pins configuration, inference records and code hashes. `audit.json` independently checks '
        'axis/pivot/rotation errors, reprojection RMSE, transform reconstruction, unit axes, positive scored depth, '
        'saved pair files and input hashes.', '',
        f"Audit: **{audit['status']}**. Elapsed evaluation time: {elapsed:.1f}s; summed CPU fitting time: "
        f"{result['cpu_fit_seconds']:.1f}s."]
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    start = time.perf_counter()
    archive = args.archive.resolve()
    output = archive / 'candidate_val'
    manifest = json.loads((archive / 'run.json').read_text())
    assert json.loads((archive / 'audit.json').read_text())['status'] == 'passed'
    split = manifest['splits']['val']
    paths = sorted((archive / 'val/records').glob('*.json'))
    records = [json.loads(p.read_text()) for p in paths]
    assert len(split['scenes']) == 80 and len(records) == len(split['pairs']) == 222
    assert manifest['checkpoint_step'] == 86000
    sources = frozen_sources()
    test_protocol = json.loads((archive / 'candidate_test1/protocol.json').read_text())
    assert sources == test_protocol['algorithm_sources'], 'Candidate differs from test1 run'
    assert CONFIG == test_protocol['config']
    assert manifest['checkpoint_sha256'] == test_protocol['cohort']['checkpoint_sha256']
    selected = [r for r in records if r['gt_type'] == 'revolute' and r['scoreable']]
    assert len({r['pair_index'] for r in records}) == len(records)
    for r in records:
        assert r['split'] == 'val' and r['status'] == 'predicted'
    val_ids = {r['object_id'] for r in selected}
    test_ids = {r['object_id'] for r in json.loads((archive / 'candidate_test1/rows.json').read_text())}
    cohort = dict(split='val', scenes=80, source_joints=len(split['inventory']),
        source_kind_counts=dict(Counter(r['kind'] for r in split['inventory'])),
        eligible_predictions=len(records), scored_revolute_pairs=len(selected),
        scored_scenes=len({r['scene_index'] for r in selected}), objects=len(val_ids),
        test1_object_overlap=sorted(val_ids & test_ids),
        fixed_pairs_excluded=sum(r['gt_type']=='fixed' for r in records),
        no_future_targets_excluded=sum(r['gt_type']=='revolute' and not r['scoreable'] for r in records),
        initially_unobservable_joints=sum(not r['eligible'] for r in split['inventory']),
        checkpoint_sha256=manifest['checkpoint_sha256'])
    protocol = dict(archive=str(archive), split='val', run_manifest_sha256=digest(archive/'run.json'),
        record_hashes={p.name: digest(p) for p in paths}, algorithm_sources=sources,
        runner_sha256=digest(Path(__file__)), config=CONFIG, primary_hypothesis=0,
        fitting_states=[2,3,4], initial_geometry_state=0, coordinates='source pixels / world geometry',
        tuning=False, cohort=cohort)
    if (output / 'protocol.json').exists():
        assert json.loads((output/'protocol.json').read_text()) == protocol
    write(output/'protocol.json', protocol)
    write(output/'exclusions.json', dict(predicted_pairs=[dict(pair_index=r['pair_index'],
        scene_index=r['scene_index'], joint_index=r['joint_index'],
        reason='fixed joint' if r['gt_type']=='fixed' else 'no valid future targets')
        for r in records if r['gt_type'] != 'revolute' or not r['scoreable']],
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
            for r in batch:
                write(output/'pairs'/f"pair_{r['pair_index']:05d}.json", r)
            if completed % 20 == 0:
                print(json.dumps(dict(event='progress', scenes=completed, pairs=len(rows),
                    seconds=time.perf_counter()-start)), flush=True)
    rows.sort(key=lambda r: r['pair_index'])
    assert len(rows) == len(selected) == len({r['pair_index'] for r in rows})
    audit = verify(output, archive, protocol, rows)
    result = report(output, archive, manifest, cohort, rows, time.perf_counter()-start, audit)
    print(json.dumps(dict(event='complete', summary=result['overall'], output=str(output))), flush=True)


if __name__ == '__main__':
    main()


