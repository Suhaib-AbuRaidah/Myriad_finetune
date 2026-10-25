"""Evaluate the original revolute fitter on the same tracks; export four tables."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from finetuning_pair.analyze_joints import fit_joint
from testing_joint_fitting.axis_motion.common import score, summary
from testing_joint_fitting.experiment import write
from testing_joint_fitting.contact_axis_orientation.validation_transfer.evaluate import frozen_sources
from .evaluate import DEFAULT, digest


def worker(job):
    archive, previous = job
    case = {k: previous[k] for k in ('pair_index', 'scene_index', 'joint_ordinal', 'object_id')}
    case.update(gt_axis=np.asarray(previous['gt_axis']),
        gt_pivot=None if previous['gt_pivot'] is None else np.asarray(previous['gt_pivot']),
        gt_rotations=np.asarray(previous['gt_transforms'])[:, :3, :3])
    path = Path(archive) / 'test1' / previous['prediction_file']
    assert digest(path) == previous['prediction_sha256']
    with np.load(path, allow_pickle=False) as a:
        xyz, uv, valid, k, e = (a[key] for key in ('reference_world',
            'predictions_source_pixels', 'valid', 'camera_intrinsic', 'camera_extrinsic'))
    fit, exception = None, None
    started = time.perf_counter()
    if valid[2:5].sum() >= 8:
        try:
            result = fit_joint(xyz.astype(float), uv[0, 2:5].astype(float),
                valid[2:5].astype(bool), k.astype(float), e.astype(float))
            # Match the assumed-revolute cohort and the original baseline in
            # contact_axis_orientation/solver.py. No GT parameter enters fit_joint.
            fit = None if result is None else result['fits'].get('revolute')
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
            exception = repr(error)
    else:
        exception = 'Fewer than eight valid fitting observations'
    row = score(case, fit)
    row.update(fit_seconds=time.perf_counter()-started, exception=exception,
        prediction_file=previous['prediction_file'], prediction_sha256=previous['prediction_sha256'],
        rmse_px=None if fit is None else fit['rmse_px'])
    if fit is not None:
        p = np.asarray(row['params'])
        axis, target = p[:3]/np.linalg.norm(p[:3]), case['gt_axis']/np.linalg.norm(case['gt_axis'])
        independent = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, target)), abs(axis @ target))))
        assert abs(independent-row['axis_error_deg']) < 1e-5
        if case['gt_pivot'] is not None:
            distance = float(np.linalg.norm(np.cross(case['gt_pivot']-p[3:6], axis)))
            assert abs(distance-row['pivot_distance']) < 1e-10
    return row


def latex_tables(algorithms):
    tables = [r'% Requires \usepackage{booktabs}; axis degrees, pivot dataset world units.',
        '% Same cropped checkpoint (step 86000), test1, hypothesis 0, fitting states 2--4.',
        '% Matched successful revolute fits; fixed joints excluded.']
    for name, data in algorithms.items():
        title = 'Original fitting algorithm' if name == 'original' else r'Candidate fitting algorithm'
        s = data['overall']
        tables += ['', r'\begin{table}[ht]', r'\centering',
            r'\caption{'+title+r' on test1: all joints together.}',
            r'\label{tab:'+name+r'_fitting_overall}', r'\begin{tabular}{lc}',
            r'\toprule', r'Measure & Value \\', r'\midrule',
            rf"Mean axis orientation error $\downarrow$ & {s['axis']['mean']:.3f}$^\circ$ \\",
            rf"90th percentile axis orientation error $\downarrow$ & {s['axis']['p90']:.3f}$^\circ$ \\",
            r'\midrule',
            rf"Mean pivot distance $\downarrow$ & {s['pivot']['mean']:.5f} \\",
            rf"90th percentile pivot distance $\downarrow$ & {s['pivot']['p90']:.5f} \\",
            r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
            r'\begin{table}[ht]', r'\centering',
            r'\caption{'+title+r' on test1: results by joint order.}',
            r'\label{tab:'+name+r'_fitting_per_joint}', r'\begin{tabular}{lrrrr}',
            r'\toprule',
            r'Joint & Mean axis ($^\circ$) $\downarrow$ & P90 axis ($^\circ$) $\downarrow$ & Mean pivot $\downarrow$ & P90 pivot $\downarrow$ \\',
            r'\midrule']
        for order, stats in data['by_joint'].items():
            ordinal = {'1':'First', '2':'Second', '3':'Third', '4':'Fourth'}.get(order, order)
            tables.append(rf"{ordinal} & {stats['axis']['mean']:.3f} & {stats['axis']['p90']:.3f} & {stats['pivot']['mean']:.5f} & {stats['pivot']['p90']:.5f} \\")
        tables += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(tables)+'\n'


def main():
    archive = DEFAULT.resolve()
    output = archive/'original_test1'
    previous = json.loads((archive/'candidate_test1/rows.json').read_text())
    protocol = json.loads((archive/'candidate_test1/protocol.json').read_text())
    sources = frozen_sources()
    assert sources == protocol['algorithm_sources']
    assert digest(archive/'run.json') == protocol['run_manifest_sha256']
    fingerprint = dict(algorithm_sources=sources, checkpoint_sha256=protocol['cohort']['checkpoint_sha256'],
        candidate_rows_sha256=digest(archive/'candidate_test1/rows.json'),
        runner_sha256=digest(Path(__file__)), primary_hypothesis=0, fitting_states=[2,3,4],
        baseline='finetuning_pair.analyze_joints.fit_joint -> fits[revolute]',
        classification='not evaluated; assumed-revolute cohort matches candidate')
    if (output/'protocol.json').exists():
        assert json.loads((output/'protocol.json').read_text()) == fingerprint
    write(output/'protocol.json', fingerprint)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(worker, [(str(archive), r) for r in previous], chunksize=8):
            rows.append(row)
            if len(rows)%200 == 0:
                print(json.dumps(dict(event='baseline_progress', pairs=len(rows))), flush=True)
    assert len(rows) == len(previous) and frozen_sources() == sources
    write(output/'rows.json', rows)
    write(output/'summary.json', dict(overall=summary(rows), wall_seconds=time.perf_counter()-started,
        by_joint={str(j):summary([r for r in rows if r['joint_ordinal']==j])
            for j in sorted({r['joint_ordinal'] for r in rows})}))
    old = {r['pair_index']:r for r in rows}
    new = {r['pair_index']:r for r in previous}
    matched = sorted(i for i in old if not old[i]['failed'] and not new[i]['failed'])
    algorithms = {}
    for name, indexed in (('original', old), ('candidate', new)):
        selected = [indexed[i] for i in matched]
        algorithms[name] = dict(overall=summary(selected),
            by_joint={str(j):summary([r for r in selected if r['joint_ordinal']==j])
                for j in sorted({r['joint_ordinal'] for r in selected})})
    comparison = dict(split='test1', checkpoint_step=86000, algorithms=algorithms,
        pairs_considered=len(rows), matched_successful_pairs=len(matched),
        matched_pair_indices=matched, original_failures=sum(r['failed'] for r in rows),
        candidate_failures=sum(r['failed'] for r in previous),
        axis_unit='degrees, sign-invariant', pivot_unit='dataset world units',
        pivot_definition='GT transform-consistent pivot to fitted infinite axis line',
        axis_and_pivot_independent_recalculations='passed', algorithm_sources_unchanged=True)
    write(archive/'fitting_comparison_tables.json', comparison)
    (archive/'fitting_comparison_tables.tex').write_text(latex_tables(algorithms))
    print(json.dumps(dict(event='complete', comparison=comparison)), flush=True)


if __name__ == '__main__':
    main()
