"""Rank complete saved candidate scenes with strict per-joint error limits."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


DEFAULT = Path(__file__).resolve().parents[2]/'training_from_scratch_cropped/runs/full_test1_val80_20261021'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--max-axis-deg', type=float, default=2.)
    parser.add_argument('--max-pivot', type=float, default=.03)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.max_axis_deg, args.max_pivot)):
        parser.error('Both thresholds must be finite and positive')
    root = args.run_dir.expanduser().resolve()
    fit_root = root/'candidate_test1'
    manifest = json.loads((root/'run.json').read_text())
    protocol = json.loads((fit_root/'protocol.json').read_text())
    assert manifest['checkpoint_sha256'] == protocol['cohort']['checkpoint_sha256']
    assert digest(root/'run.json') == protocol['run_manifest_sha256']
    for source, expected in protocol['algorithm_sources'].items():
        assert digest(Path(__file__).resolve().parents[2]/source) == expected, source
    assert json.loads((fit_root/'audit.json').read_text())['status'] == 'passed'
    fits = json.loads((fit_root/'rows.json').read_text())
    keyed = {(r['scene_index'], r['joint_index']):r for r in fits}
    assert len(keyed) == len(fits)
    inventory = manifest['splits']['test1']['inventory']
    ranked, rejected = [], []
    for scene in manifest['splits']['test1']['scenes']:
        index = scene['scene_index']
        joints = [r for r in inventory if r['scene_index'] == index and r['kind'] == 'revolute']
        failures, rows = [], []
        if not joints:
            failures.append('no revolute joints')
        for joint in joints:
            row = keyed.get((index, joint['joint_index']))
            if row is None:
                failures.append(f"joint {joint['joint_ordinal']}: missing saved fit")
            elif row['failed']:
                failures.append(f"joint {joint['joint_ordinal']}: failed or insufficient-observation fit")
            elif not all(row.get(k) is not None and math.isfinite(row[k]) for k in ('axis_error_deg','pivot_distance')):
                failures.append(f"joint {joint['joint_ordinal']}: undefined error")
            else:
                rows.append(row)
        if failures:
            rejected.append(dict(scene_index=index, category='incomplete', reasons=failures))
            continue
        maxima = dict(axis=max(r['axis_error_deg'] for r in rows), pivot=max(r['pivot_distance'] for r in rows))
        if maxima['axis'] > args.max_axis_deg or maxima['pivot'] > args.max_pivot:
            rejected.append(dict(scene_index=index, category='threshold', maximum_axis_deg=maxima['axis'], maximum_pivot=maxima['pivot']))
            continue
        axis = statistics.mean(r['axis_error_deg'] for r in rows)
        pivot = statistics.mean(r['pivot_distance'] for r in rows)
        rotation = statistics.mean(r['rotation_error_deg'] for r in rows)
        ranked.append(dict(scene_index=index, original_scene_index=scene['original_scene_index'],
            object_id=scene['object_id'], sample_id=scene['sample_id'], joints=len(rows),
            mean_axis_deg=axis, max_axis_deg=maxima['axis'], mean_pivot=pivot, max_pivot=maxima['pivot'],
            mean_rotation_deg=rotation,
            score=.5*(axis/args.max_axis_deg+pivot/args.max_pivot),
            fixed_joints_excluded=sum(r['scene_index']==index and r['kind']=='fixed' for r in inventory),
            per_joint=[{k:r[k] for k in ('joint_index','joint_ordinal','pair_index',
                'axis_error_deg','pivot_distance','rotation_error_deg')} for r in rows]))
    ranked.sort(key=lambda r:(r['score'], r['mean_axis_deg'], r['mean_pivot'], r['scene_index']))
    for i, row in enumerate(ranked,1):
        row['rank'] = i
    result = dict(algorithm='testing_joint_fitting/contact_axis_orientation/candidate.py',
        checkpoint_sha256=manifest['checkpoint_sha256'], checkpoint_step=manifest['checkpoint_step'],
        input_rows_sha256=digest(fit_root/'rows.json'), max_axis_deg=args.max_axis_deg, max_pivot=args.max_pivot,
        score='0.5 * (mean_axis_deg / max_axis_deg + mean_pivot / max_pivot); lower is better',
        cohort='All source revolute joints must have successful fits and finite errors; fixed joints excluded',
        scenes_considered=len(manifest['splits']['test1']['scenes']), scenes_passing=len(ranked),
        rejection_counts=dict(Counter(r['category'] for r in rejected)), ranked=ranked, rejected=rejected)
    tag = f"axis{args.max_axis_deg:g}_pivot{args.max_pivot:g}".replace('.','p')
    output = fit_root/f'best_examples_{tag}'
    output.mkdir(parents=True, exist_ok=True)
    (output/'ranking.json').write_text(json.dumps(result,indent=2)+'\n')
    fields = ('rank','scene_index','original_scene_index','joints','mean_axis_deg','max_axis_deg',
        'mean_pivot','max_pivot','mean_rotation_deg','score','object_id','sample_id')
    with (output/'ranking.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ranked)
    lines = ['# Best complete candidate test1 examples', '',
        f"Algorithm: `{result['algorithm']}`; cropped checkpoint step {manifest['checkpoint_step']}; hypothesis 0.", '',
        f"Every revolute joint must have axis error ≤{args.max_axis_deg:g}° and pivot distance ≤{args.max_pivot:g} world units. "
        'All source revolute joints are required, including initially unobservable ones: a missing fit rejects the whole scene. '
        'Fixed joints have no motion axis and are excluded. No inference, fitting or visualization was run.', '',
        f"Passing: **{len(ranked)} / {result['scenes_considered']} scenes**. Rejected counts: {result['rejection_counts']}.", '',
        'Rank by the average of mean axis error divided by its threshold and mean pivot distance divided by its threshold. '
        'This gives the two metrics equal weight without adding incompatible units. Means weight joints equally. '
        'Rotation error is shown for context and is not part of the requested filter/ranking.', '',
        '| Rank | Scene index | Revolute joints | Mean axis ° | Max axis ° | Mean pivot | Max pivot | Mean rotation ° |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in ranked:
        lines.append(f"| {r['rank']} | {r['scene_index']} | {r['joints']} | {r['mean_axis_deg']:.4f} | "
            f"{r['max_axis_deg']:.4f} | {r['mean_pivot']:.6f} | {r['max_pivot']:.6f} | {r['mean_rotation_deg']:.3f} |")
    lines += ['', '## Per-joint values', '',
        '| Scene | Joint order | Pair index | Axis error ° | Pivot distance | Rotation error ° |',
        '|---|---:|---:|---:|---:|---:|']
    for r in ranked:
        for j in r['per_joint']:
            lines.append(f"| {r['scene_index']} | {j['joint_ordinal']} | {j['pair_index']} | "
                f"{j['axis_error_deg']:.5f} | {j['pivot_distance']:.6f} | {j['rotation_error_deg']:.3f} |")
    lines += ['', '## Visualize', '', 'Indices are the archive scene indices accepted by `visualize_saved`.', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m testing_joint_fitting.contact_axis_orientation.visualize_saved \\',
        f"  --scene-index {ranked[0]['scene_index'] if ranked else 0} --cloud mesh --show-gt", '```', '',
        '`ranking.json` retains all qualifying scenes, per-joint results, object/sample identifiers and rejected scene reasons. '
        '`ranking.csv` contains the ranked scene aggregates. These are deliberately score-selected presentation examples.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Check saved coverage/threshold claims independently of the sorting code.
    for r in ranked:
        assert len(r['per_joint']) == sum(j['scene_index']==r['scene_index'] and j['kind']=='revolute' for j in inventory)
        assert all(j['axis_error_deg']<=args.max_axis_deg and j['pivot_distance']<=args.max_pivot for j in r['per_joint'])
    assert len(ranked)+len(rejected) == result['scenes_considered']
    assert digest(fit_root/'rows.json') == result['input_rows_sha256']
    print(json.dumps(dict(output=str(output), scenes_passing=len(ranked),
        rejection_counts=result['rejection_counts'], ranked=[{k:r[k] for k in fields} for r in ranked]),indent=2))


if __name__ == '__main__':
    main()
