"""Independently verify all saved predictions and reported metric denominators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from finetuning_pair.evaluate import file_hash, write_json


def audit(out, periodic=None):
    out = Path(out)
    manifest = json.loads((out / 'run.json').read_text())
    size = manifest['train_config']['image_size']
    checked, unscored_checked, comparisons = 0, 0, []
    if file_hash(out / 'checkpoint.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash changed')
    previous = {}
    if periodic is not None and Path(periodic).exists():
        summary = json.loads(Path(periodic).read_text())
        if summary['checkpoint_step'] != manifest['checkpoint_step']:
            raise ValueError('Periodic reference uses another checkpoint step')
        previous = {r['index']:r['modes']['without'] for r in summary['scenes']}
    for path in sorted((out / 'records').glob('*.json')):
        row = json.loads(path.read_text())
        data = out / 'predictions' / f"{row['pair_index']:04d}_{row['mode']}.npz"
        if row.get('skipped') and not data.exists():
            continue
        with np.load(data, allow_pickle=False) as a:
            predictions = a['predictions'].astype(np.float64)
            truth, valid = a['ground_truth'].astype(np.float64), a['valid']
            start = int(a['known_states'])
        if predictions.shape != (manifest['ensemble_size'], *truth.shape) or not np.isfinite(predictions).all():
            raise ValueError(f'Invalid predictions: {data}')
        if row.get('skipped'):
            if valid[start:].any():
                raise ValueError('A valid future target was marked unscorable')
            unscored_checked += 1
            continue
        mask = valid[start:]
        errors = np.linalg.norm((predictions-truth[None])*(size-1), axis=-1)
        ade = errors[0,start:][mask].mean()
        mse = np.square(2*(predictions[:,start:]-truth[None,start:])).sum(-1)
        mse = (mse*mask).sum((1,2))/mask.sum()
        np.testing.assert_allclose(ade,row['ADE_1_pixels'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.square(errors[0,start:][mask]).mean()),
                                   row['RMSE_1_pixels'],rtol=1e-10,atol=1e-10)
        if valid[-1].any():
            np.testing.assert_allclose(errors[0,-1][valid[-1]].mean(),row['FDE_1_pixels'],rtol=1e-10,atol=1e-10)
        elif row['FDE_1_pixels'] is not None:
            raise ValueError('Undefined endpoint error was assigned a numeric value')
        np.testing.assert_allclose(mse[0],row['minADE_1'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(mse[:5].min(),row['minADE_5'],rtol=1e-10,atol=1e-10)
        path_length = np.linalg.norm(np.diff(truth,axis=0)*(size-1),axis=-1).sum(0)
        complete = valid.all(0)
        eligible = complete & (path_length >= manifest['min_path_pixels']) & (row['gt_type'] != 'fixed')
        if int(eligible.sum()) != row['normalized_tracks']:
            raise ValueError('Normalized-track coverage mismatch')
        if eligible.any():
            ratio = 100*errors[0,start:].mean(0)[eligible]/path_length[eligible]
            np.testing.assert_allclose(ratio.mean(),row['ADE_percent_full_path'],rtol=1e-10,atol=1e-10)
        elif row['ADE_percent_full_path'] is not None:
            raise ValueError('Undefined percentage was assigned a numeric value')
        if row['tracks_total'] != row['complete_tracks']+row['incomplete_tracks']:
            raise ValueError('Complete-track coverage mismatch')
        if row['complete_tracks'] != row['normalized_tracks']+row['fixed_tracks_excluded']+row['short_path_tracks_excluded']:
            raise ValueError('Percentage-exclusion accounting mismatch')
        if row.get('axis_error_deg') is not None:
            axis, gt = np.array(row['estimated_axis_world']), np.array(row['gt_axis_world'])
            angle = np.rad2deg(np.arccos(np.clip(abs(axis@gt)/(np.linalg.norm(axis)*np.linalg.norm(gt)),0,1)))
            np.testing.assert_allclose(angle,row['axis_error_deg'],rtol=1e-7,atol=1e-5)
        for metric,axis_key,pivot_key,origin in (
            ('pivot_distance','estimated_axis_world','estimated_pivot_world',row.get('gt_pivot_world')),
            ('kinematic_pivot_distance','estimated_axis_world','estimated_pivot_world',row.get('kinematic_reference',{}).get('pivot')),
            ('forced_revolute_pivot_distance','forced_revolute_axis_world','forced_revolute_pivot_world',row.get('gt_pivot_world')),
            ('forced_revolute_kinematic_pivot_distance','forced_revolute_axis_world','forced_revolute_pivot_world',row.get('kinematic_reference',{}).get('pivot'))):
            if row.get(metric) is not None:
                axis=np.asarray(row[axis_key]); axis=axis/np.linalg.norm(axis)
                distance=np.linalg.norm(np.cross(np.asarray(origin)-np.asarray(row[pivot_key]),axis))
                np.testing.assert_allclose(distance,row[metric],rtol=1e-10,atol=1e-10)
        if row['mode'] == 'without_poke' and row['pair_index'] in previous:
            old = previous[row['pair_index']]
            comparisons.append(dict(pair_index=row['pair_index'],delta_minADE_1=row['minADE_1']-old['minADE_1'],
                                    delta_minADE_5=row['minADE_5']-old['minADE_5']))
        checked += 1
    expected = len(manifest['pairs'])*len(manifest['modes'])
    rows_count = len(list((out/'records').glob('*.json')))
    if rows_count != expected:
        raise ValueError(f'Incomplete records: {rows_count}/{expected}')
    result = dict(status='passed',records=rows_count,scored_arrays_checked=checked,
                  unscorable_prediction_arrays_checked=unscored_checked,
                  checkpoint_sha256_verified=True,periodic_comparisons=comparisons,
                  periodic_max_abs_difference=max([abs(c[k]) for c in comparisons for k in ('delta_minADE_1','delta_minADE_5')],default=None))
    write_json(out/'audit.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--periodic-reference',type=Path)
    args=parser.parse_args()
    audit(args.output_dir,args.periodic_reference)
