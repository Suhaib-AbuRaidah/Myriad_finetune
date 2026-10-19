"""Retain inference for initially visible points with no valid future GT score.

These records remain excluded from metric denominators. With-poke cases with
no points valid at both observed states cannot supply that diagnostic input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from finetuning_pair.evaluate import write_json
from finetuning_pair.geometry import TARGET_DEFINITION
from training_from_scratch.dataset import _backproject
from training_from_scratch.testing import sample_rollout
from .run import make_dataset, load_cached_weights, atomic_npz


def complete(out, device="cuda:1"):
    out=Path(out)
    manifest=json.loads((out/'run.json').read_text())
    cfg=manifest['train_config']
    args=argparse.Namespace(data_root=None,split=manifest['split'],seed=manifest['seed'],
                            cache_dir=Path('training_from_scratch/cache'))
    dataset=make_dataset(args,cfg)
    model=None
    filled=[]; unavailable=[]
    for path in sorted((out/'records').glob('*.json')):
        row=json.loads(path.read_text())
        if not row.get('skipped'):
            continue
        prediction=out/'predictions'/f"{row['pair_index']:04d}_{row['mode']}.npz"
        if prediction.exists():
            continue
        sample=dataset[row['pair_index']]
        start=2 if row['mode']=='with_poke' else 1
        selected=sample['point_valid'] & sample['position_valid'][:start].all(0)
        if not bool(selected.any()):
            row.update(inference_completed=False,reason='No points valid in all required observed states')
            write_json(path,row)
            unavailable.append([row['pair_index'],row['mode']])
            continue
        if model is None:
            torch.cuda.set_device(torch.device(device))
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            model=load_cached_weights(out/'checkpoint.pt',torch.device(device),Path(manifest['feature_cache']))
        started=time.perf_counter()
        truth=sample['trajectories'][:,selected]
        pred=sample_rollout(model,sample['x'],truth[:start],sample['state_times'],
            ensemble_size=manifest['ensemble_size'],diffusion_steps=manifest['diffusion_steps'],
            seed=manifest['seed']+row['pair_index'],amp_dtype='none').float().cpu().numpy()
        valid=sample['position_valid'][:,selected].numpy()
        source_file=f"sources/scene_{row['scene_index']:04d}.npz"
        with np.load(out/source_file,allow_pickle=False) as source:
            pixels=sample['reference_pixels'][selected].numpy().astype(float)
            depth=source['initial_depth'][np.rint(pixels[:,1]).astype(int),np.rint(pixels[:,0]).astype(int)]
            xyz=_backproject(pixels,depth,source['camera_intrinsic'],source['camera_extrinsic'])
        atomic_npz(prediction,predictions=pred,ground_truth=truth.numpy(),valid=valid,
            known_states=np.array(start),image=sample['x'].numpy(),link_nodes=sample['point_link_nodes'][selected].numpy(),
            reference_world=xyz,source_pixels=pixels,state_times=sample['state_times'].numpy(),
            pair_labels=sample['pair_labels'].numpy(),joint_index=np.array(row['joint_index']),
            point_indices=np.flatnonzero(selected.numpy()),target_definition=np.array(TARGET_DEFINITION),
            source_file=np.array(source_file))
        row.update(inference_completed=True,reason='Inference saved; no valid future GT positions to score',
                   inference_seconds=time.perf_counter()-started)
        write_json(path,row)
        filled.append([row['pair_index'],row['mode']])
    result=dict(filled_unscorable_predictions=filled,unavailable_observed_points=unavailable)
    write_json(out/'unscorable_inference.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    complete(args.output_dir,args.device)
