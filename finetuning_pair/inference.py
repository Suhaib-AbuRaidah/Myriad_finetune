"""Predict child-relative rollouts from current RGB, pair masks and query points.

No dataset scene, depth, future image, joint tree or ground-truth transform is
needed. Source-pixel queries are cropped/resized identically to training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from finetuning_pair.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
from finetuning_pair.conditioning import blend_roles, crop_geometry, normalized_pixels
from finetuning_pair.dataset import _image_tensor
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.model_adapter import FineTuningModel, LoRAConfig
from finetuning_pair.paths import artifact_path
from finetuning_pair.testing import sample_rollout
from finetuning_pair.train import build_myriad, _resolve_device
from finetuning_pair.visualizer import render_rollout


def prepare_inputs(rgb, parent_mask, child_mask, points, *, image_size=384,
                   overlay_alpha=0.85, point_space='source-pixels', max_track_ids=256):
    """Prepare exactly the same conditioning as training, with child-only queries."""
    rgb = np.asarray(rgb)
    parent = np.asarray(parent_mask, dtype=bool)
    child = np.asarray(child_mask, dtype=bool)
    conditioned = blend_roles(rgb, parent, child, overlay_alpha)
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= max_track_ids
            or not np.isfinite(points).all()):
        raise ValueError('Query points must be finite [N,2], 1 <= N <= max_track_ids')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Query points must be unique child points')
    h, w = rgb.shape[:2]
    top, left, side = crop_geometry(h, w)
    if point_space == 'source-pixels':
        pixels = points
        positions = normalized_pixels(pixels, h, w, image_size)
    elif point_space == 'normalized-model':
        positions = points
        pixels = ((points * (image_size - 1) + 0.5) * side / image_size - 0.5) + [left, top]
    else:
        raise ValueError('point_space must be source-pixels or normalized-model')
    if ((positions < 0) | (positions > 1)).any():
        raise ValueError('Query points must lie inside the resized center crop')
    px = np.rint(pixels[:, 0]).astype(int)
    py = np.rint(pixels[:, 1]).astype(int)
    if np.any(px < 0) or np.any(px >= w) or np.any(py < 0) or np.any(py >= h) or not child[py, px].all():
        raise ValueError('Every query point must be initially visible inside the selected child mask')
    image = _image_tensor(conditioned, top, left, side, image_size)
    return image, torch.from_numpy(positions.astype(np.float32))[None]


def read_mask(path):
    if path.suffix.lower() == '.npy':
        mask = np.load(path, allow_pickle=False)
        if mask.ndim != 2:
            raise ValueError('Mask .npy files must be [H,W]')
        return mask != 0
    return np.asarray(Image.open(path).convert('L')) > 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--parent-mask', type=Path, required=True, help='Visible parent mask, PNG or .npy')
    parser.add_argument('--child-mask', type=Path, required=True, help='Visible child mask, PNG or .npy')
    parser.add_argument('--points', type=Path, required=True, help='.npy [N,2] child query points')
    parser.add_argument('--point-space', choices=('source-pixels', 'normalized-model'), default='source-pixels')
    parser.add_argument('--state-times', help='Motion progress times; default comes from checkpoint')
    parser.add_argument('--output-dir', type=Path, default=Path('finetuning_pair/runs/inference'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--ensemble-size', type=int, default=5)
    parser.add_argument('--diffusion-steps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp-dtype', choices=('auto', 'none', 'bf16'), default='auto')
    args = parser.parse_args(argv)
    args.output_dir = artifact_path(args.output_dir)
    if args.ensemble_size < 1 or args.diffusion_steps < 1 or args.seed < 0:
        parser.error('Need positive ensemble/steps and a nonnegative seed')
    return args


def main(argv=None):
    args = parse_args(argv)
    info = inspect_checkpoint(args.checkpoint)
    cfg = info['train_config']
    image, known = prepare_inputs(
        np.asarray(Image.open(args.image).convert('RGB')),
        read_mask(args.parent_mask), read_mask(args.child_mask),
        np.load(args.points, allow_pickle=False), image_size=cfg['image_size'],
        overlay_alpha=cfg['overlay_alpha'], point_space=args.point_space)
    if args.state_times:
        times = np.array([float(t) for t in args.state_times.split(',')], dtype=np.float32)
    else:
        times = (np.rint(np.linspace(0, 4, cfg['num_transitions'] + 1)) * (cfg['time_scale'] / 4)).astype(np.float32)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or times[0] != 0:
        raise ValueError('State times must start at zero, be finite and strictly increasing')
    device = _resolve_device(args.device)
    bare = build_myriad()
    reference = load_pretrained_weights(bare, Path(info['base_checkpoint']['path']))
    model = FineTuningModel(bare, LoRAConfig(**info['lora_config'])).to(device)
    step = load_finetuned_weights(args.checkpoint, model, base_checkpoint=reference)
    times = torch.from_numpy(times)
    predicted = sample_rollout(model, image, known, times, ensemble_size=args.ensemble_size,
                               diffusion_steps=args.diffusion_steps, seed=args.seed, amp_dtype=args.amp_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / 'predictions.npz',
                        predictions=predicted.detach().cpu().numpy(), known_positions=known.numpy(),
                        state_times=times.numpy(), image=image.numpy(), target_definition=np.array(TARGET_DEFINITION))
    result = dict(target_definition=TARGET_DEFINITION, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_step=step, points=int(known.shape[1]), overlay_alpha=cfg['overlay_alpha'],
                  ensemble_size=args.ensemble_size, diffusion_steps=args.diffusion_steps)
    result['artifacts'] = render_rollout(image, predicted, torch.ones(known.shape[1], dtype=torch.long),
                                         args.output_dir, observed_states=1, title='Child relative to frozen initial parent')
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
