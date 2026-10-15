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
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.paths import artifact_path


def prepare_inputs(rgb, parent_mask, child_mask, points, *, image_size=384,
                   overlay_alpha=0.85, point_space='source-pixels', max_track_ids=256):
    """Prepare exactly the same conditioning as training, with child-only queries."""
    import torch
    from finetuning_pair.dataset import _image_tensor
    from finetuning_pair.conditioning import blend_roles, crop_geometry, normalized_pixels
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
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--parent-mask', type=Path, help='Visible parent mask, PNG or .npy')
    parser.add_argument('--child-mask', type=Path, help='Visible child mask, PNG or .npy')
    parser.add_argument('--points', type=Path, help='.npy [N,2] child query points')
    parser.add_argument('--point-space', choices=('source-pixels', 'normalized-model'), default='source-pixels')
    parser.add_argument('--state-times', help='Motion progress times; default comes from checkpoint')
    parser.add_argument('--output-dir', type=Path, default=Path('finetuning_pair/runs/inference'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--ensemble-size', type=int, default=5)
    parser.add_argument('--diffusion-steps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp-dtype', choices=('auto', 'none', 'bf16'), default='auto')
    parser.add_argument('--saved-run', type=Path, help='Replay saved RGB-D predictions and all joints, without loading weights')
    parser.add_argument('--mode', choices=('with_poke','without_poke'), default='with_poke')
    parser.add_argument('--scene-index', type=int, help='Saved scene to display (all its eligible joints)')
    parser.add_argument('--top-k', type=int, help='Browse the best K complete scenes by mean axis error')
    parser.add_argument('--show-pointcloud', action='store_true', help='Open a LIVE Open3D window with axes and all joint trajectories')
    parser.add_argument('--list-top', action='store_true', help='Write/print ranked saved scenes without opening a window')
    parser.add_argument('--depth', type=Path, help='Current image depth .npy in world units (live RGB-D inference)')
    parser.add_argument('--intrinsics', type=Path, help='Camera intrinsic .npy [3,3]')
    parser.add_argument('--extrinsics', type=Path, help='T_camera_world .npy [4,4]; default identity')
    parser.add_argument('--object-mask', type=Path, help='Whole-object mask for live RGB-D cloud; default parent | child')
    parser.add_argument('--viewer-test-seconds',type=float,default=0,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir = artifact_path(args.output_dir)
    if args.ensemble_size < 1 or args.diffusion_steps < 1 or args.seed < 0:
        parser.error('Need positive ensemble/steps and a nonnegative seed')
    if args.saved_run:
        if args.top_k is not None and args.top_k < 1:
            parser.error('--top-k must be positive')
        if args.top_k is not None and args.scene_index is not None:
            parser.error('Choose --top-k or --scene-index')
        if not args.show_pointcloud and not args.list_top:
            parser.error('Use --show-pointcloud or --list-top with --saved-run')
    else:
        missing=[flag for flag in ('checkpoint','image','parent_mask','child_mask','points') if getattr(args,flag) is None]
        if missing:
            parser.error('Raw inference requires '+', '.join('--'+flag.replace('_','-') for flag in missing))
        if args.top_k is not None or args.scene_index is not None or args.list_top:
            parser.error('--top-k, --scene-index and --list-top require --saved-run')
        if args.show_pointcloud and (args.depth is None or args.intrinsics is None):
            parser.error('Live RGB-D inference requires --depth and --intrinsics')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.saved_run:
        from finetuning_pair.pointcloud_viewer import show_saved_run
        output=None if args.output_dir == artifact_path('finetuning_pair/runs/inference') else args.output_dir
        show_saved_run(args.saved_run, args.mode, args.top_k, args.scene_index, output,
                       args.list_top, args.viewer_test_seconds)
        return
    import torch
    from finetuning_pair.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
    from finetuning_pair.model_adapter import FineTuningModel, LoRAConfig
    from finetuning_pair.testing import sample_rollout
    from finetuning_pair.train import build_myriad, _resolve_device
    from finetuning_pair.visualizer import render_rollout
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
    if args.show_pointcloud:
        from finetuning_pair.pointcloud_viewer import show_current_prediction
        show_current_prediction(np.asarray(Image.open(args.image).convert('RGB')),
            np.load(args.depth,allow_pickle=False),read_mask(args.object_mask) if args.object_mask else read_mask(args.parent_mask)|read_mask(args.child_mask),
            np.load(args.intrinsics,allow_pickle=False),np.load(args.extrinsics,allow_pickle=False) if args.extrinsics else np.eye(4),
            image.numpy(),predicted.detach().float().cpu().numpy(),known.numpy(),args.output_dir,args.viewer_test_seconds)


if __name__ == '__main__':
    main()
