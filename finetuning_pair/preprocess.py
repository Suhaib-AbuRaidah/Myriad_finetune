"""Populate versioned deterministic pair caches and verify/render target examples.

Run without model weights or a GPU. Verification independently applies the
world-frame frozen-parent formula, then E and K, and checks initial round trips.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from finetuning_pair.conditioning import blend_roles, crop_geometry, normalized_pixels
from finetuning_pair.dataset import ArticulatedDataset
from finetuning_pair.geometry import TARGET_DEFINITION
from finetuning_pair.paths import artifact_path
from finetuning_pair.visualizer import _reference_image, _draw_rollout


def verify_sample(sample):
    with np.load(sample['scene_path'], allow_pickle=False) as source:
        labels = source['link_indices'].tolist()
        p, c = [labels.index(int(label)) for label in sample['pair_labels']]
        states = sample['trajectories'].shape[0]
        selected = np.rint(np.linspace(0, len(source['state_fractions']) - 1, states)).astype(int)
        transforms = source['link_transforms_world'][selected].astype(np.float64)
        e, k = source['camera_extrinsic'].astype(np.float64), source['camera_intrinsic'].astype(np.float64)
        h, w = source['rgb'].shape[1:3]
        points = sample['local_points'].numpy().astype(np.float64)
        points = np.c_[points, np.ones(len(points))]
        expected = np.array([e @ transforms[0, p] @ np.linalg.inv(t[p]) @ t[c] @ points.T for t in transforms]).transpose(0, 2, 1)
        xyz = expected[..., :3] / expected[..., 3:4]
        projected = xyz @ k.T
        pixels = projected[..., :2] / projected[..., 2:3]
        uv = normalized_pixels(pixels, h, w, sample['x'].shape[-1])
        valid = sample['position_valid'].numpy()
        point_valid = sample['point_valid'].numpy()
        max_error = float(np.abs(uv[valid] - sample['trajectories'].numpy()[valid]).max())
        initial = normalized_pixels(sample['reference_pixels'].numpy()[point_valid], h, w, sample['x'].shape[-1])
        initial_error = float(np.abs(initial - sample['trajectories'][0].numpy()[point_valid]).max())
        parent_label, child_label = sample['pair_labels'].tolist()
        seg = source['link_segmentation'][0]
        rgb = source['rgb'][0]
        blended = blend_roles(rgb, seg == parent_label, seg == child_label, sample['overlay_alpha'])
        outside = (seg != parent_label) & (seg != child_label)
        unchanged = bool(np.array_equal(blended[outside], rgb[outside]))
        j = source['joint_indices'].tolist().index(sample['joint_index'])
        fixed = int(source['joint_types'][j]) == 4
        stationary = float(np.abs(sample['trajectories'].numpy()[:, point_valid] - sample['trajectories'][0].numpy()[point_valid]).max())
        difference = float(np.linalg.norm(sample['actual_trajectories'][-1].numpy()[point_valid] - uv[-1, point_valid], axis=-1).mean() * (sample['x'].shape[-1] - 1))
    if max_error > 2e-5 or initial_error > 2e-5 or not unchanged or (fixed and stationary > 2e-5):
        raise AssertionError(f'Relative target verification failed: {sample["sample_id"]}')
    return dict(sample_id=sample['sample_id'], scene_path=sample['scene_path'],
                pair_labels=sample['pair_labels'].tolist(), joint_index=sample['joint_index'],
                target_definition=TARGET_DEFINITION, overlay_alpha=sample['overlay_alpha'],
                max_formula_error_normalized=max_error, initial_roundtrip_error_normalized=initial_error,
                other_parts_background_unchanged=unchanged, fixed_joint=fixed,
                max_relative_displacement_normalized=stationary,
                endpoint_actual_relative_difference_pixels=difference,
                valid_points=int(point_valid.sum()), cache_key=sample['cache_key'])


def save_visuals(sample, output_dir):
    output = artifact_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original, reference = _reference_image(sample['original_x']), _reference_image(sample['x'])
    reference.save(output / 'model_input.png')
    pos, valid = sample['trajectories'].numpy(), sample['position_valid'].numpy()
    actual, actual_valid = sample['actual_trajectories'].numpy(), sample['actual_position_valid'].numpy()
    links = sample['point_link_nodes'].numpy()
    relative = _draw_rollout(reference, pos, valid, links, len(pos)-1, observed_states=1)
    actual_panel = _draw_rollout(reference, actual, actual_valid, links, len(pos)-1, observed_states=1)
    # Explicit initial-query rings remain visible even when trajectories move far.
    for panel in (relative, actual_panel):
        draw = ImageDraw.Draw(panel)
        for xy in pos[0, sample['point_valid'].numpy()]:
            x, y = xy * (reference.width - 1)
            draw.ellipse((x-4, y-4, x+4, y+4), fill='white', outline='black', width=1)
    relative.save(output / 'relative_trajectories.png')
    width, height = reference.size
    sheet = Image.new('RGB', (4*width, height+60), '#10151d')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 6), f'Parent {sample["pair_labels"][0]} BLUE | Child {sample["pair_labels"][1]} ORANGE | alpha {sample["overlay_alpha"]}', fill='white')
    for i, (label, panel) in enumerate(zip(('Original RGB', 'Exact model input', 'RELATIVE target: parent frozen', 'Actual target: diagnostic only'), (original, reference, relative, actual_panel))):
        sheet.paste(panel, (i*width, 60))
        draw.text((i*width+8, 34), label, fill='white')
    sheet.save(output / 'comparison.png')
    frames = []
    for t in range(len(pos)):
        panel = _draw_rollout(reference, pos, valid, links, t, observed_states=1)
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f'Relative target state {t} | frozen parent', fill='white', stroke_width=1, stroke_fill='black')
        frames.append(panel)
    frames[0].save(output / 'relative.gif', save_all=True, append_images=frames[1:], duration=800, loop=0)
    np.savez_compressed(output / 'targets.npz', trajectories=pos, position_valid=valid,
                        actual_trajectories=actual, actual_position_valid=actual_valid,
                        relative_camera_points=sample['relative_camera_points'].numpy(),
                        reference_pixels=sample['reference_pixels'].numpy(), point_valid=sample['point_valid'].numpy(),
                        pair_labels=sample['pair_labels'].numpy(), image=sample['x'].numpy(),
                        target_definition=np.array(TARGET_DEFINITION))
    report = verify_sample(sample)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/Robotic_arm'))
    parser.add_argument('--split', choices=('train', 'val', 'test1', 'test2'), default='train')
    parser.add_argument('--max-scenes', type=int, help='Optional seeded subset; default all scenes')
    parser.add_argument('--visualize-examples', type=int, default=6)
    parser.add_argument('--cache-dir', type=Path, default=Path('finetuning_pair/cache'))
    parser.add_argument('--output-dir', type=Path, default=Path('finetuning_pair/verification'))
    parser.add_argument('--overlay-alpha', type=float, default=0.85)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.visualize_examples < 0:
        parser.error('--visualize-examples must be nonnegative')
    output = artifact_path(args.output_dir)
    ds = ArticulatedDataset(args.data_root, split=args.split, training=False, seed=args.seed,
                            max_scenes=args.max_scenes, overlay_alpha=args.overlay_alpha, cache_dir=args.cache_dir)
    reports = []
    for i in range(len(ds)):
        sample = ds[i]
        report = save_visuals(sample, output / f'pair_{i:04d}') if i < args.visualize_examples else verify_sample(sample)
        reports.append(report)
        print(json.dumps(dict(example=i, pair=report['pair_labels'], formula_error=report['max_formula_error_normalized'])), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(target_definition=TARGET_DEFINITION, split=args.split, examples=len(ds),
                   max_formula_error_normalized=max(r['max_formula_error_normalized'] for r in reports),
                   all_other_pixels_unchanged=all(r['other_parts_background_unchanged'] for r in reports), examples_verified=reports)
    (output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
