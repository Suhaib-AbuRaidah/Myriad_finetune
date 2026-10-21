"""Save a scene's cropped-track montage, refit its joints, and open Open3D.

Run as a module from the repository root; see VISUALIZATION.md. No neural model
is loaded. The Open3D window is interactive and is never captured to an image.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / 'training_from_scratch_cropped/runs/full_test1_val80_20261021'
COLORS = ((230, 70, 35), (45, 110, 230), (185, 65, 195), (220, 145, 15), (0, 145, 140))
GT_COLOR = (20, 185, 100)


def load_scene(run, split, scene_index, sample_id, hypothesis):
    manifest = json.loads((run / 'run.json').read_text())
    scenes = manifest['splits'][split]['scenes']
    if sample_id is not None:
        matches = [s for s in scenes if s['sample_id'] == sample_id or Path(s['archive']).stem == sample_id]
        if len(matches) != 1:
            raise ValueError(f'Expected one archived scene for sample ID {sample_id!r}; found {len(matches)}')
        scene_index = matches[0]['scene_index']
    matches = [s for s in scenes if s['scene_index'] == scene_index]
    if len(matches) != 1:
        raise ValueError(f'No archived {split} scene {scene_index}; available local indices: 0..{len(scenes)-1}')
    scene = matches[0]
    records = [json.loads(p.read_text()) for p in sorted((run / split / 'records').glob('*.json'))]
    records = sorted((r for r in records if r['scene_index'] == scene_index), key=lambda r: r['joint_index'])
    expected = [p for s, p in manifest['splits'][split]['pairs'] if s == scene_index]
    if len(records) != len(expected) or not records:
        raise ValueError('This scene has missing prediction records or no eligible parent-child pairs')
    pairs = []
    keys = ('reference_world', 'predictions_source_pixels', 'predictions_model_pixels',
        'ground_truth_crop_normalized', 'valid', 'input_image', 'cropped_rgb',
        'camera_intrinsic', 'camera_extrinsic', 'pair_labels')
    for row in records:
        if row['status'] != 'predicted':
            raise ValueError(f"Pair {row['pair_index']} has no saved prediction")
        with np.load(run / split / row['prediction_file'], allow_pickle=False) as arrays:
            data = {key: arrays[key] for key in keys}
        if not 0 <= hypothesis < len(data['predictions_source_pixels']):
            raise ValueError(f"Hypothesis must be 0..{len(data['predictions_source_pixels'])-1}")
        if list(data['pair_labels']) != [row['parent_link'], row['child_link']]:
            raise ValueError('Pair labels disagree with the prediction record')
        pairs.append(dict(record=row, data=data, color=COLORS[row['joint_index'] % len(COLORS)]))
    source_keys = ('mesh_points', 'mesh_normals', 'mesh_point_link_labels', 'rgb', 'depth',
        'link_segmentation', 'camera_intrinsic', 'camera_extrinsic', 'link_indices',
        'joint_indices', 'joint_axes_world', 'joint_origins_world', 'link_transforms_world')
    with np.load(run / split / scene['archive'], allow_pickle=False) as arrays:
        # Retain only the initial meshes and RGB-D frames for visualization.
        initial = {'mesh_points', 'mesh_normals', 'mesh_point_link_labels', 'rgb', 'depth', 'link_segmentation'}
        source = {key: arrays[key][0].copy() if key in initial else arrays[key] for key in source_keys}
    return manifest, scene, pairs, source


def font(size):
    for name in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_track(draw, points, color, width, valid=None, dashed=False):
    finite = np.isfinite(points).all(-1)
    if valid is not None:
        finite &= valid
    for state in range(1, len(points)):
        if not (finite[state-1] and finite[state]):
            continue
        start, end = points[state-1], points[state]
        if dashed:
            distance = np.linalg.norm(end-start)
            count = max(1, int(math.ceil(distance/7)))
            for part in range(0, count, 2):
                a, b = part/count, min(part+1, count)/count
                draw.line([tuple(start+a*(end-start)), tuple(start+b*(end-start))], fill=color, width=width)
        else:
            draw.line([tuple(start), tuple(end)], fill=color, width=width)
    for point in points[finite]:
        x, y = point
        radius = max(2, width)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def save_montage(pairs, scene, manifest, args, target):
    count = len(pairs)
    columns = min(count, args.columns or math.ceil(math.sqrt(count)))
    rows = math.ceil(count/columns)
    size, gap, heading = args.tile_size, 16, 84
    canvas = Image.new('RGB', (columns*size+(columns+1)*gap, heading+rows*size+(rows+1)*gap), '#eef1f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), f"{args.split} scene {scene['scene_index']} | hypothesis {args.hypothesis} | step {manifest['checkpoint_step']}",
        font=font(23), fill='#182333')
    legend = 'Predictions: joint colors; white circles: observed anchors'
    if args.gt_trajectories:
        legend += '; green dashed: GT'
    draw.text((gap, 48), legend, font=font(14), fill='#425269')
    for index, pair in enumerate(pairs):
        record, a, color = pair['record'], pair['data'], pair['color']
        tile = Image.new('RGB', (size, size), 'white')
        td = ImageDraw.Draw(tile)
        td.rectangle((0, 0, size, 5), fill=color)
        td.text((12, 13), f"Joint {record['joint_ordinal']} | {record['gt_type']} | pair {record['pair_index']}",
            font=font(18), fill=color)
        td.text((12, 38), f"Parent {record['parent_link']} -> child {record['child_link']}", font=font(14), fill='#425269')
        rgb = a['cropped_rgb'].transpose(1, 2, 0) if args.image == 'raw' else np.clip(
            (a['input_image'].transpose(1, 2, 0)+1)*127.5, 0, 255).astype(np.uint8)
        body_size = size-78
        image = Image.fromarray(rgb).resize((body_size, body_size), Image.Resampling.LANCZOS)
        ink = ImageDraw.Draw(image)
        scale = body_size/rgb.shape[1]
        prediction = (a['predictions_model_pixels'][args.hypothesis]+.5)*scale-.5
        truth = (a['ground_truth_crop_normalized']*(rgb.shape[1]-1)+.5)*scale-.5
        for track in range(prediction.shape[1]):
            if args.gt_trajectories:
                draw_track(ink, truth[:, track], GT_COLOR, 2, a['valid'][:, track], dashed=True)
            draw_track(ink, prediction[:, track], color, 2)
            if np.isfinite(prediction[0, track]).all():
                x, y = prediction[0, track]
                ink.ellipse((x-4, y-4, x+4, y+4), fill='white', outline=color, width=2)
        tile.paste(image, ((size-body_size)//2, 65))
        canvas.paste(tile, (gap+(index % columns)*(size+gap), heading+gap+(index//columns)*(size+gap)))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    print(f'Cropped-pair montage: {target}', flush=True)


def fit_pairs(pairs, source, hypothesis):
    from .candidate import CONFIG, estimate_motion
    from testing_joint_fitting.axis_motion.common import score
    from training_from_scratch.full_evaluation.metrics import kinematic_reference

    points, normals, labels = (source[key] for key in ('mesh_points', 'mesh_normals', 'mesh_point_link_labels'))
    lookup = {int(label): index for index, label in enumerate(source['link_indices'])}
    results = []
    for pair in pairs:
        row, a = pair['record'], pair['data']
        result = dict(joint_index=row['joint_index'], joint_ordinal=row['joint_ordinal'],
            pair_index=row['pair_index'], gt_type=row['gt_type'], color=list(pair['color']),
            prediction_file=row['prediction_file'], anchor=a['reference_world'].mean(0), fit=None)
        if row['gt_type'] != 'revolute':
            result['status'] = 'not fitted: candidate assumes revolute joints'
        elif int(a['valid'][2:5].sum()) < 8:
            result['status'] = 'not fitted: fewer than eight valid observations in states 2..4'
        else:
            parent_mask, child_mask = (labels == row[key] for key in ('parent_link', 'child_link'))
            try:
                fitted = estimate_motion(a['reference_world'], a['predictions_source_pixels'][hypothesis],
                    a['valid'], a['camera_intrinsic'], a['camera_extrinsic'],
                    parent_points=points[parent_mask], child_points=points[child_mask],
                    parent_normals=normals[parent_mask], child_normals=normals[child_mask])
                result.update(fit=fitted, status='fitted' if fitted is not None else 'fitter returned no solution')
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                result['status'] = f'fit failed: {error}'
        # GT parameters and future transforms are accessed only after fitting.
        if row['gt_type'] == 'revolute':
            jp = int(np.flatnonzero(source['joint_indices'] == row['joint_index'])[0])
            parent, child = (lookup[row[key]] for key in ('parent_link', 'child_link'))
            reference = kinematic_reference(source, jp, parent, child)
            result.update(gt_axis=source['joint_axes_world'][0, jp],
                gt_pivot=reference.get('pivot'), kinematic_reference=reference)
            if result['fit'] is not None:
                t = source['link_transforms_world'].astype(float)
                rotations = [(t[0, parent] @ np.linalg.inv(t[s, parent]) @ t[s, child] @ np.linalg.inv(t[0, child]))[:3, :3]
                    for s in (2, 3, 4)]
                case = {key: row[key] for key in ('pair_index', 'scene_index', 'joint_ordinal', 'object_id')}
                case.update(gt_axis=result['gt_axis'],
                    gt_pivot=None if result['gt_pivot'] is None else np.asarray(result['gt_pivot']), gt_rotations=np.stack(rotations))
                result['metrics'] = score(case, result['fit']['fit'])
        metrics = result.get('metrics')
        message = f"; axis error {metrics['axis_error_deg']:.3f} deg; pivot distance {metrics['pivot_distance']:.5f}" if metrics and metrics.get('pivot_distance') is not None else ''
        print(f"Joint {row['joint_ordinal']} (pair {row['pair_index']}): {result['status']}{message}", flush=True)
        results.append(result)
    return results, CONFIG


def show_viewer(source, results, montage, scene, args):
    import open3d as o3d
    from open3d.visualization import gui, rendering
    from finetuning_pair.pointcloud_viewer import axis_segment, backproject_cloud

    if args.cloud == 'rgbd':
        xyz, colors = backproject_cloud(source['rgb'], source['depth'], source['camera_intrinsic'],
            source['camera_extrinsic'], np.isin(source['link_segmentation'], source['link_indices']))
    else:
        mask = np.isin(source['mesh_point_link_labels'], source['link_indices']) & np.isfinite(source['mesh_points']).all(-1)
        xyz, labels = source['mesh_points'][mask].astype(float), source['mesh_point_link_labels'][mask]
        colors = np.full(xyz.shape, .68)
        for label in np.unique(labels):
            color = np.array(COLORS[int(label) % len(COLORS)])/255
            colors[labels == label] = .55 + .40*color
    if not len(xyz):
        raise ValueError('The initial object point cloud is empty')
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    app = gui.Application.instance
    app.initialize()
    # A dedicated image window keeps the montage readable at presentation size.
    # Both windows share Open3D's event loop; neither captures the 3D viewport.
    with Image.open(montage) as saved_image:
        montage_array = np.asarray(saved_image.convert('RGB')).copy()
    montage_window = app.create_window(
        f"MYRIAD | {args.split} scene {scene['scene_index']} | cropped pair trajectories", 920, 900)
    montage_widget = gui.ImageWidget(o3d.geometry.Image(montage_array))
    montage_window.add_child(montage_widget)

    def layout_montage(context):
        rect = montage_window.content_rect
        height, width = montage_array.shape[:2]
        factor = min(rect.width/width, rect.height/height)
        shown_width, shown_height = int(width*factor), int(height*factor)
        montage_widget.frame = gui.Rect(rect.x+(rect.width-shown_width)//2,
            rect.y+(rect.height-shown_height)//2, shown_width, shown_height)

    montage_window.set_on_layout(layout_montage)
    window = app.create_window(f"MYRIAD | {args.split} scene {scene['scene_index']} | joint axes", 1440, 940)
    view = gui.SceneWidget()
    view.scene = rendering.Open3DScene(window.renderer)
    view.scene.set_background([.96, .97, .99, 1.])
    point_material = rendering.MaterialRecord()
    point_material.shader, point_material.point_size = 'defaultUnlit', args.point_size * window.scaling
    view.scene.add_geometry('Object point cloud', cloud, point_material)
    gt_names = []
    joint_names = {}
    bounds = cloud.get_axis_aligned_bounding_box()

    def material(color, opacity=1., *, transparent=False):
        mat = rendering.MaterialRecord()
        # Open3D 0.19's transparent unlit shader lacks srgbColor, while its
        # unlit updater unconditionally sets it, aborting inside Filament.
        # Lit transparency uses a separate updater and preserves real alpha.
        mat.shader = 'defaultLitTransparency' if transparent else 'defaultLit'
        mat.base_color = [*(np.asarray(color)/255).tolist(), float(opacity)]
        mat.base_roughness = .65
        mat.base_metallic = 0.
        return mat

    def add_axis(name, axis, pivot, anchor, color, opacity, radius, *, transparent=False):
        nonlocal bounds
        length = scale*args.axis_length
        endpoints, center = axis_segment(axis, pivot, anchor, length)
        direction = (endpoints[1]-endpoints[0])/length
        # A cylinder is undirected: flipping axis sign changes neither line nor mesh.
        cross = np.cross([0., 0., 1.], direction)
        sine, cosine = np.linalg.norm(cross), float(direction[2])
        if sine < 1e-12:
            rotation = np.eye(3) if cosine > 0 else np.diag([1., -1., -1.])
        else:
            v = cross/sine
            skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation = np.eye(3)+sine*skew+(1-cosine)*(skew@skew)
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius*scale, height=length, resolution=32)
        mesh.rotate(rotation, center=(0, 0, 0)).translate(center)
        mesh.compute_vertex_normals()
        view.scene.add_geometry(name, mesh, material(color, opacity, transparent=transparent))
        bounds += mesh.get_axis_aligned_bounding_box()

    for result in results:
        names = []
        ordinal, fitted = result['joint_ordinal'], result['fit']
        if fitted is not None:
            name = f'Predicted joint {ordinal}'
            add_axis(name, fitted['axis'], fitted['pivot'], result['anchor'], result['color'], 1., args.axis_radius)
            names.append(name)
        if result.get('gt_pivot') is not None:
            name = f'GT joint {ordinal}'
            add_axis(name, result['gt_axis'], result['gt_pivot'], result['anchor'], GT_COLOR,
                args.gt_opacity, args.axis_radius*1.6, transparent=True)
            view.scene.show_geometry(name, args.show_gt)
            gt_names.append(name)
            names.append(name)
        joint_names[ordinal] = names

    em = window.theme.font_size
    panel = gui.ScrollableVert(.45*em, gui.Margins(em, em, em, em))
    panel.add_child(gui.Label(f"Scene {scene['scene_index']} | hypothesis {args.hypothesis}"))
    panel.add_child(gui.Label('Solid axes: joint colors\nTransparent green: ground truth'))
    pred_toggle, gt_toggle = gui.Checkbox('Predicted axes'), gui.Checkbox('Ground-truth axes')
    pred_toggle.checked, gt_toggle.checked = True, args.show_gt
    enabled = {r['joint_ordinal']: True for r in results}

    def update_visibility(*_):
        for ordinal, names in joint_names.items():
            for name in names:
                view.scene.show_geometry(name, enabled[ordinal] and (gt_toggle.checked if name in gt_names else pred_toggle.checked))
        view.force_redraw()

    pred_toggle.set_on_checked(update_visibility)
    gt_toggle.set_on_checked(update_visibility)
    panel.add_child(pred_toggle)
    panel.add_child(gt_toggle)
    panel.add_child(gui.Label('Ground-truth opacity'))
    opacity = gui.Slider(gui.Slider.DOUBLE)
    opacity.set_limits(.02, 1.)
    opacity.double_value = args.gt_opacity

    def update_opacity(value):
        for name in gt_names:
            # Keep the same transparent shader even at opacity 1.0.
            view.scene.modify_geometry_material(name, material(GT_COLOR, value, transparent=True))
        view.force_redraw()

    opacity.set_on_value_changed(update_opacity)
    panel.add_child(opacity)
    for result in results:
        ordinal = result['joint_ordinal']
        checkbox = gui.Checkbox(f'Joint {ordinal}')
        checkbox.checked = True

        def toggle(checked, ordinal=ordinal):
            enabled[ordinal] = checked
            update_visibility()

        checkbox.set_on_checked(toggle)
        panel.add_child(checkbox)
        metrics = result.get('metrics')
        text = f"Axis error: {metrics['axis_error_deg']:.3f} deg" if metrics else result['status']
        label = gui.Label(text)
        label.text_color = gui.Color(*(np.array(result['color'])/255))
        panel.add_child(label)
    view.setup_camera(55., bounds, bounds.get_center())

    def reset_camera():
        world_camera = np.linalg.inv(source['camera_extrinsic'].astype(float))
        center = bounds.get_center()
        direction = world_camera[:3, 3]-center
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        distance = max(float(np.linalg.norm(bounds.get_extent())), scale)*1.5
        view.scene.camera.look_at(center, center+distance*direction, -world_camera[:3, 1])
        view.force_redraw()

    reset = gui.Button('Reset to source-camera view')
    reset.set_on_clicked(reset_camera)
    panel.add_child(reset)
    preview = Image.open(montage).convert('RGB')
    preview.thumbnail((350, 350), Image.Resampling.LANCZOS)
    panel.add_child(gui.ImageWidget(o3d.geometry.Image(np.asarray(preview).copy())))
    panel.add_child(gui.Label('Drag to rotate; wheel to zoom.\nMontage also shown in a separate window.\nClose both windows to exit.\nNo 3D screenshot is saved.'))
    window.add_child(view)
    window.add_child(panel)

    def layout(context):
        rect = window.content_rect
        width = int(min(390*window.scaling, rect.width*.4))
        view.frame = gui.Rect(rect.x, rect.y, rect.width-width, rect.height)
        panel.frame = gui.Rect(rect.get_right()-width, rect.y, width, rect.height)

    window.set_on_layout(layout)
    reset_camera()
    app.run()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--split', choices=('test1', 'val'), default='test1')
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--scene-index', type=int, default=0, help='Local scene index in the saved inference archive')
    select.add_argument('--sample-id', help='Original scene filename stem instead of a local index')
    parser.add_argument('--hypothesis', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show-gt', action='store_true', help='Initially show transparent GT axis meshes; also toggleable in the viewer')
    parser.add_argument('--gt-opacity', type=float, default=.28)
    parser.add_argument('--gt-trajectories', action='store_true', help='Add green dashed GT tracks to the saved montage')
    parser.add_argument('--image', choices=('model-input', 'raw'), default='model-input')
    parser.add_argument('--cloud', choices=('mesh', 'rgbd'), default='mesh', help='Initial sampled mesh cloud or visible RGB-D surface')
    parser.add_argument('--tile-size', type=int, default=512, help='Each pair occupies a square tile of this many pixels')
    parser.add_argument('--columns', type=int, default=0, help='0 chooses a compact grid automatically')
    parser.add_argument('--axis-length', type=float, default=.65, help='Axis segment length as a fraction of object bounding-box diagonal')
    parser.add_argument('--axis-radius', type=float, default=.006, help='Predicted cylinder radius as a fraction of object diagonal')
    parser.add_argument('--point-size', type=float, default=4.)
    args = parser.parse_args()
    if args.tile_size < 256 or args.columns < 0 or args.scene_index < 0:
        parser.error('tile-size must be >=256; columns and scene-index must be nonnegative')
    if not .02 <= args.gt_opacity <= 1.:
        parser.error('gt-opacity must be in [0.02, 1]')
    if not all(math.isfinite(v) and v > 0 for v in (args.axis_length, args.axis_radius, args.point_size)):
        parser.error('axis-length, axis-radius and point-size must be finite and positive')
    return args


def main():
    args = parse_args()
    run = args.run_dir.expanduser().resolve()
    manifest, scene, pairs, source = load_scene(run, args.split, args.scene_index, args.sample_id, args.hypothesis)
    output = (args.output_dir or run / 'presentation_visualizations' /
        f"{args.split}_scene_{scene['scene_index']:03d}_hyp_{args.hypothesis}").expanduser().resolve()
    montage = output / 'cropped_pairs.png'
    save_montage(pairs, scene, manifest, args, montage)
    excluded = [r for r in manifest['splits'][args.split]['inventory'] if r['scene_index']==scene['scene_index'] and not r['eligible']]
    if excluded:
        print('No saved crop for initially unobservable joints: '+', '.join(str(r['joint_ordinal']) for r in excluded), flush=True)
    results, config = fit_pairs(pairs, source, args.hypothesis)
    from testing_joint_fitting.experiment import write
    write(output / 'joint_parameters.json', dict(scene=scene, split=args.split, run_dir=str(run),
        hypothesis=args.hypothesis, checkpoint_step=manifest['checkpoint_step'],
        checkpoint_sha256=manifest['checkpoint_sha256'], candidate_config=config,
        fitting_states=[2,3,4], coordinate_frame='world', results=results,
        unavailable_crops=excluded, arguments=vars(args) | {'run_dir': str(run), 'output_dir': str(output)}))
    print('Opening the montage and Open3D windows. Close both to exit; no geometry image will be saved.', flush=True)
    show_viewer(source, results, montage, scene, args)


if __name__ == '__main__':
    main()
