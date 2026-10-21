"""All-pair FPS dataset. Originally valid targets remain supervised outside crops."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from finetuning_pair.conditioning import blend_roles
from finetuning_pair.geometry import TARGET_DEFINITION, relative_camera_points, project_camera_points
from .paths import artifact_path
from .geometry import mask_crop, resize_mask, image_tensor as _image_tensor
from training_from_scratch.dataset import farthest_point_indices, _interior, _backproject, _training_targets


class ArticulatedDataset(Dataset):
    """Every eligible pair, with randomized training FPS and fixed evaluation FPS."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        points_per_link: int = 8,
        image_size: int = 384,
        num_transitions: int = 4,
        time_scale: float = 4.0,
        boundary_margin: int = 1,
        training: bool | None = None,
        seed: int = 42,
        max_scenes: int | None = None,
        supervise_occluded: bool = True,
        depth_atol: float = 0.005,
        depth_rtol: float = 0.005,
        max_track_ids: int = 256,
        include_final_image: bool = False,
        overlay_alpha: float = 0.85,
        cache_dir: str | Path | None = None,
        crop_padding: float = .1,
    ):
        if points_per_link < 1 or image_size < 2 or not 1 <= num_transitions <= 4:
            raise ValueError("points_per_link must be positive; 1 <= num_transitions <= 4; image_size >= 2")
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if boundary_margin < 0 or depth_atol < 0 or depth_rtol < 0:
            raise ValueError("boundary_margin and depth tolerances must be nonnegative")
        if max_track_ids < 1 or (max_scenes is not None and max_scenes < 1):
            raise ValueError("max_track_ids and max_scenes must be positive")
        if not np.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be in [0,1]")
        if not supervise_occluded:
            raise ValueError("Relative occlusion needs counterfactual renders; supervise_occluded must be True")
        if points_per_link > max_track_ids:
            raise ValueError("Child point slots exceed max_track_ids")
        self.crop_padding = crop_padding
        if not np.isfinite(crop_padding) or crop_padding < 0:
            raise ValueError("crop_padding must be finite and nonnegative")
        self.overlay_alpha = overlay_alpha
        self.cache_dir = artifact_path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scene_dir = Path(root) / split / "scenes"
        self.files = sorted(self.scene_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ scenes found in {self.scene_dir}")
        if max_scenes is not None and max_scenes < len(self.files):
            subset_rng = np.random.default_rng(seed)
            indices = sorted(subset_rng.choice(len(self.files), max_scenes, replace=False))
            self.files = [self.files[i] for i in indices]
        self.points_per_link = points_per_link
        self.image_size = image_size
        self.num_transitions = num_transitions
        self.time_scale = time_scale
        self.boundary_margin = boundary_margin
        self.training = split == "train" if training is None else training
        self.seed = seed
        self.supervise_occluded = supervise_occluded
        self.depth_atol, self.depth_rtol = depth_atol, depth_rtol
        self.max_track_ids = max_track_ids
        self.include_final_image = include_final_image
        self.pairs = []
        self.examples = []
        for scene_index, path in enumerate(self.files):
            stat = path.stat()
            signature = json.dumps([TARGET_DEFINITION, str(path.resolve()), stat.st_size,
                                    stat.st_mtime_ns, self.image_size, self.crop_padding, "cropped_v1"])
            index_key = hashlib.sha256(signature.encode()).hexdigest()
            index_path = self.cache_dir / f"scene_pairs_{index_key}.json" if self.cache_dir is not None else None
            if index_path is not None and index_path.exists():
                pairs = [tuple(pair) for pair in json.loads(index_path.read_text())]
            else:
                pairs = self._scene_pairs(path)
                if index_path is not None:
                    fd, temporary = tempfile.mkstemp(prefix=".index-", suffix=".json", dir=self.cache_dir)
                    with os.fdopen(fd, "w") as handle:
                        json.dump(pairs, handle)
                    os.replace(temporary, index_path)
            self.pairs.append(pairs)
            self.examples.extend((scene_index, pair) for pair in pairs)
        if not self.examples:
            raise ValueError("No tree edges with initially visible valid-depth child points")
        self.training_indices = [i for i, pairs in enumerate(self.pairs) if pairs]

    def _scene_pairs(self, path):
        with np.load(path, allow_pickle=False) as archive:
            links = archive["link_indices"].astype(int)
            parents = archive["joint_parent_link_indices"].astype(int)
            children = archive["joint_child_link_indices"].astype(int)
            joints = archive["joint_indices"].astype(int)
            if (parents.ndim != 1 or parents.shape != children.shape or joints.shape != parents.shape
                    or len(set(joints.tolist())) != len(joints)
                    or len(set(children.tolist())) != len(children)
                    or len(children) != len(links) - 1
                    or not np.isin(parents, links).all() or not np.isin(children, links).all()
                    or np.any(parents == children)):
                raise ValueError(f"Invalid ground-truth tree in {path}")
            roots = set(links.tolist()) - set(children.tolist())
            reached = set(roots)
            if len(roots) != 1:
                raise ValueError(f"Tree must have exactly one root in {path}")
            while True:
                added = {int(c) for p, c in zip(parents, children) if int(p) in reached} - reached
                if not added:
                    break
                reached |= added
            if reached != set(links.tolist()):
                raise ValueError(f"Disconnected/cyclic tree in {path}")
            # Eligibility depends on initial evidence only, never future positions.
            seg = archive["link_segmentation"][0]
            dep = archive["depth"][0]
            ok = archive["depth_valid_mask"][0].astype(bool) & np.isfinite(dep) & (dep > 0)
            return [(int(parents[j]), int(children[j]), int(joints[j]))
                    for j in np.argsort(joints, kind="stable") if np.any(ok & (seg == children[j]))]

    def initial_image(self, index):
        scene_index, (parent, child, joint) = self.examples[index]
        path = self.files[scene_index]
        with np.load(path, allow_pickle=False) as a:
            rgb, seg = a["rgb"][0], a["link_segmentation"][0]
        pm, cm = seg == parent, seg == child
        crop = mask_crop(pm, cm, margin=self.crop_padding, image_size=self.image_size)
        image = _image_tensor(blend_roles(rgb, pm, cm, self.overlay_alpha),
                              crop.top, crop.left, crop.side, crop.image_size)
        return dict(x=image, crop=crop)

    def cache_key(self, path, pair):
        stat = path.stat()
        settings = dict(crop_padding=self.crop_padding, version="cropped_v1", target=TARGET_DEFINITION, source=str(path.resolve()), size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, pair=list(pair), seed=self.seed,
                        points=self.points_per_link, image_size=self.image_size,
                        transitions=self.num_transitions, time_scale=self.time_scale,
                        boundary_margin=self.boundary_margin, overlay_alpha=self.overlay_alpha,
                        depth_atol=self.depth_atol, depth_rtol=self.depth_rtol,
                        include_final_image=self.include_final_image)
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_index, pair = self.examples[index]
        if self.training:
            rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, ()).item()))
        else:
            digest = hashlib.blake2b(f"{self.seed}:{self.files[scene_index].name}:{pair}:{TARGET_DEFINITION}".encode(), digest_size=8)
            rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        path = self.files[scene_index]
        key = self.cache_key(path, pair)
        try:
            sample = self._load(path, rng, pair)
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Invalid scene {path}, pair {pair}: {error}") from error
        sample["cache_key"] = key
        return sample

    def _load(self, path: Path, rng: np.random.Generator, pair) -> dict[str, Any]:
        states = self.num_transitions + 1
        required = (
            "schema_version", "metadata_json", "rgb", "depth", "depth_valid_mask",
            "link_segmentation", "link_indices", "link_transforms_world",
            "camera_intrinsic", "camera_extrinsic", "state_fractions",
        )
        with np.load(path, allow_pickle=False) as archive:
            missing = set(required) - set(archive.files)
            if missing:
                raise ValueError(f"Missing arrays: {sorted(missing)}")
            if archive["schema_version"].item() != 1:
                raise ValueError("Only schema_version=1 is supported")
            metadata = json.loads(str(archive["metadata_json"].item()))
            available = len(archive["state_fractions"])
            if available != 5:
                raise ValueError("Schema-v1 pair targets require five states spanning initial to rest")
            state_indices = np.rint(np.linspace(0, available - 1, states)).astype(int)
            rgb = archive["rgb"][state_indices]
            depth = archive["depth"][state_indices].astype(np.float64)
            depth_valid = archive["depth_valid_mask"][state_indices].astype(bool)
            segmentation = archive["link_segmentation"][state_indices]
            links = archive["link_indices"].astype(np.int64)
            transforms = archive["link_transforms_world"][state_indices].astype(np.float64)
            intrinsic = archive["camera_intrinsic"].astype(np.float64)
            extrinsic = archive["camera_extrinsic"].astype(np.float64)
            fractions = archive["state_fractions"][state_indices].astype(np.float64)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 [states,H,W,3]")
        if rgb.shape[0] < states or depth.shape[0] != states:
            raise ValueError(f"Need {states} rendered states for {self.num_transitions} transitions")
        h, w = rgb.shape[1:3]
        if min(h, w) < 2 or depth.shape != (states, h, w):
            raise ValueError("depth must match RGB state/image dimensions")
        if segmentation.shape != depth.shape or depth_valid.shape != depth.shape:
            raise ValueError("depth mask and segmentation must match depth")
        if links.ndim != 1 or not len(links) or len(np.unique(links)) != len(links):
            raise ValueError("link_indices must be a nonempty, unique vector")
        if (links < -1).any() or not np.isin(segmentation, np.append(links, -2)).all():
            raise ValueError("Link labels must use -1 for base and -2 for background")
        tracks = self.points_per_link
        if tracks > self.max_track_ids:
            raise ValueError(f"{tracks} track slots exceed max_track_ids={self.max_track_ids}")
        if transforms.shape != (states, len(links), 4, 4):
            raise ValueError("link_transforms_world must be [states,num_links,4,4]")
        if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
            raise ValueError("Expected camera_intrinsic [3,3] and camera_extrinsic [4,4]")
        if not all(np.isfinite(a).all() for a in (transforms, intrinsic, extrinsic, fractions)):
            raise ValueError("Transforms, camera calibration and state fractions must be finite")
        if fractions.shape != (states,) or not (np.diff(fractions) > 0).all():
            raise ValueError("state_fractions must be strictly increasing")
        if not np.allclose(intrinsic[2], [0, 0, 1]) or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
            raise ValueError("Expected a pinhole intrinsic with positive focal lengths")
        for name, matrices in (("links", transforms), ("camera", extrinsic)):
            rotation = matrices[..., :3, :3]
            if not np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-5):
                raise ValueError(f"{name} transforms must be homogeneous")
            if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-4):
                raise ValueError(f"{name} transforms must be rigid")
            if not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                raise ValueError(f"{name} rotations must have determinant +1")
        convention = metadata.get("camera", {}).get("extrinsic_convention")
        if convention != "T_camera_world (world coordinates to camera coordinates)":
            raise ValueError(f"Unsupported or unspecified camera convention: {convention!r}")

        if not np.isclose(fractions[0], 0) or not np.isclose(fractions[-1], 1):
            raise ValueError("Targets must span initial state through zero-joint rest state")
        parent_label, child_label, joint_index = pair
        by_label = {int(label): node for node, label in enumerate(links)}
        parent_node, child_node = by_label[parent_label], by_label[child_label]

        parent_mask, child_mask = segmentation[0] == parent_label, segmentation[0] == child_label
        crop = mask_crop(parent_mask, child_mask, margin=self.crop_padding, image_size=self.image_size)
        top, left, side = crop.top, crop.left, crop.side
        reference_valid = depth_valid[0] & np.isfinite(depth[0]) & (depth[0] > 0)
        local_points = np.zeros((tracks, 3), dtype=np.float64)
        reference_pixels = np.zeros((tracks, 2), dtype=np.float64)
        point_valid = np.zeros(tracks, dtype=bool)
        link_nodes = np.full(tracks, 1, dtype=np.int64)  # pair node 1 = child
        point_labels = np.full(tracks, child_label, dtype=np.int64)
        for node, label in [(child_node, child_label)]:
            candidates = reference_valid & (segmentation[0] == label)
            interior = _interior(candidates, self.boundary_margin)
            if int(interior.sum()) >= self.points_per_link:
                candidates = interior
            y, x = np.nonzero(candidates)
            pixels = np.column_stack((x, y))
            selected = farthest_point_indices(pixels, self.points_per_link, rng)
            pixels = pixels[selected]
            count = len(selected)
            if not count:
                continue
            world = _backproject(pixels, depth[0, y[selected], x[selected]], intrinsic, extrinsic)
            world_h = np.column_stack((world, np.ones(count)))
            local = (world_h @ np.linalg.inv(transforms[0, node]).T)[:, :3]
            slots = slice(0, count)
            local_points[slots], reference_pixels[slots], point_valid[slots] = local, pixels, True
        if not point_valid.any():
            raise ValueError("No visible valid-depth link points in the reference image")
        if self.training:
            order = rng.permutation(tracks)
            local_points, reference_pixels = local_points[order], reference_pixels[order]
            link_nodes, point_labels, point_valid = link_nodes[order], point_labels[order], point_valid[order]

        relative_h, actual_h = relative_camera_points(
            local_points, transforms[:, parent_node], transforms[:, child_node], extrinsic)
        pixels, z, front = project_camera_points(relative_h, intrinsic)
        actual_pixels, _, actual_front = project_camera_points(actual_h, intrinsic)
        # Actual rendered masks cannot describe a counterfactual relative scene.
        # Only initial visibility is observed; future visibility stays unknown.
        visible = np.zeros((states, tracks), dtype=bool)
        visible[0] = point_valid
        visibility_known = np.zeros((states, tracks), dtype=bool)
        visibility_known[0] = point_valid
        resized = (pixels - np.array([left, top]) + 0.5) * self.image_size / side - 0.5
        positions = resized / (self.image_size - 1)
        position_valid = front & point_valid[None] & np.isfinite(positions).all(-1)
        source_position_valid = position_valid & ((pixels >= 0) & (pixels <= [w-1, h-1])).all(-1)
        crop_position_valid = position_valid & ((positions >= 0) & (positions <= 1)).all(-1)
        position_valid = source_position_valid
        positions = np.where((front & point_valid[None])[..., None], positions, 0.0)
        parent_mask, child_mask = segmentation[0] == parent_label, segmentation[0] == child_label
        conditioned_rgb = blend_roles(rgb[0], parent_mask, child_mask, self.overlay_alpha)
        image = _image_tensor(conditioned_rgb, top, left, side, self.image_size)
        actual_resized = (actual_pixels - [left, top] + 0.5) * self.image_size / side - 0.5
        actual_positions = actual_resized / (self.image_size - 1)
        actual_valid = actual_front & point_valid[None] & np.isfinite(actual_positions).all(-1)
        actual_crop_valid = actual_valid & ((actual_positions >= 0) & (actual_positions <= 1)).all(-1)
        actual_valid &= ((actual_pixels >= 0) & (actual_pixels <= [w-1, h-1])).all(-1)
        actual_positions = np.where((actual_front & point_valid[None])[..., None], actual_positions, 0.0)
        sample = {
            "crop_transform": torch.from_numpy(crop.matrix),
            "crop_inverse": torch.from_numpy(np.linalg.inv(crop.matrix)),
            "crop_bounds": torch.tensor([top, left, side, self.image_size]),
            "source_image_hw": torch.tensor([h, w]),
            "camera_intrinsic_source": torch.from_numpy(intrinsic),
            "camera_intrinsic_model": torch.from_numpy(crop.matrix @ intrinsic),
            "camera_extrinsic": torch.from_numpy(extrinsic),
            "source_trajectories": torch.from_numpy(pixels.astype(np.float32)),
            "source_actual_trajectories": torch.from_numpy(actual_pixels.astype(np.float32)),
            "reference_world": torch.from_numpy(((relative_h[0] @ np.linalg.inv(extrinsic).T)[:, :3]).astype(np.float32)),
            "source_position_valid": torch.from_numpy(source_position_valid),
            "crop_position_valid": torch.from_numpy(crop_position_valid),
            "actual_crop_position_valid": torch.from_numpy(actual_crop_valid),
            "reference_pixels_model": torch.from_numpy((crop.normalize(reference_pixels)*(self.image_size-1)).astype(np.float32)),
            "x": image,
            "original_x": _image_tensor(rgb[0], top, left, side, self.image_size),
            "parent_mask": resize_mask(parent_mask, crop),
            "child_mask": resize_mask(child_mask, crop),
            "pair_labels": torch.tensor([parent_label, child_label]),
            "joint_index": joint_index,
            "overlay_alpha": self.overlay_alpha,
            "target_definition": TARGET_DEFINITION,
            "visibility_known": torch.from_numpy(visibility_known),
            "relative_camera_points": torch.from_numpy(relative_h[..., :3].astype(np.float32)),
            "actual_trajectories": torch.from_numpy(actual_positions.astype(np.float32)),
            "actual_position_valid": torch.from_numpy(actual_valid),
            "trajectories": torch.from_numpy(positions.astype(np.float32)),
            "position_valid": torch.from_numpy(position_valid),
            "visibility": torch.from_numpy(visible),
            "point_valid": torch.from_numpy(point_valid),
            "local_points": torch.from_numpy(local_points.astype(np.float32)),
            "reference_pixels": torch.from_numpy(reference_pixels.astype(np.float32)),
            "point_link_nodes": torch.from_numpy(link_nodes),
            "point_link_labels": torch.from_numpy(point_labels),
            "link_indices": torch.tensor([parent_label, child_label]),
            "link_valid": torch.ones(2, dtype=torch.bool),
            "state_fractions": torch.from_numpy(fractions.astype(np.float32)),
            "state_times": torch.from_numpy(((fractions - fractions[0]) * self.time_scale).astype(np.float32)),
            "camera_static": torch.tensor(True),
            "supervise_occluded": self.supervise_occluded,
            "num_tracks": tracks,
            "num_links": 2,
            "sample_id": f"{metadata.get('sample_id', path.stem)}:parent={parent_label}:child={child_label}:joint={joint_index}",
            "object_id": str(metadata.get("object", {}).get("object_id", "")),
            "scene_path": str(path),
        }
        if self.include_final_image:
            # There is no rendered frozen-parent endpoint in the source dataset.
            # All relative visualization panels use the conditioned initial image.
            sample["final_x"] = image.clone()
        sample.update(_training_targets(sample))
        return sample



# Fixed-size child slots permit the existing collation and unchanged FM targets.
from training_from_scratch.dataset import collate_articulated
PairDataset = ArticulatedDataset
