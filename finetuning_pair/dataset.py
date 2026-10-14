"""One ground-truth parent/child edge per example, child-only relative targets.

Training samples an eligible edge at random on each scene visit. Evaluation
indexes every eligible edge in scene/joint order. Points use the original unique
visible-depth FPS, fixed slots, and false padding masks. Visibility after state
zero is unknown without rendering the counterfactual frozen-parent scene:
position_valid means geometric in-frame validity, never actual-scene occlusion.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from finetuning_pair.conditioning import blend_roles, crop_geometry, model_mask
from finetuning_pair.geometry import TARGET_DEFINITION, relative_camera_points, project_camera_points
from finetuning_pair.paths import artifact_path


def farthest_point_indices(
    pixels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Select up to count unique candidates, starting with a random candidate."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("FPS candidates must be finite [N, 2] image coordinates")
    if count < 1:
        raise ValueError("count must be positive")
    count = min(count, len(pixels))
    selected = np.empty(count, dtype=np.int64)
    if count == 0:
        return selected
    distances = np.full(len(pixels), np.inf)
    next_index = int(rng.integers(len(pixels)))
    for i in range(count):
        selected[i] = next_index
        distances = np.minimum(distances, np.square(pixels - pixels[next_index]).sum(-1))
        distances[selected[: i + 1]] = -np.inf
        next_index = int(distances.argmax())
    return selected


def _interior(mask: np.ndarray, margin: int) -> np.ndarray:
    """Erode with a 3x3 neighborhood without an OpenCV dependency."""
    for _ in range(margin):
        padded = np.pad(mask, 1, constant_values=False)
        h, w = mask.shape
        mask = np.logical_and.reduce(
            [padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]
        )
    return mask


def _backproject(
    pixels: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    pixel_h = np.column_stack((pixels, np.ones(len(pixels))))
    rays = pixel_h @ np.linalg.inv(intrinsic).T
    camera = rays * (depth / rays[:, 2])[:, None]
    camera_h = np.column_stack((camera, np.ones(len(camera))))
    return (camera_h @ np.linalg.inv(extrinsic).T)[:, :3]


def _training_targets(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    positions = sample["trajectories"]
    states, tracks, _ = positions.shape
    source = positions[:-1].reshape(-1, 2)
    flow = (positions[1:] - positions[:-1]).reshape(-1, 2)
    origins = positions[0].unsqueeze(0).expand(states - 1, -1, -1).reshape(-1, 2)
    times = sample["state_times"][:-1, None].expand(-1, tracks).reshape(-1)
    ids = torch.arange(tracks).expand(states - 1, -1).reshape(-1)
    valid = sample["position_valid"][:-1] & sample["position_valid"][1:]
    if not sample["supervise_occluded"]:
        valid &= sample["visibility"][:-1] & sample["visibility"][1:]
    mask = valid.reshape(-1)
    # Invalid displacements are not fed as observations, even if a trainer forgets
    # to zero them. Attention and loss still require the explicit masks above.
    flow = torch.where(mask[:, None], flow, torch.zeros_like(flow))
    result = {"flow_loss_mask": mask, "motion_token_mask": mask}
    for suffix in ("poke", "query"):
        result.update({
            f"pos_{suffix}": source,
            f"flow_{suffix}": flow,
            f"pos_orig_{suffix}": origins,
            f"t_{suffix}": times,
            f"id_{suffix}": ids,
        })
    return result


def _image_tensor(rgb: np.ndarray, top: int, left: int, side: int, size: int) -> torch.Tensor:
    cropped = np.ascontiguousarray(rgb[top : top + side, left : left + side])
    image = torch.from_numpy(cropped).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return F.interpolate(
        image, size=(size, size), mode="bilinear", align_corners=False, antialias=True,
    )[0].mul(2).sub(1).clamp(-1, 1)


class ArticulatedDataset(Dataset):
    """Child-only FPS and parent-frozen targets, preserving dataset splits.

    Training length counts scenes with eligible children. Evaluation length
    counts eligible pairs and enumerates them by scene filename/joint index.
    FPS seeds include the pair, scene and seed. Training never caches its random
    points. Every example has points_per_link child slots with false padding.
    """

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
                                    stat.st_mtime_ns, self.image_size])
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
            h, w = seg.shape
            top, left, side = crop_geometry(h, w)
            yy, xx = np.indices((h, w))
            ux = (xx - left + 0.5) * self.image_size / side - 0.5
            uy = (yy - top + 0.5) * self.image_size / side - 0.5
            ok &= (ux >= 0) & (ux <= self.image_size - 1) & (uy >= 0) & (uy <= self.image_size - 1)
            return [(int(parents[j]), int(children[j]), int(joints[j]))
                    for j in np.argsort(joints, kind="stable") if np.any(ok & (seg == children[j]))]

    def cache_key(self, path, pair):
        stat = path.stat()
        settings = dict(target=TARGET_DEFINITION, source=str(path.resolve()), size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, pair=list(pair), seed=self.seed,
                        points=self.points_per_link, image_size=self.image_size,
                        transitions=self.num_transitions, time_scale=self.time_scale,
                        boundary_margin=self.boundary_margin, overlay_alpha=self.overlay_alpha,
                        depth_atol=self.depth_atol, depth_rtol=self.depth_rtol,
                        include_final_image=self.include_final_image)
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.training_indices) if self.training else len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            scene_index = self.training_indices[index]
            rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, ()).item()))
            choices = self.pairs[scene_index]
            pair = choices[int(rng.integers(len(choices)))]
        else:
            scene_index, pair = self.examples[index]
            digest = hashlib.blake2b(f"{self.seed}:{self.files[scene_index].name}:{pair}:{TARGET_DEFINITION}".encode(), digest_size=8)
            rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        path = self.files[scene_index]
        key = self.cache_key(path, pair)
        cache = self.cache_dir / f"{key}.pt" if self.cache_dir is not None and not self.training else None
        if cache is not None and cache.exists():
            sample = torch.load(cache, map_location="cpu", weights_only=True)
            if sample.get("cache_key") != key or sample.get("target_definition") != TARGET_DEFINITION:
                raise ValueError(f"Incompatible relative cache: {cache}")
            return sample
        try:
            sample = self._load(path, rng, pair)
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Invalid scene {path}, pair {pair}: {error}") from error
        sample["cache_key"] = key
        if cache is not None:
            fd, temporary = tempfile.mkstemp(prefix=".pair-", suffix=".pt", dir=self.cache_dir)
            os.close(fd)
            try:
                torch.save(sample, temporary)
                os.replace(temporary, cache)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
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

        # Match DINO's square center crop, then resize pixel centers consistently
        # with interpolate(align_corners=False). RGB and trajectories share it.
        side = min(h, w)
        top, left = round((h - side) / 2), round((w - side) / 2)
        yy, xx = np.indices((h, w))
        resized_x = (xx - left + 0.5) * self.image_size / side - 0.5
        resized_y = (yy - top + 0.5) * self.image_size / side - 0.5
        crop_valid = (resized_x >= 0) & (resized_x <= self.image_size - 1)
        crop_valid &= (resized_y >= 0) & (resized_y <= self.image_size - 1)
        reference_valid = depth_valid[0] & np.isfinite(depth[0]) & (depth[0] > 0) & crop_valid
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
        position_valid &= ((positions >= 0) & (positions <= 1)).all(-1)
        positions = np.where((front & point_valid[None])[..., None], positions, 0.0)
        parent_mask, child_mask = segmentation[0] == parent_label, segmentation[0] == child_label
        conditioned_rgb = blend_roles(rgb[0], parent_mask, child_mask, self.overlay_alpha)
        image = _image_tensor(conditioned_rgb, top, left, side, self.image_size)
        actual_resized = (actual_pixels - [left, top] + 0.5) * self.image_size / side - 0.5
        actual_positions = actual_resized / (self.image_size - 1)
        actual_valid = actual_front & point_valid[None] & np.isfinite(actual_positions).all(-1)
        actual_valid &= ((actual_positions >= 0) & (actual_positions <= 1)).all(-1)
        actual_positions = np.where((actual_front & point_valid[None])[..., None], actual_positions, 0.0)
        sample = {
            "x": image,
            "original_x": _image_tensor(rgb[0], top, left, side, self.image_size),
            "parent_mask": model_mask(parent_mask, self.image_size),
            "child_mask": model_mask(child_mask, self.image_size),
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


def _pad(tensor: torch.Tensor, size: int, axis: int, fill: int | float = 0) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[axis] = size - shape[axis]
    return torch.cat((tensor, tensor.new_full(shape, fill)), dim=axis)


def collate_articulated(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad point/link slots, then rebuild time-major MYRIAD tensors.

    Padding a flattened sequence at its end would mix timestep boundaries when
    scenes have different link counts. We pad each state's point dimension first.
    Padded IDs are safe indices; their attention/loss masks are always false.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    states = samples[0]["trajectories"].shape[0]
    if any(s["trajectories"].shape[0] != states for s in samples):
        raise ValueError("All batch scenes must have the same number of rendered states")
    if any(s["x"].shape != samples[0]["x"].shape for s in samples):
        raise ValueError("All batch images must have the same shape")
    tracks = max(s["num_tracks"] for s in samples)
    links = max(s["num_links"] for s in samples)
    sequence_keys = {"trajectories", "position_valid", "visibility", "visibility_known", "relative_camera_points", "actual_trajectories", "actual_position_valid"}
    point_keys = {"point_valid", "local_points", "reference_pixels", "point_link_nodes", "point_link_labels"}
    link_keys = {"link_indices", "link_valid"}
    target_keys = set(_training_targets(samples[0]))
    padded = []
    for sample in samples:
        row = {}
        for key, value in sample.items():
            if key in target_keys:
                continue
            if key in sequence_keys:
                value = _pad(value, tracks, axis=1)
            elif key in point_keys:
                fill = -1 if key == "point_link_nodes" else (-2 if key == "point_link_labels" else 0)
                value = _pad(value, tracks, axis=0, fill=fill)
            elif key in link_keys:
                value = _pad(value, links, axis=0, fill=-2 if key == "link_indices" else 0)
            row[key] = value
        row.update(_training_targets(row))
        padded.append(row)
    batch = {}
    for key in padded[0]:
        values = [row[key] for row in padded]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        elif isinstance(values[0], (bool, int, float)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values
    return batch


PairDataset = ArticulatedDataset
collate_pairs = collate_articulated
