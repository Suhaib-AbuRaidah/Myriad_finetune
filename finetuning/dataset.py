"""Rendered articulated scenes for MYRIAD fine-tuning.

Only numpy and torch are required. No tracking model or preprocessing files are
used: eight visible surface points per link are selected by image-space FPS,
back-projected with reference depth, attached to their rigid link, and moved by
the supplied link transforms. Schema-v1 cameras use +Z depth and a world-to-camera
extrinsic; this convention was checked against this dataset's visible points.

Example::

    from torch.utils.data import DataLoader
    from finetuning.dataset import ArticulatedDataset, collate_articulated
    val = ArticulatedDataset("data/Robotic_arm", split="val", max_scenes=32)
    loader = DataLoader(val, batch_size=2, collate_fn=collate_articulated)

``x`` is the square reference RGB image in [-1, 1]. Positions are normalized
to [0, 1] over the resized image's pixel centers, in (x, y) order. Five states
produce four displacement targets. Time is motion progress * time_scale
(default 4), NOT seconds or an assumed video frame rate.

The returned training tensors follow myriad_make_train_fns' naming. The future
fine-tuning trainer MUST additionally consume ``motion_token_mask`` to exclude
invalid/padded motion keys from attention, and ``flow_loss_mask`` for loss.
The original trainer does not implement the required padding-aware attention.
Evaluation should use ``trajectories``, ``position_valid`` and ``visibility``.
Occluded points retain geometric targets by default; points behind the camera
or outside the model image are invalid. Padding never duplicates surface points.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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


def project_link_points(
    local_points: np.ndarray,
    link_nodes: np.ndarray,
    transforms: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-image pixels [S,N,2], camera Z [S,N], and front validity."""
    local_h = np.column_stack((local_points, np.ones(len(local_points))))
    world = np.einsum("snij,nj->sni", transforms[:, link_nodes], local_h)
    camera = world @ extrinsic.T
    z = camera[..., 2]
    projected = camera[..., :3] @ intrinsic.T
    valid = np.isfinite(projected).all(-1) & np.isfinite(z) & (z > 1e-8)
    denominator = np.where(valid, projected[..., 2], 1.0)
    pixels = projected[..., :2] / denominator[..., None]
    pixels = np.where(valid[..., None], pixels, 0.0)
    return pixels, z, valid


def _visibility(
    pixels: np.ndarray,
    z: np.ndarray,
    front: np.ndarray,
    labels: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    segmentation: np.ndarray,
    atol: float,
    rtol: float,
) -> np.ndarray:
    states, h, w = depth.shape
    # Clip before conversion: invalid/very distant projections cannot overflow ints.
    px = np.rint(np.clip(pixels[..., 0], 0, w - 1)).astype(np.int64)
    py = np.rint(np.clip(pixels[..., 1], 0, h - 1)).astype(np.int64)
    inside = front & (pixels[..., 0] >= 0) & (pixels[..., 0] <= w - 1)
    inside &= (pixels[..., 1] >= 0) & (pixels[..., 1] <= h - 1)
    state = np.arange(states)[:, None]
    observed = depth[state, py, px]
    return (
        inside
        & depth_valid[state, py, px]
        & (segmentation[state, py, px] == labels[None])
        & (np.abs(observed - z) <= atol + rtol * np.abs(z))
    )


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
    """Load schema-v1 NPZ scenes directly, preserving existing dataset splits.

    ``max_scenes`` chooses a seeded, fixed subset without replacement. Validation
    FPS is keyed by scene filename + seed and is independent of worker/batch order.
    Training draws FPS seeds from torch's worker RNG and shuffles whole tracks;
    seed torch/the DataLoader generator to reproduce the training sampling stream.
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
    ):
        if points_per_link < 1 or image_size < 2 or num_transitions < 1:
            raise ValueError("points_per_link/num_transitions must be positive; image_size >= 2")
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if boundary_margin < 0 or depth_atol < 0 or depth_rtol < 0:
            raise ValueError("boundary_margin and depth tolerances must be nonnegative")
        if max_track_ids < 1 or (max_scenes is not None and max_scenes < 1):
            raise ValueError("max_track_ids and max_scenes must be positive")
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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        if self.training:
            rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, ()).item()))
        else:
            digest = hashlib.blake2b(f"{self.seed}:{path.name}".encode(), digest_size=8)
            rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        try:
            return self._load(path, rng)
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Invalid scene {path}: {error}") from error

    def _load(self, path: Path, rng: np.random.Generator) -> dict[str, Any]:
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
            rgb = archive["rgb"]
            depth = archive["depth"][:states].astype(np.float64)
            depth_valid = archive["depth_valid_mask"][:states].astype(bool)
            segmentation = archive["link_segmentation"][:states]
            links = archive["link_indices"].astype(np.int64)
            transforms = archive["link_transforms_world"][:states].astype(np.float64)
            intrinsic = archive["camera_intrinsic"].astype(np.float64)
            extrinsic = archive["camera_extrinsic"].astype(np.float64)
            fractions = archive["state_fractions"][:states].astype(np.float64)
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
        tracks = len(links) * self.points_per_link
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
        link_nodes = np.repeat(np.arange(len(links)), self.points_per_link)
        point_labels = np.repeat(links, self.points_per_link)
        for node, label in enumerate(links):
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
            slots = slice(node * self.points_per_link, node * self.points_per_link + count)
            local_points[slots], reference_pixels[slots], point_valid[slots] = local, pixels, True
        if not point_valid.any():
            raise ValueError("No visible valid-depth link points in the reference image")
        if self.training:
            order = rng.permutation(tracks)
            local_points, reference_pixels = local_points[order], reference_pixels[order]
            link_nodes, point_labels, point_valid = link_nodes[order], point_labels[order], point_valid[order]

        pixels, z, front = project_link_points(local_points, link_nodes, transforms, intrinsic, extrinsic)
        visible = _visibility(
            pixels, z, front, point_labels, depth, depth_valid, segmentation,
            self.depth_atol, self.depth_rtol,
        ) & point_valid[None]
        resized = (pixels - np.array([left, top]) + 0.5) * self.image_size / side - 0.5
        positions = resized / (self.image_size - 1)
        position_valid = front & point_valid[None] & np.isfinite(positions).all(-1)
        position_valid &= ((positions >= 0) & (positions <= 1)).all(-1)
        positions = np.where((front & point_valid[None])[..., None], positions, 0.0)
        image = _image_tensor(rgb[0], top, left, side, self.image_size)
        sample = {
            "x": image,
            "trajectories": torch.from_numpy(positions.astype(np.float32)),
            "position_valid": torch.from_numpy(position_valid),
            "visibility": torch.from_numpy(visible),
            "point_valid": torch.from_numpy(point_valid),
            "local_points": torch.from_numpy(local_points.astype(np.float32)),
            "reference_pixels": torch.from_numpy(reference_pixels.astype(np.float32)),
            "point_link_nodes": torch.from_numpy(link_nodes),
            "point_link_labels": torch.from_numpy(point_labels),
            "link_indices": torch.from_numpy(links),
            "link_valid": torch.ones(len(links), dtype=torch.bool),
            "state_fractions": torch.from_numpy(fractions.astype(np.float32)),
            "state_times": torch.from_numpy(((fractions - fractions[0]) * self.time_scale).astype(np.float32)),
            "camera_static": torch.tensor(True),
            "supervise_occluded": self.supervise_occluded,
            "num_tracks": tracks,
            "num_links": len(links),
            "sample_id": str(metadata.get("sample_id", path.stem)),
            "object_id": str(metadata.get("object", {}).get("object_id", "")),
            "scene_path": str(path),
        }
        if self.include_final_image:
            sample["final_x"] = _image_tensor(rgb[states - 1], top, left, side, self.image_size)
        sample.update(_training_targets(sample))
        if not sample["flow_loss_mask"].any():
            raise ValueError("Scene has no valid consecutive displacement targets")
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
    sequence_keys = {"trajectories", "position_valid", "visibility"}
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
