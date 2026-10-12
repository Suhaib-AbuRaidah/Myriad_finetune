"""Sample and visualize articulated MYRIAD trajectories from a saved adapter.

Use a dataset scene for scored inference, or provide an image plus normalized
reference-frame points for a new scene. Coordinates refer to the center-cropped
square image after resizing to the checkpoint's image size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from finetuning.checkpoint_io import inspect_checkpoint, load_finetuned_weights, load_pretrained_weights
from finetuning.dataset import ArticulatedDataset
from finetuning.model_adapter import FineTuningModel, LoRAConfig
from finetuning.testing import _scene_inputs, sample_rollout
from finetuning.train import build_myriad, _resolve_device
from finetuning.validation import paper_style_mse
from finetuning.visualizer import render_rollout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or latest.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning/runs/inference"))
    parser.add_argument("--device", default="auto")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test1", "test2"), default="test1")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--points", type=Path, help="Raw input: .npy [points,2] normalized reference positions")
    parser.add_argument("--final-image", type=Path, help="Raw input: optional rendered final-state image for the overlay")
    parser.add_argument("--poke-points", type=Path, help="Raw input: .npy [points,2] positions at state 1")
    parser.add_argument("--link-ids", type=Path, help="Raw input: optional .npy [points] link identifiers")
    parser.add_argument("--state-times", default="0,1,2,3,4", help="Comma-separated motion times for raw input")
    parser.add_argument("--poke-mode", choices=("with", "without"), default="without")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("auto", "none", "bf16"), default="auto")
    args = parser.parse_args(argv)
    if args.ensemble_size < 1 or args.diffusion_steps < 1 or args.seed < 0 or args.scene_index < 0:
        parser.error("ensemble size, diffusion steps and seed/scene index must be valid nonnegative values")
    if args.image and not args.points:
        parser.error("--points is required with --image")
    if args.image and args.poke_mode == "with" and not args.poke_points:
        parser.error("--poke-points is required for raw with-poke inference")
    if args.data_root and (args.points or args.poke_points or args.link_ids or args.final_image):
        parser.error("raw point files may only be used with --image")
    return args


def _raw_image(path: Path, image_size: int) -> torch.Tensor:
    source = Image.open(path).convert("RGB")
    width, height = source.size
    edge = min(width, height)
    source = source.crop(((width-edge)//2, (height-edge)//2,
                          (width-edge)//2+edge, (height-edge)//2+edge))
    source = source.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(source, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1


def _raw_inputs(args: argparse.Namespace, image_size: int):
    image = _raw_image(args.image, image_size)
    positions = np.load(args.points, allow_pickle=False).astype(np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
        raise ValueError("--points must contain [N,2] positions")
    known = [positions]
    if args.poke_mode == "with":
        known.append(np.load(args.poke_points, allow_pickle=False).astype(np.float32))
    if any(array.shape != positions.shape or not np.isfinite(array).all()
           or (array < 0).any() or (array > 1).any() for array in known):
        raise ValueError("Known positions must be finite, in [0,1], and have equal [N,2] shapes")
    links = np.zeros(len(positions), dtype=np.int64) if args.link_ids is None else np.load(args.link_ids, allow_pickle=False)
    if links.shape != (len(positions),) or not np.issubdtype(links.dtype, np.integer):
        raise ValueError("--link-ids must contain one integer per point")
    times = np.asarray([float(item) for item in args.state_times.split(",")], dtype=np.float32)
    if times.ndim != 1 or len(times) <= len(known) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("--state-times must be finite, increasing and include a future state")
    final_image = _raw_image(args.final_image, image_size) if args.final_image else None
    return image, torch.from_numpy(np.stack(known)), torch.from_numpy(times), torch.from_numpy(links.astype(np.int64)), final_image


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    info = inspect_checkpoint(args.checkpoint)
    config = info["train_config"]
    lora = LoRAConfig(**info["lora_config"])
    device = _resolve_device(args.device)
    base_path = Path(info["base_checkpoint"]["path"])
    bare = build_myriad()
    reference = load_pretrained_weights(bare, base_path)
    model = FineTuningModel(bare, lora).to(device)
    step = load_finetuned_weights(args.checkpoint, model, base_checkpoint=reference)
    model.eval()
    if args.data_root:
        dataset = ArticulatedDataset(
            args.data_root, split=args.split, points_per_link=config["points_per_link"],
            image_size=config["image_size"], num_transitions=config["num_transitions"],
            time_scale=config["time_scale"], seed=args.seed, training=False,
            include_final_image=True,
        )
        if args.scene_index >= len(dataset):
            raise IndexError(f"Scene index {args.scene_index} exceeds {len(dataset)} scenes")
        sample = dataset[args.scene_index]
        inputs = _scene_inputs(sample, args.poke_mode == "with", False)
        if inputs is None:
            raise ValueError("Selected scene has no conditionable future trajectory points")
        known, truth, valid, links = inputs
        image, times, final_image = sample["x"], sample["state_times"], sample["final_x"]
    else:
        image, known, times, links, final_image = _raw_inputs(args, config["image_size"])
        truth = valid = None
    predicted = sample_rollout(
        model, image, known, times, ensemble_size=args.ensemble_size,
        diffusion_steps=args.diffusion_steps, seed=args.seed, amp_dtype=args.amp_dtype,
    )
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": step,
              "poke_mode": args.poke_mode, "ensemble_size": args.ensemble_size,
              "diffusion_steps": args.diffusion_steps, "points": int(known.shape[1])}
    best_index = None
    if truth is not None:
        start = len(known)
        errors = paper_style_mse(predicted[:, start:], truth[start:].to(device), valid[start:].to(device))
        count = min(5, args.ensemble_size)
        best_index = int(errors[:count].argmin().item())
        result.update({"minADE_1": float(errors[0].item()),
                       f"minADE_{count}": float(errors[:count].min().item()),
                       "minADE_N": float(errors.min().item()),
                       "best_index": best_index,
                       "hypothesis_errors": [float(value) for value in errors.tolist()]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz",
                        predictions=predicted.detach().cpu().numpy(),
                        known_positions=known.numpy(), state_times=times.numpy(),
                        link_nodes=links.numpy(),
                        **({"ground_truth": truth.numpy(), "valid": valid.numpy()} if truth is not None else {}))
    result["artifacts"] = render_rollout(
        image, predicted, links, args.output_dir, observed_states=len(known),
        ground_truth=truth, ground_truth_valid=valid, final_image=final_image,
        best_index=best_index,
        title=f"MYRIAD inference | {args.poke_mode} poke", metrics=result,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
