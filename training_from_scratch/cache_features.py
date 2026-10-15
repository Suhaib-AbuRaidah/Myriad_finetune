"""Precompute frozen DINO features for every eligible scene/pair image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training_from_scratch.dataset import ArticulatedDataset
from training_from_scratch.features import FeatureCache, encoder_identity, load_frozen_encoder
from training_from_scratch.model import build_myriad
from training_from_scratch.train import _resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/Robotic_arm"))
    parser.add_argument("--dino-checkpoint", type=Path, default=Path("myriad_openset.pt"))
    parser.add_argument("--feature-cache", type=Path, default=Path("training_from_scratch/feature_cache"))
    parser.add_argument("--scene-cache", type=Path, default=Path("training_from_scratch/cache"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--dino-precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test1", "test2"), default=["train", "val", "test2"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--overlay-alpha", type=float, default=.85)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args(argv)
    device = _resolve_device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    encoder = build_myriad(encoder_only=True)
    load_frozen_encoder(encoder, args.dino_checkpoint)
    encoder_dtype = torch.float16 if args.dino_precision == "fp16" else torch.float32
    encoder.to(device=device, dtype=encoder_dtype)
    cache = FeatureCache(args.feature_cache, identity=encoder_identity(args.dino_checkpoint),
                         image_size=encoder.image_size, dtype=encoder_dtype, feature_dtype=dtype)
    from myriad.model import make_axial_pos_2d
    for split in args.splits:
        # Enumeration includes all edges, even though training chooses one
        # random edge per scene visit. Points do not affect image embeddings.
        data = ArticulatedDataset(args.data_root, split=split, training=False,
            image_size=args.image_size, overlay_alpha=args.overlay_alpha,
            max_scenes=args.max_scenes, cache_dir=args.scene_cache)
        for i in range(len(data)):
            image = data[i]["x"].unsqueeze(0)
            if cache.read(image) is None:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                     enabled=encoder_dtype == torch.float16):
                    features = encoder(image.to(device)).to(dtype)
                    b, _, h, w = features.shape
                    pos = make_axial_pos_2d(h, w, device=device).reshape(1, h, w, 2).expand(b, -1, -1, -1)
                    cache.write(image, {"feat": features, "pos": pos, "L": h * w})
            if (i + 1) % 100 == 0 or i + 1 == len(data):
                print(json.dumps({"split": split, "pairs": i + 1, "total": len(data),
                                  "hits": cache.hits, "computed": cache.misses}), flush=True)


if __name__ == "__main__":
    main()
