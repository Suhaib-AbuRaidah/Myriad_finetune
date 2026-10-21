"""Verify/fill the exact cropped-image feature cache for selected scene pairs."""
import argparse
import json
from pathlib import Path
import torch
from training_from_scratch_cropped.features import FeatureCache, fill_cache
from training_from_scratch_cropped.config import write_json
from .common import DEFAULT_OUT, DEFAULT_CHECKPOINT, prepare, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    manifest = prepare(args.output, args.checkpoint)
    cache = FeatureCache(manifest["feature_cache"], manifest["train_config"]["cache_namespace"])
    data = [make_dataset(manifest, split) for split in ("test1", "val")]
    def emit(row):
        write_json(args.output/"cache_status.json", row)
        print(json.dumps(row), flush=True)
    _, coverage = fill_cache(cache, [(dataset, range(len(dataset))) for dataset in data],
        checkpoint=manifest["frozen_encoder"]["path"], device=args.device, emit=emit)
    write_json(args.output/"cache_coverage.json", coverage)
    write_json(args.output/"image_inventory.json", cache.image_inventory)
    assert coverage["remaining"]==0


if __name__=="__main__": main()
