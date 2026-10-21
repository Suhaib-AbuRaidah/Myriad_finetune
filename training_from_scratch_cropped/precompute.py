"""Resumable cropped-feature precomputation before allocating motion/AdamW."""
import json
import torch
from .config import parse_args, prepare, write_json
from .features import FeatureCache, fill_cache


def main(argv=None):
    args, saved = parse_args(argv, mode="precompute")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    emit = lambda r: print(json.dumps(r, sort_keys=True), flush=True)
    train, val, manifest, namespace, config, ref = prepare(args, saved, emit=emit)
    cache = FeatureCache(args.feature_cache, namespace)
    paths, coverage = fill_cache(cache, [(train, range(len(train))), (val, [r["index"] for r in manifest["pairs"]])],
        checkpoint=ref["path"], device=args.device, limit=args.precompute_limit, emit=emit)
    write_json(args.output_dir/"cache_coverage.json", dict(**coverage, namespace=namespace, images=cache.image_inventory,
                                                        paths=list(map(str, sorted(set(paths))))))
    return coverage


if __name__ == "__main__":
    main()
