"""Validate a compact cropped checkpoint on its immutable 100-pair cohort."""
import json
import torch
from .checkpoint_io import _read_checkpoint
from .config import parse_args, prepare, write_json
from .features import FeatureCache, fill_cache
from .model import CachedFP32Model, build_cached_myriad
from .validation import evaluate


def main(argv=None):
    args,saved = parse_args(argv,mode="validation/inference")
    if args.resume is None:
        raise ValueError("Supply --resume path/to/best.pt or latest.pt")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    emit = lambda r: print(json.dumps(r,sort_keys=True),flush=True)
    train,val,manifest,namespace,config,ref = prepare(args,saved,emit=emit)
    cache = FeatureCache(args.feature_cache,namespace,require_cached=not args.precompute)
    paths,coverage = fill_cache(cache,[(val,[r["index"] for r in manifest["pairs"]])],
        checkpoint=ref["path"],device=args.device,emit=emit)
    if coverage["remaining"]:
        raise ValueError("Validation cache incomplete")
    cache.require_cached=True
    model = CachedFP32Model(build_cached_myriad(image_size=args.image_size)).place(torch.device(args.device))
    model.frozen_encoder_reference=ref
    model.configure_cropped_cache(cache)
    payload=_read_checkpoint(args.resume)
    model.load_state_dict(payload["model"],strict=True)
    result=evaluate(model,val,manifest,diffusion_steps=args.val_diffusion_steps,seed=args.val_seed,emit=emit,
        predictions_dir=args.output_dir/"inference"/f"step_{saved['step']:08d}")
    write_json(args.output_dir/"inference"/f"step_{saved['step']:08d}.json",result)
    emit(result["overall"])
    return result


if __name__ == "__main__":
    main()
