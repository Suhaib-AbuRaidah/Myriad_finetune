"""Initial-only inference for one enumerated train/validation pair."""
import argparse
import json
import numpy as np
import torch

from finetuning_pair.testing import sample_rollout
from training_from_scratch.dataset import _backproject
from training_from_scratch.full_evaluation.metrics import clean
from .config import parse_args, prepare, write_json
from .checkpoint_io import _read_checkpoint
from .features import FeatureCache, fill_cache
from .geometry import SquareCrop
from .model import CachedFP32Model, build_cached_myriad
from .fitting import fit_predictions


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("--split", choices=("train","val"), default="val")
    opts,rest=p.parse_known_args(argv)
    args,saved=parse_args(rest,mode="single-pair inference")
    if not args.resume:
        raise ValueError("Supply --resume best.pt or latest.pt")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    emit=lambda r: print(json.dumps(r,sort_keys=True),flush=True)
    train,val,manifest,namespace,config,ref=prepare(args,saved,emit=emit)
    data=train if opts.split=="train" else val
    data.training=False
    if not 0 <= opts.pair_index < len(data):
        raise ValueError("Pair index outside dataset inventory")
    cache=FeatureCache(args.feature_cache,namespace,require_cached=not args.precompute)
    fill_cache(cache,[(data,[opts.pair_index])],checkpoint=ref["path"],device=args.device,emit=emit)
    cache.require_cached=True
    model=CachedFP32Model(build_cached_myriad(image_size=args.image_size)).place(torch.device(args.device))
    model.configure_cropped_cache(cache)
    payload=_read_checkpoint(args.resume)
    model.load_state_dict(payload["model"],strict=True)
    sample=data[opts.pair_index]
    selected=sample["point_valid"] & sample["source_position_valid"][0]
    pred=sample_rollout(model,sample["x"],sample["trajectories"][:1,selected],sample["state_times"],
        ensemble_size=1,diffusion_steps=args.val_diffusion_steps,seed=args.val_seed+opts.pair_index,
        amp_dtype="none")[0].float().cpu().numpy()
    crop=SquareCrop(*sample["crop_bounds"].tolist())
    uv=crop.unnormalize(pred)
    with np.load(sample["scene_path"],allow_pickle=False) as a:
        depth=a["depth"][0].copy(); depth[~a["depth_valid_mask"][0].astype(bool)]=0
        seg=a["link_segmentation"][0]
    pixels=sample["reference_pixels"].numpy()[selected]
    k,e=sample["camera_intrinsic_source"].numpy(),sample["camera_extrinsic"].numpy()
    xyz=_backproject(pixels,depth[np.rint(pixels[:,1]).astype(int),np.rint(pixels[:,0]).astype(int)],k,e)
    fit=fit_predictions(xyz,uv,k,e,initial_depth=depth,initial_segmentation=seg,
                        pair_labels=sample["pair_labels"].numpy())
    out=args.output_dir/"inference"/f"{opts.split}_pair_{opts.pair_index:05d}_step_{saved['step']:08d}"
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"),predictions_source_pixels=uv,predictions_crop_normalized=pred,
        reference_world=xyz,point_indices=np.flatnonzero(selected.numpy()),
        crop_transform=crop.matrix,crop_inverse=np.linalg.inv(crop.matrix),
        camera_intrinsic=k,camera_extrinsic=e,state_times=sample["state_times"].numpy())
    write_json(out.with_suffix(".json"),clean(dict(scene_path=sample["scene_path"],pair=sample["pair_labels"].tolist(),
        joint_index=sample["joint_index"],initial_states_supplied=1,fit=fit)))
    emit(dict(event="inference_complete",output=str(out)))


if __name__ == "__main__":
    main()
