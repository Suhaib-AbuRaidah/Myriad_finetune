"""Fill missing original-image DINO features with the frozen encoder, once."""
import argparse
import json
from pathlib import Path
import time
import torch

from training_from_scratch.checkpoint_io import _read_checkpoint, resolve_encoder_reference
from training_from_scratch.features import FeatureCache, encoder_identity, load_frozen_encoder
from training_from_scratch.model import build_myriad
from .inference import selected_dataset as make_dataset
from finetuning_pair.evaluate import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("training_from_scratch/runs/val50_contact_axis_transfer_20261021"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    manifest = json.loads((args.evaluation/"run.json").read_text())
    cfg = manifest["train_config"]
    data_args = argparse.Namespace(data_root=None, split="val", seed=manifest["seed"],
        cache_dir=Path("training_from_scratch/cache"))
    dataset = make_dataset(data_args, cfg)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder_dtype = {"fp16":torch.float16, "fp32":torch.float32}[cfg["dino_precision"]]
    feature_dtype = {"fp16":torch.float16, "fp32":torch.float32}[cfg["feature_cache_dtype"]]
    cache = FeatureCache(manifest["feature_cache"], identity=cfg["dino_identity"], image_size=cfg["image_size"],
        dtype=encoder_dtype, feature_dtype=feature_dtype)
    encoder = None; started = time.perf_counter()
    from myriad.model import make_axial_pos_2d
    for index in range(len(dataset)):
        image = dataset[index]["x"].unsqueeze(0)
        if cache.read(image) is None:
            if encoder is None:
                payload = _read_checkpoint(args.evaluation/"checkpoint.pt")
                reference = resolve_encoder_reference(payload["frozen_encoder"])
                assert encoder_identity(reference["path"]) == cfg["dino_identity"]
                encoder = build_myriad(encoder_only=True)
                load_frozen_encoder(encoder, reference["path"])
                encoder.to(device=device, dtype=encoder_dtype).eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                    enabled=encoder_dtype == torch.float16):
                features = encoder(image.to(device=device, dtype=encoder_dtype)).to(feature_dtype)
                b, _, h, w = features.shape
                positions = make_axial_pos_2d(h, w, device=device).reshape(1, h, w, 2).expand(b, -1, -1, -1)
                assert torch.isfinite(features).all()
                cache.write(image, dict(feat=features, pos=positions, L=h*w))
        if (index+1)%50 == 0 or index+1==len(dataset):
            status = dict(stage="cache_validation_features", pairs=index+1, total=len(dataset),
                hits=cache.hits, computed=cache.misses, seconds=time.perf_counter()-started)
            write_json(args.evaluation/"cache_status.json", status)
            print(json.dumps(status), flush=True)


if __name__ == "__main__": main()
