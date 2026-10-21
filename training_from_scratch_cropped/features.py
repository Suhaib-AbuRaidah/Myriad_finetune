"""Validated, atomic persistent FP32-DINO / FP16-storage feature cache."""
import gc
import hashlib
import json
import pickle
import time
import numpy as np
from pathlib import Path

import torch
from finetuning_pair.checkpoint_io import _atomic_save
from training_from_scratch.features import load_frozen_encoder
from training_from_scratch.model import build_myriad
from .paths import artifact_path


class FeatureCache:
    def __init__(self, root, namespace, *, require_cached=False, channels=1024):
        self.root = artifact_path(root)
        self.namespace = json.dumps(namespace, sort_keys=True)
        self.root = self.root / hashlib.sha256(self.namespace.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        self.require_cached, self.channels = require_cached, channels
        self.size = namespace["image_size"]
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32}[namespace["feature_cache_dtype"]]
        self.hits = self.misses = self.invalid = 0

    def path(self, image):
        image = image.detach().cpu().float().contiguous().reshape(1, 3, self.size, self.size)
        digest = hashlib.sha256(self.namespace.encode())
        digest.update(image.numpy().tobytes())
        return self.root / f"{digest.hexdigest()}.pt"

    def validate(self, saved):
        if saved["namespace"] != self.namespace:
            raise ValueError("Cache metadata mismatch")
        row = saved["embedding"]
        h = self.size//16
        if row["L"] != h*h or row["feat"].shape != (1, self.channels, h, h) or row["pos"].shape != (1, h, h, 2):
            raise ValueError("Invalid feature shape/token count")
        if row["feat"].dtype != self.dtype or row["pos"].dtype != torch.float32:
            raise ValueError("Invalid feature storage precision")
        if not all(torch.isfinite(row[k]).all() for k in ("feat", "pos")):
            raise ValueError("Nonfinite cache entry")
        return row

    def read_path(self, path, *, count=True, strict=None):
        row = None
        if Path(path).exists():
            try:
                row = self.validate(torch.load(path, map_location="cpu", weights_only=True))
            except (ValueError, KeyError, RuntimeError, EOFError, OSError, TypeError, AttributeError, pickle.UnpicklingError):
                if count:
                    self.invalid += 1
        if count:
            if row is None:
                self.misses += 1
            else:
                self.hits += 1
        if row is None and self.require_cached and strict is not False:
            raise FileNotFoundError(f"Missing/invalid cropped feature {path}; run precompute")
        return row

    def read(self, image):
        return self.read_path(self.path(image))

    def write(self, image, row):
        path = self.path(image)
        if self.read_path(path, count=False, strict=False) is not None:
            return
        saved = dict(namespace=self.namespace, embedding=dict(
            feat=row["feat"].detach().to("cpu", self.dtype),
            pos=row["pos"].detach().to("cpu", torch.float32), L=int(row["L"])))
        self.validate(saved)
        _atomic_save(saved, path)

    def coverage(self, paths):
        paths = sorted(set(map(str, paths)))
        valid = sum(self.read_path(path, count=False, strict=False) is not None for path in paths)
        return dict(required=len(paths), valid=valid, remaining=len(paths)-valid,
                    hits=self.hits, misses=self.misses, invalid=self.invalid)


def fill_cache(cache, datasets, *, checkpoint, device, limit=None, emit=print):
    """Only allocate DINO if a miss occurs; release it before optimizer creation."""
    encoder = None
    started = time.perf_counter()
    paths, computed = [], 0
    cache.image_inventory = []
    try:
        for dataset, indices in datasets:
            for index in indices:
                initial = dataset.initial_image(index)
                image, crop = initial["x"].unsqueeze(0), initial["crop"]
                paths.append(cache.path(image))
                scene, pair = dataset.examples[index]
                cache.image_inventory.append(dict(source=str(dataset.files[scene].resolve()),
                    split=dataset.scene_dir.parent.name, parent=pair[0], child=pair[1], joint_index=pair[2],
                    feature_path=str(paths[-1]), crop_bounds=[crop.top,crop.left,crop.side,crop.image_size],
                    source_to_crop=crop.matrix.tolist(),crop_to_source=np.linalg.inv(crop.matrix).tolist()))
                if cache.read(image) is None:
                    if limit is not None and computed >= limit:
                        continue
                    if encoder is None:
                        emit(dict(event="loading_frozen_encoder",device=str(device),precision="fp32"))
                        loading = time.perf_counter()
                        encoder = build_myriad(encoder_only=True)
                        encoder.image_size = cache.size
                        load_frozen_encoder(encoder, checkpoint)
                        encoder.to(device=device, dtype=torch.float32).eval()
                        emit(dict(event="frozen_encoder_ready",seconds=time.perf_counter()-loading))
                    from myriad.model import make_axial_pos_2d
                    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
                        feat = encoder(image.to(device))
                        _, _, h, w = feat.shape
                        pos = make_axial_pos_2d(h, w, device=device).reshape(1, h, w, 2)
                        cache.write(image, dict(feat=feat, pos=pos, L=h*w))
                    computed += 1
                if len(paths) % 100 == 0:
                    emit(dict(event="precompute_progress", visited=len(paths), computed=computed,
                              hits=cache.hits, misses=cache.misses))
    finally:
        if encoder is not None:
            del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    coverage = cache.coverage(paths)
    emit(dict(event="cache_coverage", **coverage, encoder_released=True,
              precompute_seconds=time.perf_counter()-started))
    return paths, coverage
