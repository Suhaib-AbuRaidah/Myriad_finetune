"""Shared configuration, inventories, and validation/cache identity."""
import argparse
import json
import math
import os
from pathlib import Path

from .checkpoint_io import inspect_checkpoint, make_encoder_reference, resolve_encoder_reference
from .dataset import ArticulatedDataset
from .paths import ROOT, artifact_path
from .selection import select_validation, selection_spec
from .traversal import inventory
from .scoring import DEFINITION

DEFAULTS = dict(data_root="data/Robotic_arm", batch_size=1, gradient_accumulation_steps=4,
    lr=1e-4, weight_decay=.01, warmup_steps=500, max_steps=200000, grad_clip=1.,
    points_per_link=8, image_size=384, num_transitions=4, time_scale=4.,
    max_train_scenes=None, seed=42, val_seed=42, crop_padding=.1, overlay_alpha=.85,
    feature_cache_dtype="fp16", gradient_checkpointing=False,
    save_every_steps=1000, val_every_steps=1000, val_diffusion_steps=16, log_every_steps=100)


def parse_args(argv=None, *, mode="train"):
    p = argparse.ArgumentParser(description=f"Isolated cropped Myriad {mode}")
    p.add_argument("--resume", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--scene-cache", type=Path)
    p.add_argument("--dino-checkpoint", type=Path)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--optimizer-device", default="cuda:0")
    p.add_argument("--run-steps", type=int)
    p.add_argument("--smoke-test", action="store_true",
                   help="Bounded check with temporary outputs and no checkpoint writes")
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=None,
                   help="Defaults off for bounded --run-steps; always off for --smoke-test")
    p.add_argument("--evaluate-at-end", action="store_true")
    p.add_argument("--precompute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precompute-limit", type=int)
    p.add_argument("--threads", type=int, default=4)
    for key, value in DEFAULTS.items():
        if isinstance(value, bool):
            p.add_argument("--"+key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument("--"+key.replace("_", "-"), type=int if value is None else type(value))
    args = p.parse_args(argv)
    if args.smoke_test:
        if args.output_dir is not None:
            p.error("--smoke-test uses temporary outputs; omit --output-dir")
        if args.save_checkpoints:
            p.error("--smoke-test cannot save checkpoints")
        args.run_steps = args.run_steps or 2
    args.feature_cache_explicit = args.feature_cache is not None
    args.scene_cache_explicit = args.scene_cache is not None
    if args.save_checkpoints is None:
        args.save_checkpoints = args.run_steps is None and not args.smoke_test
    saved = inspect_checkpoint(args.resume) if args.resume else None
    old = saved["train_config"] if saved else {}
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, old.get(key, value))
    if args.smoke_test and args.resume is None and args.max_train_scenes is None:
        args.max_train_scenes = 1
    args.output_dir = artifact_path(args.output_dir or (args.resume.parent if args.resume else ROOT/"runs"/"cropped"))
    args.feature_cache = artifact_path(args.feature_cache or ROOT/"feature_cache")
    args.scene_cache = artifact_path(args.scene_cache or ROOT/"scene_cache")
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.dino_checkpoint = args.dino_checkpoint or Path(saved["frozen_encoder"]["path"] if saved else "myriad_openset.pt")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Use one process with cuda:1 motion model / cuda:0 AdamW")
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "image_size", "threads",
                "save_every_steps", "val_every_steps", "val_diffusion_steps", "log_every_steps"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.points_per_link != 8 or args.num_transitions != 4 or args.image_size % 16:
        p.error("Requires eight child points, four transitions, image-size divisible by 16")
    if not 0 <= args.warmup_steps < args.max_steps or min(args.seed, args.val_seed) < 0:
        p.error("Invalid warmup or seed")
    for key in ("lr", "weight_decay", "grad_clip", "time_scale", "crop_padding", "overlay_alpha"):
        if not math.isfinite(getattr(args,key)):
            p.error(f"{key} must be finite")
    if min(args.lr, args.grad_clip, args.time_scale) <= 0 or min(args.crop_padding,args.weight_decay) < 0 or not 0 <= args.overlay_alpha <= 1:
        p.error("Invalid optimizer, crop, or overlay settings")
    if args.feature_cache_dtype not in ("fp16", "fp32"):
        p.error("feature-cache-dtype must be fp16 or fp32")
    for key in ("max_train_scenes", "run_steps", "precompute_limit"):
        if getattr(args,key) is not None and getattr(args,key) < 1:
            p.error(f"{key} must be positive")
    return args, saved


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temp.replace(path)


def prepare(args, saved=None, *, emit=print):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = (resolve_encoder_reference(saved["frozen_encoder"], args.dino_checkpoint) if saved
                 else make_encoder_reference(args.dino_checkpoint))
    def dataset(split, training):
        return ArticulatedDataset(args.data_root, split=split, training=training,
            image_size=args.image_size, points_per_link=args.points_per_link,
            num_transitions=args.num_transitions, time_scale=args.time_scale,
            overlay_alpha=args.overlay_alpha, crop_padding=args.crop_padding,
            seed=args.seed if training else args.val_seed,
            max_scenes=args.max_train_scenes if training else None,
            cache_dir=args.scene_cache)
    train = dataset("train", True)
    val = dataset("val", False)
    inv = inventory(train)
    inv_path = args.output_dir/"pair_inventory.json"
    if inv_path.exists() and json.loads(inv_path.read_text()) != inv:
        raise ValueError("Existing training inventory differs")
    write_json(inv_path, inv)
    emit(dict(event="inventory", train_scenes=len(train.training_indices), train_pairs=len(train), val_pairs=len(val)))
    manifest_path = args.output_dir/"validation_manifest.json"
    reviewed_manifest = ROOT/"manifests"/"validation_100_revolute.json"
    if not manifest_path.exists() and reviewed_manifest.exists():
        reviewed = json.loads(reviewed_manifest.read_text())
        if reviewed["spec"] == selection_spec(val,args.val_seed):
            # Copy the reviewed cohort only for an identical source/config;
            # existing run manifests remain immutable and checked below.
            write_json(manifest_path,select_validation(val,reviewed_manifest,seed=args.val_seed))
    manifest = select_validation(val,manifest_path,seed=args.val_seed)
    namespace = dict(version="cropped_dino_v1", encoder=reference["identity"],
        train_inventory=inv["sha256"], val_inventory=inventory(val)["sha256"],
        image_size=args.image_size, crop_padding=args.crop_padding, overlay_alpha=args.overlay_alpha,
        preprocessing="initial_union_square_edgepad_bilinear_antialias_pixelcenter_imagenet_v1",
        roles="parent_0_120_255_child_255_145_0_existing_blend_roles", dino_precision="fp32",
        feature_cache_dtype=args.feature_cache_dtype)
    config = {k:getattr(args,k) for k in DEFAULTS}
    config.update(data_root=str(args.data_root), training_implementation="cropped_all_pairs_fp32_v1",
        target_definition=train[0]["target_definition"], architecture="myriad_large_dinov3_large",
        initialization="random_motion_pretrained_frozen_dino", freeze_dino=True,
        dino_identity=reference["identity"], dino_precision="fp32", precision="fp32", amp="none",
        pair_selection="scene_shuffle_all_joints_v1", train_inventory=inv["sha256"],
        validation_manifest=manifest["sha256"], cache_namespace=namespace,
        scoring=DEFINITION, future_crop_policy="retain_originally_valid_unclamped")
    if saved:
        compare_config(saved["train_config"], config)
    config_path = args.output_dir/"config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if saved or any((args.output_dir/n).exists() for n in ("latest.pt","best.pt")):
            compare_config(previous, config)
        else:
            # A precompute plan fixes source/crop/cohort identity, not a motion
            # optimizer that has not been allocated yet.
            plan_keys = ("data_root","train_inventory","validation_manifest","cache_namespace",
                         "pair_selection","target_definition","points_per_link","num_transitions","time_scale")
            compare_config({k:previous[k] for k in plan_keys},{k:config[k] for k in plan_keys})
    write_json(config_path, config)
    emit(dict(event="validation_manifest", pairs=100, scenes=manifest["scene_count"],
              objects=manifest["object_count"], axis_cohort=len(manifest["axis_cohort"]), constraints=manifest["constraints"]))
    return train, val, manifest, namespace, config, reference


def compare_config(old, new):
    ignore = {"optimizer_parameter_names", "save_every_steps", "val_every_steps", "log_every_steps"}
    changed = [k for k in old.keys() | new.keys() if k not in ignore and old.get(k) != new.get(k)]
    if changed:
        raise ValueError("Incompatible resume/configuration: "+", ".join(sorted(changed)))
