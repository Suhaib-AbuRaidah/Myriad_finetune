"""Freeze checkpoint, source scenes and split selection before inference."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

from finetuning_pair.evaluate import file_hash
from training_from_scratch_cropped.checkpoint_io import _read_checkpoint, resolve_encoder_reference
from training_from_scratch_cropped.config import write_json
from training_from_scratch_cropped.dataset import ArticulatedDataset
from training_from_scratch_cropped.paths import artifact_path
import numpy as np

DEFAULT_OUT = Path("training_from_scratch_cropped/runs/full_test1_val80_20261021")
DEFAULT_CHECKPOINT = Path("training_from_scratch_cropped/runs/cropped/best.pt")
KINDS = {0:"revolute", 1:"prismatic", 4:"fixed"}


def snapshot_file(source, destination):
    """Hard-link immutable local inputs, with an independent-copy fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): return
    try:
        os.link(source, destination)
    except OSError:
        temporary = destination.with_suffix(destination.suffix+".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def make_dataset(manifest, split):
    cfg = manifest["train_config"]
    dataset = ArticulatedDataset(cfg["data_root"], split=split, training=False,
        seed=manifest["seed"], image_size=cfg["image_size"], points_per_link=cfg["points_per_link"],
        num_transitions=cfg["num_transitions"], time_scale=cfg["time_scale"],
        overlay_alpha=cfg["overlay_alpha"], crop_padding=cfg["crop_padding"],
        cache_dir=manifest["scene_cache"])
    wanted = set(manifest["splits"][split]["original_scene_indices"])
    lookup = {(scene, tuple(pair)):index for index,(scene,pair) in enumerate(dataset.examples)}
    indices = [i for i in range(len(dataset.files)) if i in wanted]
    dataset.files = [dataset.files[i] for i in indices]
    dataset.pairs = [dataset.pairs[i] for i in indices]
    dataset.examples = [(scene, pair) for scene, pairs in enumerate(dataset.pairs) for pair in pairs]
    dataset.original_pair_indices = [lookup[indices[scene], tuple(pair)] for scene,pair in dataset.examples]
    dataset.training_indices = [i for i, pairs in enumerate(dataset.pairs) if pairs]
    archived = manifest["splits"][split]["scenes"]
    if archived:
        dataset.files = [Path(manifest["output_dir"])/split/row["archive"] for row in archived]
    return dataset


def select_validation(files, count=80, seed=20261021):
    groups = defaultdict(list)
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as arrays:
            obj = json.loads(str(arrays["metadata_json"]))["object"]["object_id"]
        groups[obj].append(index)
    rng = np.random.default_rng(seed)
    keys = sorted(groups); rng.shuffle(keys)
    pools = {key:list(rng.permutation(groups[key]).astype(int)) for key in keys}
    selected = []
    while len(selected)<count:
        for key in keys:
            if pools[key]: selected.append(int(pools[key].pop()))
            if len(selected)==count: break
    return sorted(selected), dict(groups)


def prepare(output, checkpoint, *, ensemble_size=5, diffusion_steps=16, seed=42):
    output = artifact_path(output); output.mkdir(parents=True, exist_ok=True)
    destination = output/"checkpoint.pt"
    if (output/"run.json").exists():
        manifest = json.loads((output/"run.json").read_text())
        assert manifest["ensemble_size"]==ensemble_size and manifest["diffusion_steps"]==diffusion_steps and manifest["seed"]==seed
        assert file_hash(destination)==manifest["checkpoint_sha256"]
        return manifest
    snapshot_file(checkpoint.resolve(), destination)
    payload = _read_checkpoint(destination)
    cfg = payload["train_config"]
    assert cfg["precision"]=="fp32" and cfg["freeze_dino"] and cfg["num_transitions"]==4
    reference = resolve_encoder_reference(payload["frozen_encoder"])
    manifest = dict(schema_version="cropped_full_trajectory_archive_v1", output_dir=str(output), checkpoint_source=str(checkpoint.resolve()),
        checkpoint_sha256=file_hash(destination), checkpoint_step=payload["step"], train_config=cfg,
        frozen_encoder=reference, feature_cache=str(artifact_path("training_from_scratch_cropped/feature_cache")),
        scene_cache=str(artifact_path("training_from_scratch_cropped/cache")),
        ensemble_size=ensemble_size, diffusion_steps=diffusion_steps, seed=seed,
        primary_hypothesis=0, known_states=1, poke=False, precision="fp32", amp="none",
        fitting="not run; archive intended for subsequent frozen candidate evaluation",
        target_definition=cfg["target_definition"],
        validation_selection=dict(count=80, seed=20261021,
            method="round-robin object coverage, seeded scene order, no metric-based selection; all eligible joints"), splits={})
    for split in ("test1", "val"):
        files = sorted((Path(cfg["data_root"])/split/"scenes").glob("*.npz"))
        if split=="test1":
            assert len(files)==320
            indices = list(range(len(files)))
        else:
            indices, groups = select_validation(files)
            manifest["validation_selection"]["objects"] = len(groups)
        manifest["splits"][split] = dict(original_scene_indices=indices, scenes=[])
        dataset = make_dataset(manifest, split)
        inventory = []
        for local, (original, path) in enumerate(zip(indices, dataset.files)):
            archive = output/split/"scenes"/path.name
            snapshot_file(path.resolve(), archive)
            with np.load(archive, allow_pickle=False) as arrays:
                meta = json.loads(str(arrays["metadata_json"]))
                js, kinds = arrays["joint_indices"], arrays["joint_types"]
                pa, ch = arrays["joint_parent_link_indices"], arrays["joint_child_link_indices"]
            eligible = {pair[2] for pair in dataset.pairs[local]}
            for j, kind, parent, child in zip(js, kinds, pa, ch):
                inventory.append(dict(scene_index=local, joint_index=int(j), joint_ordinal=int(j)+1,
                    kind=KINDS.get(int(kind), "other"), parent=int(parent), child=int(child),
                    eligible=int(j) in eligible, exclusion_reason=None if int(j) in eligible else "no initially visible valid-depth child pixels"))
            stat = path.stat()
            manifest["splits"][split]["scenes"].append(dict(scene_index=local, original_scene_index=original,
                path=str(path.resolve()), archive=str(archive.relative_to(output/split)),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(archive),
                object_id=meta["object"]["object_id"], sample_id=meta.get("sample_id", path.stem)))
        manifest["splits"][split].update(pairs=[[scene, list(pair)] for scene,pair in dataset.examples],
            original_pair_indices=dataset.original_pair_indices, inventory=inventory)
        print(json.dumps(dict(event="split_ready", split=split, scenes=len(indices),
            pairs=len(dataset), source_joints=len(inventory))), flush=True)
    sources = {str(path):file_hash(path) for path in sorted(Path("training_from_scratch_cropped/full_inference").glob("*.py"))}
    manifest["archive_implementation_sha256"] = sources
    write_json(output/"run.json", manifest)
    return manifest
