"""Whitelist initial geometry for fitting; keep all scoring truth outside it."""
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from testing_joint_fitting.experiment import load_cases, write
from testing_joint_fitting.geometric import contact_evidence
from training_from_scratch.full_evaluation.metrics import kinematic_reference
from finetuning.joint_parame_estim.run import from_normalized

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


@dataclass(frozen=True)
class Inputs:
    xyz: np.ndarray
    uv: np.ndarray
    valid: np.ndarray
    k: np.ndarray
    e: np.ndarray
    parent: np.ndarray
    child: np.ndarray
    parent_normals: np.ndarray
    child_normals: np.ndarray
    rgbd_contact: dict | None


def load():
    subset_path = ROOT.parent / "results/subset.json"
    subset = json.loads(subset_path.read_text())
    evaluation = Path(subset["evaluation"])
    manifest = json.loads((evaluation / "run.json").read_text())
    cache = OUT / "cases.pkl"
    cache_manifest = OUT / "cache_manifest.json"
    if cache.exists() and cache_manifest.exists():
        cached = json.loads(cache_manifest.read_text())
        for name, expected in cached["files"].items():
            stat = Path(name).stat()
            assert [stat.st_size, stat.st_mtime_ns] == expected, f"Changed input: {name}"
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == cached["sha256"]
        # This cache is produced locally by this loader, never supplied externally.
        return pickle.loads(cache.read_bytes())
    mesh_cache = {}
    fingerprints = {}
    cases = []
    for case in load_cases(evaluation, set(subset["scenes"]), "without_poke"):
        if case["original"]["gt_type"] != "revolute":
            continue
        original, source = case["original"], case["source"]
        scene = original["scene_index"]
        if scene not in mesh_cache:
            path = Path(manifest["scenes"][scene]["path"])
            with np.load(path, allow_pickle=False) as arrays:
                # State zero is the observed random initial pose, NOT rest state 4.
                mesh_cache[scene] = tuple(arrays[key][0].copy() for key in
                    ("mesh_points", "mesh_normals", "mesh_point_link_labels"))
            stat = path.stat()
            fingerprints[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                initial_mesh_sha256=hashlib.sha256(b"".join(a.tobytes() for a in mesh_cache[scene])).hexdigest())
        points, normals, labels = mesh_cache[scene]
        parent, child = [labels == int(label) for label in case["labels"]]
        data = Inputs(case["xyz"].astype(float), case["uv"].astype(float), case["valid"],
            source["camera_intrinsic"].astype(float), source["camera_extrinsic"].astype(float),
            points[parent].astype(float), points[child].astype(float),
            normals[parent].astype(float), normals[child].astype(float),
            contact_evidence(case["context"], source["camera_intrinsic"], source["camera_extrinsic"]))
        transforms = source["link_transforms_world"].astype(float)
        rotations = []
        for state in (2, 3, 4):
            relative = (transforms[0, case["parent"]] @ np.linalg.inv(transforms[state, case["parent"]])
                @ transforms[state, case["child"]] @ np.linalg.inv(transforms[0, case["child"]]))
            rotations.append(relative[:3, :3])
        reference = kinematic_reference(source, case["joint_position"], case["parent"], case["child"])
        h, w = source["initial_depth"].shape
        cases.append(dict(data=data,
            **{key: original[key] for key in ("pair_index", "scene_index", "joint_ordinal", "object_id")},
            gt_uv=from_normalized(case["ground_truth"], h, w, int(source["image_size"])),
            gt_axis=source["joint_axes_world"][0, case["joint_position"]].astype(float),
            gt_pivot=np.asarray(reference["pivot"]) if reference.get("observable") else None,
            gt_rotations=np.stack(rotations)))
    assert len(cases) == 204 and len({c["scene_index"] for c in cases}) == 80
    provenance = dict(subset=subset, pairs=[c["pair_index"] for c in cases], mesh_sources=fingerprints,
        fitting_inputs="initial world mesh points/normals/part labels; initial RGB-D anchors; saved hypothesis 0; camera; existing validity mask",
        truth_use="revolute cohort definition and scoring only; explicitly named GT-track controls",
        states=[2, 3, 4], coordinate_frame="world", units="source dataset length units; source image pixels")
    destination = OUT / "manifest.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == provenance, "Inputs changed; use a new experiment directory"
    else:
        write(destination, provenance)
    paths = {subset_path, evaluation / "run.json"}
    paths.update(Path(name) for name in fingerprints)
    for path in (evaluation / "records").glob("*_without_poke.json"):
        record = json.loads(path.read_text())
        if record["scene_index"] in subset["scenes"]:
            paths.add(path)
            prediction = evaluation / "predictions" / f"{record['pair_index']:04d}_without_poke.npz"
            if prediction.exists(): paths.add(prediction)
    paths.update((evaluation / "sources").glob("*.npz"))
    payload = pickle.dumps(cases, protocol=5)
    cache.write_bytes(payload)
    write(cache_manifest, dict(sha256=hashlib.sha256(payload).hexdigest(),
        files={str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(paths)}))
    return cases


def snapshot():
    files = {p.name: p.read_text() for p in sorted(ROOT.glob("*.py"))}
    for name in ("geometric.py", "experiment.py"):
        files["dependencies/" + name] = (ROOT.parent / name).read_text()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    for name, content in files.items():
        target = OUT / "code" / digest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return digest
