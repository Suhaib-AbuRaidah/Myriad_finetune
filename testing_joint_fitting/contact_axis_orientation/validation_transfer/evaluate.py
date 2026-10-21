"""Evaluate frozen fitting on every scoreable validation revolute pair; no tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from testing_joint_fitting.experiment import load_cases, write
from testing_joint_fitting.geometric import contact_evidence
from testing_joint_fitting.axis_motion.common import score, summary
from testing_joint_fitting.contact_axis_orientation.data import Inputs
from testing_joint_fitting.contact_axis_orientation.candidate import CONFIG
from testing_joint_fitting.contact_axis_orientation.solver import fit
from training_from_scratch.full_evaluation.metrics import kinematic_reference
from finetuning.joint_parame_estim.run import from_normalized

ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATION = Path("training_from_scratch/runs/val50_contact_axis_transfer_20261021")
CONFIGS = dict(original=dict(baseline="original"), rgbd_position=dict(baseline="rgbd"),
    mesh_position=dict(baseline="mesh"), frozen_candidate=dict(CONFIG))


def frozen_sources():
    repo = ROOT.parents[2]
    relative = ["testing_joint_fitting/contact_axis_orientation/"+name for name in
        ("candidate.py", "solver.py", "geometry.py", "data.py")]
    relative += ["testing_joint_fitting/geometric.py", "finetuning_pair/analyze_joints.py",
        "finetuning/joint_parame_estim/run.py", "testing_joint_fitting/axis_motion/common.py"]
    return {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in relative}


def freeze_protocol():
    subset = json.loads((ROOT.parent.parent/"results/subset.json").read_text())
    selected = json.loads((ROOT.parent/"results/selected.json").read_text())
    assert CONFIG == selected["config"]
    protocol = dict(checkpoint_sha256=subset["checkpoint_sha256"], checkpoint_step=108000,
        configs=CONFIGS, algorithm_sources=frozen_sources(), split="val", all_scenes=False, scenes=50,
        subset=json.loads((ROOT/"subset50.json").read_text()),
        primary_cohort="all scoreable GT-revolute pairs, assumed revolute; skipped coverage reported separately",
        primary_hypothesis=0, ensemble_size=5, diffusion_steps=16, mode="without_poke", seed=42,
        geometry_states=[2, 3, 4], fitting_geometry_state=0,
        tuning="none; recommended strength 128 and all thresholds frozen from test1 development",
        controls="separate GT-track replacements; never used to change the candidate or choose a validation configuration")
    destination = ROOT/"protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == protocol, "Frozen protocol changed"
    else:
        write(destination, protocol)
    return protocol


def load_validation(evaluation, output):
    manifest = json.loads((evaluation/"run.json").read_text())
    protocol = freeze_protocol()
    assert len(manifest["scenes"]) == 50
    assert [scene["path"] for scene in manifest["scenes"]] == protocol["subset"]["paths"]
    assert manifest["checkpoint_sha256"] == protocol["checkpoint_sha256"]
    for key in ("split", "checkpoint_step", "ensemble_size", "diffusion_steps", "seed"):
        assert manifest[key] == protocol[key], key
    assert manifest["modes"] == ["without_poke"]
    records = [json.loads(p.read_text()) for p in sorted((evaluation/"records").glob("*_without_poke.json"))]
    assert len(records) == len(manifest["pairs"]), "Inference is incomplete"
    cases, meshes = [], {}
    fingerprints = {}
    scenes = set(range(len(manifest["scenes"])))
    for case in load_cases(evaluation, scenes, "without_poke"):
        original, source = case["original"], case["source"]
        if original["gt_type"] != "revolute": continue
        scene = original["scene_index"]
        if scene not in meshes:
            path = Path(manifest["scenes"][scene]["path"])
            with np.load(path, allow_pickle=False) as arrays:
                meshes[scene] = tuple(arrays[key][0].copy() for key in
                    ("mesh_points", "mesh_normals", "mesh_point_link_labels"))
            stat = path.stat()
            fingerprints[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                initial_mesh_sha256=hashlib.sha256(b"".join(a.tobytes() for a in meshes[scene])).hexdigest())
        points, normals, labels = meshes[scene]
        masks = [labels == int(label) for label in case["labels"]]
        data = Inputs(case["xyz"].astype(float), case["uv"].astype(float), case["valid"],
            source["camera_intrinsic"].astype(float), source["camera_extrinsic"].astype(float),
            points[masks[0]].astype(float), points[masks[1]].astype(float),
            normals[masks[0]].astype(float), normals[masks[1]].astype(float),
            contact_evidence(case["context"], source["camera_intrinsic"], source["camera_extrinsic"]))
        transforms = source["link_transforms_world"].astype(float)
        rotations = []
        for state in (2, 3, 4):
            relative = (transforms[0, case["parent"]]@np.linalg.inv(transforms[state, case["parent"]])
                @ transforms[state, case["child"]]@np.linalg.inv(transforms[0, case["child"]]))
            rotations.append(relative[:3, :3])
        reference = kinematic_reference(source, case["joint_position"], case["parent"], case["child"])
        h, w = source["initial_depth"].shape
        cases.append(dict(data=data,
            **{key:original[key] for key in ("pair_index", "scene_index", "joint_ordinal", "object_id")},
            gt_uv=from_normalized(case["ground_truth"], h, w, int(source["image_size"])),
            gt_axis=source["joint_axes_world"][0, case["joint_position"]].astype(float),
            gt_pivot=np.asarray(reference["pivot"]) if reference.get("observable") else None,
            gt_rotations=np.stack(rotations),
            trajectory_metrics={key:original.get(key) for key in
                ("ADE_1_pixels", "ADE_percent_full_path", "gt_path_pixels", "minADE_1")}))
    cohort = dict(scenes=len(manifest["scenes"]), source_joints=len(manifest["inventory"]),
        eligible_pairs=len(records), eligible_revolute_pairs=sum(r["gt_type"]=="revolute" for r in records),
        excluded_source_revolute_pairs=sum(r["gt_type"]=="revolute" and not r["eligible"] for r in manifest["inventory"]),
        skipped_revolute=[r for r in records if r["gt_type"]=="revolute" and r.get("skipped")],
        scoreable_revolute_pairs=len(cases), scoreable_scenes=len({c["scene_index"] for c in cases}),
        objects=sorted({c["object_id"] for c in cases}), pair_indices=[c["pair_index"] for c in cases],
        mesh_sources=fingerprints, evaluation=str(evaluation.resolve()), protocol=protocol)
    write(output/"cohort.json", cohort)
    return cases, cohort


def worker(job):
    case, config, control = job
    data = replace(case["data"], uv=case["gt_uv"]) if control else case["data"]
    start = time.perf_counter()
    try:
        result = fit(data, config)
        row = score(case, result)
        row["rmse_px"] = result.get("rmse_px") if result is not None else None
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        row = score(case, None)
        row["exception"] = repr(error)
    row.update(fit_seconds=time.perf_counter()-start, trajectory_metrics=case["trajectory_metrics"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    protocol = freeze_protocol()
    if args.freeze_only:
        print(json.dumps(protocol), flush=True); return
    cases, cohort = load_validation(args.evaluation, args.output)
    print(json.dumps(dict(event="loaded", pairs=len(cases), scenes=cohort["scoreable_scenes"], objects=len(cohort["objects"]))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        for control in (False, True):
            for name, config in CONFIGS.items():
                target = args.output/("controls" if control else "variants")/f"{name}.json"
                if target.exists():
                    existing = json.loads(target.read_text())
                    assert existing["config"] == config and existing["algorithm_sources"] == frozen_sources()
                    continue
                started = time.perf_counter(); rows = []
                futures = [executor.submit(worker, (case, config, control)) for case in cases]
                for future in as_completed(futures):
                    rows.append(future.result())
                    if len(rows)%100 == 0:
                        print(json.dumps(dict(event="progress", variant=name, control=control, pairs=len(rows))), flush=True)
                rows.sort(key=lambda row:row["pair_index"])
                result = dict(variant=name, config=config, control=control, algorithm_sources=frozen_sources(),
                    summary=summary(rows), rows=rows, wall_seconds=time.perf_counter()-started,
                    cpu_fit_seconds=sum(row["fit_seconds"] for row in rows))
                write(target, result)
                print(json.dumps(dict(event="result", variant=name, control=control, summary=result["summary"])), flush=True)
    assert frozen_sources() == protocol["algorithm_sources"]


if __name__ == "__main__": main()
