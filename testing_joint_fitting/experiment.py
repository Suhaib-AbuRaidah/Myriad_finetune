"""Frozen 80-scene development loop using persisted Myriad predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from finetuning_pair.analyze_joints import geometry_metrics
from finetuning.joint_parame_estim.run import from_normalized
from training_from_scratch.full_evaluation.metrics import clean, summarize, attach_reference, kinematic_reference
from .conditioning import VARIANTS, fit_conditioned

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def freeze_subset(evaluation, output):
    target = output / "subset.json"
    if target.exists():
        return json.loads(target.read_text())
    groups = defaultdict(dict)
    for path in sorted((evaluation / "records").glob("*_without_poke.json")):
        row = json.loads(path.read_text())
        groups[row["object_id"]][row["scene_index"]] = True
    rng = np.random.default_rng(20261018)
    pools = {key:list(sorted(value)) for key,value in sorted(groups.items())}
    for value in pools.values():
        rng.shuffle(value)
    selected = []
    while len(selected) < 80:
        for value in pools.values():
            if value:
                selected.append(value.pop())
                if len(selected) == 80:
                    break
    manifest = json.loads((evaluation / "run.json").read_text())
    result = dict(seed=20261018, scenes=sorted(selected), count=80,
                  sampling="Round-robin object groups; seeded scene shuffle; no score-based selection",
                  evaluation=str(evaluation.resolve()), checkpoint_sha256=manifest["checkpoint_sha256"],
                  held_out_scenes=sorted(set(range(len(manifest["scenes"]))) - set(selected)))
    write(target, result)
    return result


def load_cases(evaluation, scenes, mode):
    source_cache = {}
    cases = []
    for path in sorted((evaluation / "records").glob(f"*_{mode}.json")):
        original = json.loads(path.read_text())
        if original["scene_index"] not in scenes or original.get("skipped"):
            continue
        predictions = evaluation / "predictions" / f"{original['pair_index']:04d}_{mode}.npz"
        with np.load(predictions, allow_pickle=False) as a:
            xyz, pred, valid, gt, labels = (a[key] for key in (
                "reference_world", "predictions", "valid", "ground_truth", "pair_labels"))
            source_file = str(a["source_file"])
            state_times = a["state_times"]
        if source_file not in source_cache:
            with np.load(evaluation / source_file, allow_pickle=False) as s:
                source_cache[source_file] = {key:s[key] for key in s.files}
        source = source_cache[source_file]
        h,w = source["initial_depth"].shape
        uv = from_normalized(pred[0], h,w,int(source["image_size"]))
        uv_hypotheses=from_normalized(pred,h,w,int(source["image_size"]))
        jp = int(np.flatnonzero(source["joint_indices"] == original["joint_index"])[0])
        lookup = {int(label):node for node,label in enumerate(source["link_indices"])}
        parent, child = (lookup[int(label)] for label in labels)
        cases.append(dict(original=original, xyz=xyz, uv=uv, valid=valid, source=source,
                          joint_position=jp, parent=parent, child=child, labels=labels,
                          state_times=state_times, ground_truth=gt))
        cases[-1]["context"]=dict(depth=source["initial_depth"],segmentation=source["initial_segmentation"],
            pair_labels=labels,state_times=state_times,hypotheses_uv=uv_hypotheses)
    return cases


def short(summary):
    metrics = summary["metrics"]
    return dict(pairs=summary["scored"], failures=summary["fit_failures"],
                accuracy=summary["type_accuracy"], balanced=summary["balanced_accuracy"],
                axis=metrics["forced_revolute_axis_error_deg"]["mean"],
                pivot=metrics["forced_revolute_kinematic_pivot_distance"]["mean"],
                rotation=metrics["rotation_geodesic_deg"]["mean"],
                success=summary["joint_success"]["axis_le_15_pivot_le_0.1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path(
        "training_from_scratch/runs/test1_best_full_20261017"))
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--variants", nargs="+", default=["baseline"])
    parser.add_argument("--mode", choices=("without_poke", "with_poke"), default="without_poke")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    subset = freeze_subset(args.evaluation, args.output)
    cases = load_cases(args.evaluation, set(subset["scenes"]), args.mode)
    print(json.dumps(dict(event="loaded", scenes=80, pairs=len(cases), mode=args.mode)), flush=True)
    if args.control:
        for case in cases:
            h,w = case["source"]["initial_depth"].shape
            case["uv"] = from_normalized(case["ground_truth"],h,w,int(case["source"]["image_size"]))
            case["context"]["hypotheses_uv"]=case["uv"][None]
    sources={p.name:p.read_text() for p in sorted(ROOT.glob("*.py"))}
    code_hash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    for name,content in sources.items():
        snapshot=args.output/"code"/code_hash/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text(content)
    for variant in args.variants:
        rows=[]
        started = time.perf_counter()
        for index,case in enumerate(cases,1):
            original, source = case["original"], case["source"]
            start = time.perf_counter()
            # Whitelist context: the estimator never sees the future GT tracks,
            # transforms, true joint kind, axis, or origin used by the scorer.
            context = case["context"]
            result = fit_conditioned(case["xyz"],case["uv"],case["valid"],
                source["camera_intrinsic"],source["camera_extrinsic"],variant=variant,context=context)
            elapsed = time.perf_counter()-start
            row = {key:original[key] for key in ("pair_index", "scene_index", "joint_ordinal", "object_id", "mode")}
            row.update(geometry_metrics(result,source,case["joint_position"],case["parent"],case["child"]))
            row.update(skipped=False, improves_stationary=False, fit_seconds=elapsed)
            if row["gt_type"] == "revolute":
                attach_reference(row,kinematic_reference(source,case["joint_position"],case["parent"],case["child"]))
            if result is not None:
                row["diagnostics"] = result.get("diagnostics",{})
            rows.append(clean(row))
            if index % 50 == 0:
                print(json.dumps(dict(event="progress",variant=variant, pairs=index,
                                      seconds=time.perf_counter()-started)),flush=True)
        summary = summarize(rows)
        result = dict(variant=variant, mode=args.mode, control=args.control,
                      seconds=time.perf_counter()-started, summary=summary,
                      per_joint={str(j):summarize([r for r in rows if r["joint_ordinal"]==j])
                                 for j in sorted({r["joint_ordinal"] for r in rows})},
                      code_sha256=code_hash,rows=rows)
        destination=args.output/("control_" if args.control else "")/args.mode/f"{variant}.json"
        write(destination,result)
        print(json.dumps(dict(event="result",variant=variant,seconds=result["seconds"],**short(summary))),flush=True)


if __name__ == "__main__":
    main()
