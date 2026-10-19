"""Time the unchanged fitter on saved evaluation arrays; never run Myriad.

Run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
Sampling uses object and joint ordinal, never prediction error or fitted scores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from finetuning_pair.analyze_joints import STATES, fit_joint
from finetuning.joint_parame_estim.run import from_normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path(
        "training_from_scratch/runs/test1_best_full_20261017"))
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path(
        "testing_joint_fitting/benchmark_results.json"))
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(
        (args.evaluation / "records").glob("*_without_poke.json"))]
    rows = [r for r in rows if not r.get("skipped")]
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["object_id"], row["joint_ordinal"]].append(row)
    rng = np.random.default_rng(20261018)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < min(args.pairs, len(rows)):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) == min(args.pairs, len(rows)):
                    break
    source_cache = {}

    def inputs(row, mode):
        path = args.evaluation / "predictions" / f"{row['pair_index']:04d}_{mode}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as a:
            xyz, predicted, truth, valid = (a[k] for k in (
                "reference_world", "predictions", "ground_truth", "valid"))
            source_file = str(a["source_file"])
        if source_file not in source_cache:
            with np.load(args.evaluation / source_file, allow_pickle=False) as s:
                source_cache[source_file] = (
                    s["camera_intrinsic"], s["camera_extrinsic"],
                    s["initial_depth"].shape, int(s["image_size"]))
        k, e, (h, w), image_size = source_cache[source_file]
        return xyz, predicted, truth, valid, k, e, h, w, image_size

    first = inputs(selected[0], "without_poke")
    xyz, pred, gt, valid, k, e, h, w, image_size = first
    warmup = time.perf_counter()
    fit_joint(xyz, from_normalized(pred[0], h, w, image_size)[STATES], valid[STATES], k, e)
    warmup = time.perf_counter() - warmup
    timings = []
    start = time.perf_counter()
    for number, row in enumerate(selected, 1):
        for mode in ("without_poke", "with_poke", "ground_truth_control"):
            load_start = time.perf_counter()
            data = inputs(row, "without_poke" if mode == "ground_truth_control" else mode)
            if data is None:
                continue
            xyz, pred, gt, valid, k, e, h, w, image_size = data
            uv = from_normalized(gt if mode == "ground_truth_control" else pred[0], h, w, image_size)[STATES]
            io_seconds = time.perf_counter() - load_start
            fit_start = time.perf_counter()
            fitted = fit_joint(xyz, uv, valid[STATES], k, e)
            timings.append(dict(pair_index=row["pair_index"], scene_index=row["scene_index"],
                object_id=row["object_id"], joint_ordinal=row["joint_ordinal"], mode=mode,
                io_seconds=io_seconds, fit_seconds=time.perf_counter()-fit_start,
                fit_available=fitted is not None))
        if number % 16 == 0:
            print(json.dumps(dict(pairs_complete=number, fits=len(timings),
                elapsed_seconds=time.perf_counter()-start)), flush=True)
    totals = {}
    for mode in ("without_poke", "with_poke", "ground_truth_control"):
        entries = [r for r in timings if r["mode"] == mode]
        fit = np.array([r["fit_seconds"] for r in entries])
        io = np.array([r["io_seconds"] for r in entries])
        totals[mode] = dict(count=len(entries), mean_fit_seconds=float(fit.mean()),
            median_fit_seconds=float(np.median(fit)), p90_fit_seconds=float(np.percentile(fit, 90)),
            mean_io_seconds=float(io.mean()),
            estimated_877_pairs_seconds=float(877*(fit+io).mean()))
    output = dict(evaluation=str(args.evaluation.resolve()),
        sampling="Round-robin object_id/joint_ordinal groups, seeded shuffle; no error-based selection",
        unique_scenes=len({r["scene_index"] for r in selected}), pairs=len(selected),
        warmup_seconds=warmup, wall_seconds=time.perf_counter()-start,
        threads=1, summary=totals, timings=timings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k != "timings"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
