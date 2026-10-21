"""CPU-only frozen-cohort experiments; save each completed variant immediately."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing as mp
import time

from testing_joint_fitting.axis_motion.common import score, summary
from testing_joint_fitting.experiment import write
from .data import load, OUT, snapshot
from .solver import fit, VARIANTS


def worker(job):
    case, config, control = job
    data = replace(case["data"], uv=case["gt_uv"]) if control else case["data"]
    started = time.perf_counter()
    fitted = fit(data, config)
    row = score(case, fitted)
    if fitted is not None: row["rmse_px"] = fitted.get("rmse_px")
    row["fit_seconds"] = time.perf_counter()-started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["original", "rgbd_position", "mesh_position"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    cases = load(); code = snapshot()
    print(json.dumps(dict(event="loaded", scenes=80, pairs=len(cases), code=code)), flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        for variant in args.variants:
            config = VARIANTS[variant]
            target = OUT / ("controls" if args.control else "variants") / f"{variant}.json"
            if target.exists():
                old = json.loads(target.read_text())
                assert old["config"] == config
                print(json.dumps(dict(event="existing", variant=variant, summary=old["summary"])), flush=True)
                continue
            rows = []; begin = time.perf_counter()
            futures = [executor.submit(worker, (case, config, args.control)) for case in cases]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows)%51 == 0:
                    print(json.dumps(dict(event="progress", variant=variant, pairs=len(rows), seconds=time.perf_counter()-begin)), flush=True)
            rows.sort(key=lambda row: row["pair_index"])
            result = dict(variant=variant, config=config, control=args.control, code_sha256=code,
                wall_seconds=time.perf_counter()-begin, cpu_fit_seconds=sum(r["fit_seconds"] for r in rows),
                summary=summary(rows), rows=rows)
            write(target, result)
            print(json.dumps(dict(event="result", variant=variant, wall_seconds=result["wall_seconds"], summary=result["summary"])), flush=True)
    print(json.dumps(dict(event="complete", wall_seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__": main()
