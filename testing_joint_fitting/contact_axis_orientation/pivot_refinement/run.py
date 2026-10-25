"""Paired CPU experiments on the fixed original 80-scene development cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from testing_joint_fitting.axis_motion.common import score, summary
from testing_joint_fitting.experiment import write
from ..data import load, ROOT as ORIGINAL_ROOT
from .solver import fit, VARIANTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"


def snapshot():
    repo = ROOT.parents[2]
    paths = [ROOT/"solver.py", ROOT/"run.py"]+list(ORIGINAL_ROOT.glob("*.py"))
    paths += [repo/name for name in ("testing_joint_fitting/geometric.py",
        "testing_joint_fitting/experiment.py", "testing_joint_fitting/axis_motion/common.py",
        "finetuning_pair/analyze_joints.py", "finetuning/joint_parame_estim/run.py",
        "training_from_scratch/full_evaluation/metrics.py")]
    files = {str(p.relative_to(repo)): p.read_text() for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    for name, content in files.items():
        target = OUT/"code"/digest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return digest


def worker(job):
    case, config, control = job
    data = replace(case["data"], uv=case["gt_uv"]) if control else case["data"]
    start = time.perf_counter()
    fitted = fit(data, config)
    row = score(case, fitted)
    row["rmse_px"] = None if fitted is None else fitted["rmse_px"]
    row["fit_seconds"] = time.perf_counter()-start
    return row


def evaluate(cases, variants, *, control=False, workers=4, cohort="development"):
    code = snapshot()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        for name, config in variants.items():
            target = OUT/cohort/("controls" if control else "variants")/f"{name}.json"
            if target.exists():
                old = json.loads(target.read_text())
                assert old["config"] == config, f"Configuration changed: {target}"
                if old["code_sha256"] == code:
                    print(json.dumps(dict(event="existing", variant=name)), flush=True)
                    continue
                historical = OUT/"historical"/old["code_sha256"]/target.relative_to(OUT)
                historical.parent.mkdir(parents=True, exist_ok=True)
                target.rename(historical)
            start = time.perf_counter()
            rows = [f.result() for f in as_completed([
                executor.submit(worker, (c, config, control)) for c in cases])]
            rows.sort(key=lambda r: r["pair_index"])
            result = dict(variant=name, config=config, control=control, code_sha256=code,
                wall_seconds=time.perf_counter()-start, summary=summary(rows), rows=rows,
                cohort=cohort, scenes=len({c["scene_index"] for c in cases}))
            result["mean_rmse_px"] = float(np.mean([r["rmse_px"] for r in rows if r["rmse_px"] is not None]))
            write(target, result)
            s = result["summary"]
            print(json.dumps(dict(event="result", variant=name, control=control,
                axis=s["axis"]["mean"], pivot=s["pivot"]["mean"], rotation=s["rotation"]["mean"],
                max_axis=s["axis"]["max"], rmse=result["mean_rmse_px"],
                failures=s["failures"], seconds=result["wall_seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cases = load()
    print(json.dumps(dict(event="loaded", pairs=len(cases))), flush=True)
    evaluate(cases, {name: VARIANTS[name] for name in args.variants},
        control=args.control, workers=args.workers)


if __name__ == "__main__":
    main()
