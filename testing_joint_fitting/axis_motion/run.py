"""Run fixed-cohort length/shift ablations, with CPU process parallelism."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
import multiprocessing as mp
import time

import numpy as np

from .common import load,OUT,score,summary,snapshot,perturb
from .algorithms import fit
from testing_joint_fitting.experiment import write
from finetuning.joint_parame_estim.run import axis_error


def worker(job):
    case,variant,kind,pattern,amplitude=job
    data=case["data"]
    if kind=="ground_truth":data=replace(data,uv=case["gt_uv"],uv_hypotheses=case["gt_uv"][None])
    if amplitude:
        data=perturb(data,pattern,amplitude,20261019+case["pair_index"])
    started=time.perf_counter()
    fitted=fit(data,variant)
    row=score(case,fitted)
    row.update(variant=variant,input=kind,pattern=pattern,amplitude_model_pixels=amplitude,
               fit_seconds=time.perf_counter()-started)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",nargs="+",default=["original","contact"])
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--sensitivity",action="store_true")
    parser.add_argument("--control",action="store_true")
    parser.add_argument("--patterns",nargs="+",default=["common_x","common_y","per_track","jitter"])
    parser.add_argument("--amplitudes",nargs="+",type=float,default=[.25,.5,1.,2.,4.])
    args=parser.parse_args()
    cases=load();code=snapshot()
    jobs=[]
    kinds=("predicted","ground_truth") if args.sensitivity else (("ground_truth",) if args.control else ("predicted",))
    for kind in kinds:
        for variant in args.variants:
            jobs.extend((case,variant,kind,"none",0.) for case in cases)
            if args.sensitivity:
                for pattern in args.patterns:
                    for amplitude in args.amplitudes:
                        jobs.extend((case,variant,kind,pattern,amplitude) for case in cases)
    print(json.dumps(dict(event="start",pairs=len(cases),jobs=len(jobs),workers=args.workers)),flush=True)
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as executor:
        futures=[executor.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%204==0:
                print(json.dumps(dict(event="progress",complete=len(rows),total=len(jobs),seconds=time.perf_counter()-started)),flush=True)
    groups={}
    for row in rows:
        key=(row["variant"],row["input"],row["pattern"],row["amplitude_model_pixels"])
        groups.setdefault(key,[]).append(row)
    for (variant,kind,pattern,amplitude),entries in sorted(groups.items()):
        entries.sort(key=lambda r:r["pair_index"])
        result=dict(variant=variant,input=kind,pattern=pattern,amplitude_model_pixels=amplitude,
                    code_sha256=code,summary=summary(entries),rows=entries,
                    seconds=sum(r["fit_seconds"] for r in entries))
        if args.sensitivity:
            clean={r["pair_index"]:r for r in groups[variant,kind,"none",0.]}
            changes=[]
            for row in entries:
                old=clean[row["pair_index"]]
                if row.get("params") is not None and old.get("params") is not None:
                    row["axis_drift_deg"]=axis_error(row["params"][:3],old["params"][:3])
                    changes.append(row["axis_drift_deg"])
            from training_from_scratch.full_evaluation.metrics import distribution
            result["axis_drift"]=distribution(changes)
            target=OUT/"sensitivity"/variant/kind/f"{pattern}_{amplitude:g}.json"
        else:
            target=OUT/("controls" if args.control else "variants")/f"{variant}.json"
        write(target,result)
        if pattern=="none":print(json.dumps(dict(event="result",variant=variant,input=kind,**result["summary"])),flush=True)
    print(json.dumps(dict(event="complete",seconds=time.perf_counter()-started,groups=len(groups))),flush=True)


if __name__=="__main__":main()
