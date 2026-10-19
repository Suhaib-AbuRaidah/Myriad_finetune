"""Independent checks of cohort, axis geometry, perturbation units and inputs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common import load,OUT,summary,perturb
from testing_joint_fitting.experiment import write
from finetuning.joint_parame_estim.run import modeled_points


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    cases=load();lookup={c["pair_index"]:c for c in cases};expected=set(lookup)
    baseline=json.loads((OUT/"variants/original.json").read_text())
    previous=json.loads((OUT.parent.parent/"results/without_poke/baseline.json").read_text())
    old={r["pair_index"]:r for r in previous["rows"]}
    parity=max(abs(r["axis_error_deg"]-old[r["pair_index"]]["forced_revolute_axis_error_deg"]) for r in baseline["rows"])
    assert parity<1e-9,parity
    max_axis_difference=0.;main_files=sorted((OUT/"variants").glob("*.json"));minimum_depth=float("inf")
    for path in main_files:
        result=json.loads(path.read_text());rows=result["rows"]
        assert len(rows)==204 and {r["pair_index"] for r in rows}==expected,path
        assert {r["scene_index"] for r in rows}=={c["scene_index"] for c in cases}
        assert summary(rows)==result["summary"],path
        for row in rows:
            if row["failed"]:continue
            case=lookup[row["pair_index"]];data=case["data"];p=np.asarray(row["params"])
            assert np.isfinite(p).all() and len(p)==6+len(row["states"])
            assert abs(np.linalg.norm(p[:3])-1)<1e-6
            u=p[:3]/np.linalg.norm(p[:3]);v=case["gt_axis"]/np.linalg.norm(case["gt_axis"])
            angle=math.degrees(math.acos(float(np.clip(abs(u@v),0,1))))
            max_axis_difference=max(max_axis_difference,abs(angle-row["axis_error_deg"]))
            assert abs(angle-row["axis_error_deg"])<1e-7
            rotations=[]
            for slot,state in enumerate((2,3,4)):
                rotation=cv2.Rodrigues(u*p[6+row["states"].index(state)])[0]
                # Stored FK rotations originate in float32. Project the error
                # matrix onto SO(3) before the independent trace-angle check;
                # otherwise a tiny orthogonality error dominates small angles.
                left,_,right=np.linalg.svd(rotation@case["gt_rotations"][slot].T)
                proper=left@np.diag([1.,1.,np.linalg.det(left@right)])@right
                trace=np.trace(proper)
                rotations.append(math.degrees(math.acos(float(np.clip((trace-1)/2,-1,1)))))
            assert abs(np.mean(rotations)-row["rotation_error_deg"])<1e-5,(path,row["pair_index"],rotations,row["rotation_error_deg"])
            if case["gt_pivot"] is not None:
                displacement=case["gt_pivot"]-p[3:6]
                perpendicular=displacement-u*(u@displacement)
                assert abs(np.linalg.norm(perpendicular)-row["pivot_distance"])<1e-9
            if path.stem=="frame_length_contact":
                points=modeled_points("revolute",p,data.xyz,[(np.eye(3),np.zeros(3),True)]*len(row["states"]))
                depth=(points@data.e[:3,:3].T+data.e[:3,3])[:,:,2][data.valid[row["states"]]]
                minimum_depth=min(minimum_depth,float(depth.min()))
    assert minimum_depth>0
    sensitivity_files=sorted((OUT/"sensitivity").glob("*/*/*.json"))
    counts={};sensitivity_failures=0
    for path in sensitivity_files:
        result=json.loads(path.read_text());rows=result["rows"]
        assert len(rows)==204 and {r["pair_index"] for r in rows}==expected,path
        assert summary(rows)==result["summary"],path
        sensitivity_failures+=sum(r["failed"] for r in rows)
        for row in rows:
            assert row["variant"]==result["variant"]
            if row.get("axis_drift_deg") is not None:assert 0<=row["axis_drift_deg"]<=90.000001
        counts[result["variant"]]=counts.get(result["variant"],0)+1
    assert len(sensitivity_files)==94,counts
    for case in cases:
        data=case["data"]
        np.testing.assert_array_equal(data.uv[0],case["gt_uv"][0])
        for pattern in ("common_x","common_y","per_track","jitter"):
            altered=perturb(data,pattern,1.,20261019+case["pair_index"])
            np.testing.assert_array_equal(altered.uv[0],data.uv[0])
            np.testing.assert_allclose(np.linalg.norm(altered.uv[1:]-data.uv[1:],axis=-1)/data.source_per_model_pixel,1.,atol=1e-10)
    subset=json.loads((OUT/"manifest.json").read_text())["subset"]
    evaluation=Path(subset["evaluation"])
    hashes=json.loads((OUT.parent.parent/"results/inputs_sha256.json").read_text())
    for name,record in hashes.items():
        path=evaluation/name
        assert path.stat().st_size==record["size"] and digest(path)==record["sha256"],name
    history=OUT/"variants_history/frame_length_contact_round4.json"
    history_difference=None
    if history.exists():
        before=json.loads(history.read_text())["rows"]
        after=json.loads((OUT/"variants/frame_length_contact.json").read_text())["rows"]
        history_difference=max(float(np.max(abs(np.asarray(a["params"])-np.asarray(b["params"])))) for a,b in zip(before,after))
        assert history_difference<1e-12
    result=dict(passed=True,scenes=80,pairs=204,main_variants=len(main_files),sensitivity_conditions=len(sensitivity_files),
                sensitivity_fits=len(sensitivity_files)*204,sensitivity_failures=sensitivity_failures,
                maximum_baseline_axis_difference_deg=parity,maximum_independent_axis_difference_deg=max_axis_difference,
                candidate_minimum_valid_camera_depth=minimum_depth,input_files_unchanged=len(hashes),
                final_candidate_parameter_difference=history_difference,
                observed_state_exact_and_preserved=True,perturbation_magnitude_verified_in_384_image_pixels=True)
    write(OUT/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
