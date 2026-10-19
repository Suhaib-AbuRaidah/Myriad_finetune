"""Frozen data and scorers; estimators receive FitInput, never Case truth."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from testing_joint_fitting.experiment import load_cases, write
from testing_joint_fitting.geometric import contact_evidence
from training_from_scratch.full_evaluation.metrics import kinematic_reference, distribution
from finetuning.joint_parame_estim.run import from_normalized, axis_error, axis_distance

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"


@dataclass(frozen=True)
class FitInput:
    xyz: np.ndarray
    uv: np.ndarray
    valid: np.ndarray
    k: np.ndarray
    e: np.ndarray
    contact: dict | None
    source_per_model_pixel: float
    uv_hypotheses: np.ndarray | None = None


def load():
    subset=json.loads((ROOT.parent/"results/subset.json").read_text())
    evaluation=Path(subset["evaluation"])
    cases=[]
    for case in load_cases(evaluation,set(subset["scenes"]),"without_poke"):
        # Cohort is deliberately assumed revolute; true kind never selects an
        # algorithm, track, initialization, or per-case output after this step.
        if case["original"]["gt_type"]!="revolute":continue
        s=case["source"];h,w=s["initial_depth"].shape
        data=FitInput(case["xyz"],case["uv"],case["valid"],s["camera_intrinsic"],s["camera_extrinsic"],
                      contact_evidence(case["context"],s["camera_intrinsic"],s["camera_extrinsic"]),
                      min(h,w)/int(s["image_size"]),case["context"]["hypotheses_uv"])
        transforms=s["link_transforms_world"].astype(float)
        rotations=[]
        for state in (2,3,4):
            relative=transforms[0,case["parent"]]@np.linalg.inv(transforms[state,case["parent"]])@transforms[state,case["child"]]@np.linalg.inv(transforms[0,case["child"]])
            rotations.append(relative[:3,:3])
        reference=kinematic_reference(s,case["joint_position"],case["parent"],case["child"])
        cases.append(dict(data=data,pair_index=case["original"]["pair_index"],
                          scene_index=case["original"]["scene_index"],joint_ordinal=case["original"]["joint_ordinal"],
                          object_id=case["original"]["object_id"],
                          gt_uv=from_normalized(case["ground_truth"],h,w,int(s["image_size"])),
                          gt_axis=s["joint_axes_world"][0,case["joint_position"]].astype(float),
                          gt_pivot=np.array(reference["pivot"]) if reference.get("observable") else None,
                          gt_rotations=np.stack(rotations)))
    assert len(cases)==204 and len({c["scene_index"] for c in cases})==80
    manifest=dict(subset=subset,pair_indices=[c["pair_index"] for c in cases],scenes=80,pairs=204,
        assumed_joint_kind="revolute",scoring_states=[2,3,4],primary_hypothesis=0,
        image_resolution=384,perturbation_unit="384-image model pixels",
        estimation_inputs="initial 3D points, predicted 2D tracks, camera, initial contact region, existing correspondence masks",
        truth_use="cohort definition, scoring and explicitly labelled controlled GT-input experiments only")
    target=OUT/"manifest.json"
    if target.exists():assert json.loads(target.read_text())==manifest
    else:write(target,manifest)
    return cases


def score(case,fit):
    row={k:case[k] for k in ("pair_index","scene_index","joint_ordinal","object_id")}
    if fit is None:
        return dict(row,failed=True,axis_error_deg=None,axis_error_with_failure_penalty_deg=90.)
    parameters=np.asarray(fit["params"])
    states=list(fit.get("states",(2,3,4)))
    axis=parameters[:3]/np.linalg.norm(parameters[:3])
    errors=[]
    for slot,state in enumerate((2,3,4)):
        angle=parameters[6+states.index(state)]
        rotation=cv2.Rodrigues(axis*angle)[0]
        errors.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation@case["gt_rotations"][slot].T)[0]))))
    error=axis_error(axis,case["gt_axis"])
    return dict(row,failed=False,axis_error_deg=error,axis_error_with_failure_penalty_deg=error,
                rotation_error_deg=float(np.mean(errors)),
                pivot_distance=axis_distance(parameters[3:6],axis,case["gt_pivot"]) if case["gt_pivot"] is not None else None,
                params=parameters,states=states,diagnostics=fit.get("diagnostics",{}))


def summary(rows):
    return dict(n=len(rows),failures=sum(r["failed"] for r in rows),
        axis=distribution(r.get("axis_error_deg") for r in rows),
        axis_with_failure_penalty=distribution(r["axis_error_with_failure_penalty_deg"] for r in rows),
        rotation=distribution(r.get("rotation_error_deg") for r in rows),
        pivot=distribution(r.get("pivot_distance") for r in rows),
        success_10_deg=sum(not r["failed"] and r["axis_error_deg"]<=10 for r in rows)/max(1,len(rows)),
        success_15_deg=sum(not r["failed"] and r["axis_error_deg"]<=15 for r in rows)/max(1,len(rows)))


def lengths(uv):
    steps=np.diff(uv,axis=0)
    return dict(path=np.linalg.norm(steps,axis=-1).sum(0),
                extent=np.linalg.norm(uv-uv[:1],axis=-1).max(0),
                net=np.linalg.norm(uv[-1]-uv[0],axis=-1))


def snapshot():
    files={str(p.relative_to(ROOT.parent)):p.read_text() for p in sorted(ROOT.parent.glob("*.py"))}
    files.update({"axis_motion/"+p.name:p.read_text() for p in sorted(ROOT.glob("*.py"))})
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    for name,content in files.items():
        target=OUT/"code"/digest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
    return digest


def perturb(data,pattern,amplitude,seed):
    """All shifts affect future states only; observed state 0 remains fixed."""
    uv=data.uv.copy();rng=np.random.default_rng(seed)
    scale=amplitude*data.source_per_model_pixel
    if pattern=="common_x":noise=np.array([1.,0.])
    elif pattern=="common_y":noise=np.array([0.,1.])
    elif pattern=="per_track":
        angles=rng.uniform(-np.pi,np.pi,len(data.xyz));noise=np.c_[np.cos(angles),np.sin(angles)][None]
    elif pattern=="jitter":
        angles=rng.uniform(-np.pi,np.pi,data.uv[1:].shape[:-1]);noise=np.stack([np.cos(angles),np.sin(angles)],-1)
    else:raise ValueError(pattern)
    uv[1:]+=scale*noise
    ensemble=None if data.uv_hypotheses is None else data.uv_hypotheses.copy()
    if ensemble is not None:ensemble[:,1:]+=scale*noise
    return replace(data,uv=uv,uv_hypotheses=ensemble)
