"""Prediction-only axis estimators; no scoring truth is accepted here."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from finetuning_pair.analyze_joints import fit_joint
from finetuning.joint_parame_estim.run import modeled_points,project
from testing_joint_fitting.geometric import refine_revolute,huber_cost
from testing_joint_fitting.selection import contact_cost
from .common import lengths

VARIANTS=("original","contact","length_soft","length_square","extent_square",
          "longest4","longest6","spatial4","length_seed")


def seed_fit(data,valid=None):
    valid=data.valid[2:] if valid is None else valid
    result=fit_joint(data.xyz,data.uv[2:],valid,data.k,data.e)
    return result["fits"]["revolute"] if result is not None else None


def noise_sigma(seed):
    return max(.25,float(seed["median_px"])/math.sqrt(2*math.log(2)))


def quality_weights(data,variant):
    motion=lengths(data.uv)
    path=motion["extent"] if variant=="extent_square" else motion["path"]
    if variant=="length_soft":weights=path**2/(path**2+64.)
    else:weights=np.clip((path/max(float(np.median(path)),1e-6))**2,.05,4.)
    weights=np.maximum(weights,1e-6)
    return weights*len(weights)/weights.sum()


def subset(data,count,*,spatial=False):
    n=len(data.xyz)
    if n<=count:return np.ones(n,bool)
    quality=quality_weights(data,"length_square")
    if spatial:
        centered=data.xyz-data.xyz.mean(0)
        eigen,basis=np.linalg.eigh(centered.T@centered/n)
        whitening=basis/np.sqrt(np.maximum(eigen,.001*eigen[-1]))[None]
        design=np.c_[np.ones(n),centered@whitening]
        candidates=[]
        for indices in combinations(range(n),count):
            indices=np.array(indices)
            matrix=design[indices].T@(quality[indices,None]*design[indices])+.05*np.eye(4)
            candidates.append((np.linalg.slogdet(matrix)[1],indices))
        chosen=max(candidates,key=lambda x:x[0])[1]
    else:
        chosen=np.argsort(lengths(data.uv)["path"])[-count:]
    mask=np.zeros(n,bool);mask[chosen]=True
    # A sparse selection that cannot initialize any future pose should not
    # silently turn an otherwise solvable example into an initialization failure.
    if np.any((data.valid[2:]&mask).sum(1)<4):return np.ones(n,bool)
    return mask


def errors(data,fit):
    identity=[(np.eye(3),np.zeros(3),True)]*3
    points=modeled_points("revolute",fit["params"],data.xyz,identity)
    return np.stack([project(p,data.k,data.e) for p in points])-data.uv[2:]


def refine(data,seed,*,weights=None,valid=None,sigma=None):
    if seed is None:return None
    valid=data.valid[2:] if valid is None else valid
    # Keep the physical prior strength fixed across alternative initializations.
    contact=data.contact
    if sigma is not None and contact is not None:
        contact=dict(contact,whitening=contact["whitening"]*sigma/noise_sigma(seed))
    result=refine_revolute(seed,data.xyz,data.uv[2:],valid,data.k,data.e,
                           contact=contact,contact_start=True,weights=weights)
    if result is not None:result=dict(result,states=[2,3,4])
    return result


def fit(data,variant):
    if variant not in VARIANTS:
        if variant.startswith("flow_"):
            from .flow import fit_flow
            return fit_flow(data,variant)
        from .refinement import fit_variant
        return fit_variant(data,variant)
    seed=seed_fit(data)
    if seed is None:return None
    if variant=="original":return dict(seed,states=[2,3,4],diagnostics=dict(variant=variant))
    sigma=noise_sigma(seed);weights=None;valid=data.valid[2:];selected=np.ones(len(data.xyz),bool)
    if variant in ("length_soft","length_square","extent_square","length_seed"):
        weights=quality_weights(data,variant)
    if variant in ("longest4","longest6","spatial4"):
        selected=subset(data,6 if variant=="longest6" else 4,spatial=variant=="spatial4")
        valid=valid&selected
        alternative=seed_fit(data,valid)
        if alternative is not None:seed=alternative
    result=refine(data,seed,weights=weights,valid=valid,sigma=sigma)
    if variant=="length_seed":
        mask=data.valid[2:]&subset(data,6)
        alternative=refine(data,seed_fit(data,mask),weights=weights,sigma=sigma)
        def cost(candidate):
            norms=np.linalg.norm(errors(data,candidate),axis=-1)
            per=np.where(norms<8,norms**2,16*norms-64)
            return float((per*np.broadcast_to(weights,per.shape))[data.valid[2:]].sum())+contact_cost(candidate,data.contact,sigma)
        if alternative is not None and cost(alternative)<cost(result):result=alternative
    if result is not None:
        result["diagnostics"]=dict(variant=variant,weights=weights,selected=np.flatnonzero(selected),
                                   refinement=result.get("refinement",{}))
    return result
