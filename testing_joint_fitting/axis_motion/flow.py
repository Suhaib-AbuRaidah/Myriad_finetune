"""Axis initialization from calibrated projected rotation about initial contact."""
from __future__ import annotations

import numpy as np

from finetuning.joint_parame_estim.run import project
from testing_joint_fitting.selection import fixed_axis_fit
from .algorithms import seed_fit,noise_sigma,quality_weights,refine

VARIANTS=("flow_first","flow_long","flow_quadratic","flow_refined","flow_fixed_pivot")


def rotation_jacobian(xyz,pivot,k,e):
    camera=xyz@e[:3,:3].T+e[:3,3]
    pixels=project(xyz,k,e)
    jac=(k[:2][None]-pixels[:,:,None]*np.array([0.,0.,1.])[None,None])/camera[:,2,None,None]
    world=jac@e[:3,:3]
    cross=np.cross(np.eye(3)[None],(xyz-pivot)[:,None])
    return np.einsum("nij,nkj->nik",world,cross)


def flow_axis(data,*,long=False,quadratic=False):
    if data.contact is None:return None
    j=rotation_jacobian(data.xyz,data.contact["center"],data.k,data.e)
    displacement=data.uv[1]-data.uv[0]
    mask=data.valid[0]&data.valid[1]
    if quadratic:
        displacement=(4*data.uv[1]-data.uv[2]-3*data.uv[0])/2
        mask&=data.valid[2]
    if mask.sum()<3:return None
    weights=quality_weights(data,"length_soft") if long else np.ones(len(data.xyz))
    root=np.sqrt(weights[mask,None,None])
    design=(j[mask]*root).reshape(-1,3)
    target=(displacement[mask]*np.sqrt(weights[mask,None])).ravel()
    # SVD rank truncation only removes numerically unconstrained directions.
    omega=np.linalg.lstsq(design,target,rcond=1e-6)[0]
    if np.linalg.norm(omega)<1e-8:return None
    return omega/np.linalg.norm(omega)


def fit_flow(data,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    original=seed_fit(data)
    if original is None:return None
    axis=flow_axis(data,long=variant!="flow_first",quadratic=variant=="flow_quadratic")
    if axis is None:return refine(data,original)
    seed=dict(original,params=np.r_[axis,data.contact["center"],original["params"][6:]])
    fixed=fixed_axis_fit(axis,seed,data.xyz,data.uv[2:],data.valid[2:],data.k,data.e,data.contact,noise_sigma(original))
    if fixed is None:return refine(data,original)
    if variant=="flow_refined":fixed=refine(data,fixed,sigma=noise_sigma(original))
    if variant=="flow_fixed_pivot":
        from .refinement import fit_variant
        fixed=fit_variant(data,"contact_fixed",initial=fixed)
    result=dict(fixed,states=fixed.get("states",[2,3,4]))
    result["diagnostics"]=dict(variant=variant,axis_initialization=axis,linearization="initial projected rotational velocity about visible contact")
    return result
