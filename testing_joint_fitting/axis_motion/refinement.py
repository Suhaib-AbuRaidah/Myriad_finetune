"""Controlled shift-aware objectives in minimal axis/pivot coordinates."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.optimize import least_squares

from finetuning.joint_parame_estim.run import project
from testing_joint_fitting.geometric import tangent_frame,huber_cost
from .algorithms import seed_fit,noise_sigma,quality_weights

VARIANTS=("full_absolute","full_length","increments","hybrid_increments","bias_shared",
          "bias_track_1px","bias_track_2px","bias_track_4px","bias_track_relative",
          "bias_shared_relative","length_bias","frame_bias","two_way_bias",
          "spatial_differences","contact_tight","contact_fixed_noise","contact_fixed",
          "frame_length","frame_contact","frame_length_contact","frame_covariance",
          "frame_covariance_length","frame_covariance_consistent","geometry_weight",
          "frame_geometry","frame_length_geometry","two_way_conditioned")


def profile_offset(error,valid,weights,penalty,shared=False):
    """Exact quadratic profiling of a constant future offset with Gaussian prior.

    penalty = white_noise_variance / offset_prior_variance. Offsets are shared
    across future times, never allowed to change independently at each frame.
    The returned augmented residual preserves the quadratic penalized objective.
    """
    w=weights*valid
    if shared:
        offset=(error*w[...,None]).sum((0,1))/(w.sum()+penalty)
        adjusted=error-offset
        prior=math.sqrt(penalty)*offset
    else:
        offset=(error*w[...,None]).sum(0)/(w.sum(0)+penalty)[:,None]
        adjusted=error-offset[None]
        prior=math.sqrt(penalty)*offset
    return adjusted,prior,offset


def fit_variant(data,variant,*,initial=None,contact_multiplier=1.):
    if variant not in VARIANTS:raise ValueError(variant)
    seed=seed_fit(data) if initial is None else initial
    if seed is None:return None
    xyz,uv,valid=data.xyz,data.uv[1:],data.valid[1:]
    rows,cols=np.where(valid)
    if len(rows)<8:return None
    anchor=xyz.mean(0);scale=max(float(np.linalg.norm(xyz-anchor,axis=1).mean()),1e-5)
    axis0=seed["params"][:3]/np.linalg.norm(seed["params"][:3]);frame=tangent_frame(axis0)
    theta=np.r_[.5*seed["params"][6],seed["params"][6:]]
    if len(theta)!=4:raise ValueError("Initializer must contain states 2--4")
    x0=np.r_[0.,0.,frame.T@(seed["params"][3:6]-anchor)/scale,theta]
    sigma=noise_sigma(seed)
    if variant in ("contact_tight","frame_contact","frame_length_contact","frame_geometry","frame_length_geometry","two_way_conditioned"):contact_multiplier=4.
    if variant=="contact_fixed_noise":contact_multiplier=8./sigma
    fixed_pivot=variant=="contact_fixed" and data.contact is not None
    if fixed_pivot:x0=np.r_[x0[:2],x0[4:]]
    use_length=variant in ("full_length","length_bias","frame_length","frame_length_contact","frame_covariance_length","frame_covariance_consistent")
    weights=np.broadcast_to(quality_weights(data,"length_square") if use_length else np.ones(len(xyz)),valid.shape)
    if variant in ("frame_length","frame_length_contact","frame_covariance_length","frame_covariance_consistent","two_way_conditioned"):
        weights=np.broadcast_to(quality_weights(data,"length_soft"),valid.shape)
    if variant in ("geometry_weight","frame_geometry","frame_length_geometry") and data.contact is not None:
        from .flow import rotation_jacobian
        sensitivity=np.sum(rotation_jacobian(xyz,data.contact["center"],data.k,data.e)**2,axis=(1,2))
        geometric=np.clip(sensitivity/max(float(np.median(sensitivity)),1e-8),.05,4.)
        if variant=="frame_length_geometry":geometric*=quality_weights(data,"length_soft")
        geometric*=len(geometric)/geometric.sum()
        weights=np.broadcast_to(geometric,valid.shape)
    if variant=="frame_covariance_consistent" and data.uv_hypotheses is not None:
        motion=data.uv_hypotheses[:,1:]-data.uv_hypotheses[:,:1]
        centered=motion-motion.mean(2,keepdims=True)
        variance=np.var(centered,axis=0).sum(-1)
        weights=weights/(1+variance/64.)
        weights=weights/weights[valid].mean()
    penalty=None
    shared=variant in ("bias_shared","bias_shared_relative")
    if variant.startswith("bias_") or variant=="length_bias":
        if variant.endswith("relative"):penalty=1.
        else:
            size={"bias_track_1px":1.,"bias_track_4px":4.}.get(variant,2.)
            penalty=(sigma/(size*data.source_per_model_pixel))**2
    # Gaussian nuisance offsets shared by each frame, or by both track/frame.
    # Ridge priors keep them from freely removing actual projected motion.
    nuisance=None
    if variant in ("frame_bias","two_way_bias","frame_length","frame_contact","frame_length_contact","frame_geometry","frame_length_geometry","two_way_conditioned"):
        two_way=variant in ("two_way_bias","two_way_conditioned")
        columns=len(uv)+(len(xyz) if two_way else 0)
        design=np.zeros((len(rows),columns))
        design[np.arange(len(rows)),rows]=1.
        if two_way:design[np.arange(len(rows)),len(uv)+cols]=1.
        ridge=(sigma/(2.*data.source_per_model_pixel))**2
        obs_weights=weights[valid]
        inverse=np.linalg.solve(design.T@(obs_weights[:,None]*design)+ridge*np.eye(columns),design.T*obs_weights)
        nuisance=(design,inverse,math.sqrt(ridge)*np.eye(columns))
    if variant.startswith("frame_covariance"):
        ensemble=data.uv_hypotheses if data.uv_hypotheses is not None else data.uv[None]
        displacements=ensemble[:,1:]-ensemble[:,:1]
        means=displacements.mean(2).reshape(len(ensemble),-1)
        covariance=np.cov(means,rowvar=False) if len(ensemble)>1 else np.zeros((8,8))
        covariance=covariance+(.5*data.source_per_model_pixel)**2*np.eye(8)
        ridge=sigma**2*np.linalg.inv(covariance)
        design=np.zeros((2*len(rows),8))
        design[2*np.arange(len(rows)),2*rows]=1.
        design[2*np.arange(len(rows))+1,2*rows+1]=1.
        obs_weights=np.repeat(weights[valid],2)
        inverse=np.linalg.solve(design.T@(obs_weights[:,None]*design)+ridge,design.T*obs_weights)
        eigen,basis=np.linalg.eigh(ridge)
        prior=(basis*np.sqrt(np.maximum(eigen,0))[None]).T
        nuisance=(design,inverse,prior)

    def unpack(x):
        rotation=cv2.Rodrigues(frame@x[:2])[0]
        if fixed_pivot:return rotation@axis0,data.contact["center"],x[2:]
        return rotation@axis0,anchor+scale*(rotation@frame)@x[2:4],x[4:]

    def prediction(x):
        axis,pivot,angles=unpack(x)
        relative=xyz-pivot;cross=np.cross(axis,relative);cross2=np.cross(axis,cross)
        points=xyz[None]+np.sin(angles)[:,None,None]*cross+(1-np.cos(angles))[:,None,None]*cross2
        return np.stack([project(p,data.k,data.e) for p in points]),points

    def contact_residual(x):
        if data.contact is None:return np.zeros(0)
        axis,pivot,_=unpack(x)
        w=data.contact["whitening"]
        direction=w@axis;offset=w@(pivot-data.contact["center"])
        return contact_multiplier*sigma*(offset-direction*(offset@direction)/(direction@direction))

    def residual(x):
        pixels,points=prediction(x);error=pixels-uv
        if nuisance is not None:
            design,inverse,prior=nuisance
            if variant.startswith("frame_covariance"):
                flat=error[valid].ravel();offsets=inverse@flat
                values=np.r_[(flat-design@offsets)*np.sqrt(np.repeat(weights[valid],2)),prior@offsets]
            else:
                offsets=inverse@error[valid]
                values=np.r_[((error[valid]-design@offsets)*np.sqrt(weights[valid,None])).ravel(),(prior@offsets).ravel()]
        elif variant=="spatial_differences":
            count=valid.sum(1)
            mean=(error*valid[...,None]).sum(1)/np.maximum(count[:,None],1)
            values=np.r_[(error-mean[:,None])[valid].ravel(),.5*error[valid].ravel()]
        elif penalty is not None:
            adjusted,prior,_=profile_offset(error,valid,weights,penalty,shared)
            values=np.r_[(adjusted*np.sqrt(weights[...,None]))[valid].ravel(),prior.ravel()]
        elif variant in ("increments","hybrid_increments"):
            # Deliberately omit the state-0->1 edge: a future-only constant shift
            # cancels in later increments, but corrupts that first jump.
            mask=valid[1:]&valid[:-1]
            delta=np.diff(pixels,axis=0)-np.diff(uv,axis=0)
            values=delta[mask].ravel()/math.sqrt(2)
            if variant=="hybrid_increments":values=np.r_[values,.5*error[valid].ravel()]
        else:values=(error*np.sqrt(weights[...,None]))[valid].ravel()
        depth=(points@data.e[:3,:3].T+data.e[:3,3])[rows,cols,2]
        return np.r_[values,contact_residual(x),np.minimum(depth/scale,0.)*100]

    starts=[x0]
    if data.contact is not None and not fixed_pivot:
        alternative=x0.copy();alternative[2:4]=frame.T@(data.contact["center"]-anchor)/scale
        starts.append(alternative)
    choices=[]
    for start in starts:
        solved=least_squares(residual,start,loss="soft_l1",f_scale=8.,x_scale="jac",max_nfev=110,
                              ftol=1e-6,xtol=1e-6,gtol=1e-6)
        pixels,points=prediction(solved.x)
        depth=(points@data.e[:3,:3].T+data.e[:3,3])[rows,cols,2]
        if np.isfinite(solved.x).all() and np.all(depth>0):choices.append(solved)
    if not choices:return dict(seed,states=[2,3,4],diagnostics=dict(variant=variant,fallback=True))
    solved=min(choices,key=lambda s:s.cost)
    axis,pivot,angles=unpack(solved.x);pixels,_=prediction(solved.x)
    norms=np.linalg.norm((pixels-uv)[valid],axis=-1)
    diagnostics=dict(variant=variant,evaluations=solved.nfev,converged=solved.success,objective=float(2*solved.cost),
                     fitting_states=[1,2,3,4],weights=weights[0],white_sigma_source_pixels=sigma,
                     offset_penalty=penalty,contact_multiplier=contact_multiplier)
    if penalty is not None:
        _,_,offset=profile_offset(pixels-uv,valid,weights,penalty,shared)
        diagnostics["estimated_offsets_model_pixels"]=offset/data.source_per_model_pixel
    if nuisance is not None:
        design,inverse,prior=nuisance
        if variant.startswith("frame_covariance"):
            offsets=(inverse@(pixels-uv)[valid].ravel()).reshape(4,2)
        else:offsets=inverse@(pixels-uv)[valid]
        diagnostics["estimated_nuisance_corrections_model_pixels"]=offsets/data.source_per_model_pixel
    return dict(params=np.r_[axis,pivot,angles],states=[1,2,3,4],sse=huber_cost(norms),
                rmse_px=float(np.sqrt(np.mean(norms**2))),median_px=float(np.median(norms)),n=len(norms),
                diagnostics=diagnostics)
