"""Geometry-derived axis hypotheses and complexity-aware interpretation."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.optimize import least_squares

from finetuning.joint_parame_estim.run import project, unit
from .geometric import surface_points, tangent_frame, huber_cost
from finetuning.joint_parame_estim.run import initial_axis, fit_model


def shape_axes(context,k,e):
    clouds=[surface_points(context["depth"],context["segmentation"],label,k,e)
            for label in context["pair_labels"]]
    result=[]
    for label,cloud in zip(("parent","child"),clouds):
        if len(cloud)<8:
            continue
        eigen,basis=np.linalg.eigh(np.cov(cloud.T))
        for index,axis in enumerate(basis.T):
            # An eigendirection of nearly equal eigenvalues is arbitrary.
            gaps=[abs(eigen[index]-eigen[j]) for j in range(3) if j != index]
            if min(gaps)<.1*eigen[-1]:
                continue
            if not any(abs(axis@u)>.99 for _,u in result):
                result.append((f"{label}_pca_{index}",axis))
    return result


def contact_cost(fit,contact,sigma):
    if contact is None:
        return 0.
    axis=unit(fit["params"][:3]);pivot=fit["params"][3:6]
    w=contact["whitening"]
    direction=w@axis;offset=w@(pivot-contact["center"])
    perpendicular=offset-direction*(offset@direction)/(direction@direction)
    return float(sigma**2*(perpendicular@perpendicular))


def fixed_axis_fit(axis,seed,xyz,uv,valid,k,e,contact,sigma):
    axis=unit(axis);frame=tangent_frame(axis);anchor=xyz.mean(0)
    scale=max(float(np.linalg.norm(xyz-anchor,axis=1).mean()),1e-5)
    rows,cols=np.where(valid)
    pivot=contact["center"] if contact is not None else seed["params"][3:6]

    def unpack(x):
        return anchor+scale*frame@x[:2],x[2:]

    def residual(x):
        pivot,angles=unpack(x)
        relative=xyz-pivot;cross=np.cross(axis,relative);cross2=np.cross(axis,cross)
        points=xyz[None]+np.sin(angles)[:,None,None]*cross+(1-np.cos(angles))[:,None,None]*cross2
        image=np.stack([project(p,k,e) for p in points])
        return image-uv,points

    # Independent 1D angular searches avoid an axis-direction sign ambiguity
    # or a zero-angle initialization trapping a candidate at no motion.
    angles=[]
    for state in range(len(uv)):
        options=[]
        for angle in np.linspace(-math.pi,math.pi,25):
            r=cv2.Rodrigues(axis*angle)[0]
            error=project((xyz-pivot)@r.T+pivot,k,e)-uv[state]
            options.append(huber_cost(np.linalg.norm(error[valid[state]],axis=1)))
        angles.append(np.linspace(-math.pi,math.pi,25)[np.argmin(options)])
    initial=np.r_[frame.T@(pivot-anchor)/scale,angles]

    def errors(x):
        image,points=residual(x)
        pivot,angles=unpack(x)
        prior=[]
        if contact is not None:
            direction=contact["whitening"]@axis
            offset=contact["whitening"]@(pivot-contact["center"])
            prior=sigma*(offset-direction*(offset@direction)/(direction@direction))
        depth=(points@e[:3,:3].T+e[:3,3])[rows,cols,2]
        return np.r_[image[rows,cols].ravel(),prior,np.minimum(depth/scale,0.)*100]
    solved=least_squares(errors,initial,loss="soft_l1",f_scale=8.,x_scale="jac",
                          max_nfev=70,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    image,points=residual(solved.x)
    if np.any((points@e[:3,:3].T+e[:3,3])[rows,cols,2]<=0):
        return None
    pivot,angles=unpack(solved.x)
    norms=np.linalg.norm(image[rows,cols],axis=1)
    return dict(params=np.r_[axis,pivot,angles],sse=huber_cost(norms),
                rmse_px=float(np.sqrt(np.mean(norms**2))),median_px=float(np.median(norms)),n=len(norms),
                refinement=dict(evaluations=solved.nfev,converged=solved.success,axis_fixed=True))


def select_shape_fit(free,seed,xyz,uv,valid,k,e,contact,context,*,always=False):
    axes=shape_axes(context,k,e)
    if not axes:
        return free,dict(selected="free",candidates=0)
    sigma=max(.25,float(seed["median_px"])/math.sqrt(2*math.log(2)))
    n=2*int(valid.sum()); frames=len(uv)
    def criterion(fit,dof,search):
        return (fit["sse"]+contact_cost(fit,contact,sigma))/sigma**2+dof*math.log(n)+2*math.log(search)
    choices=[] if always else [(criterion(free,4+frames,1),"free",free)]
    for label,axis in axes:
        candidate=fixed_axis_fit(axis,seed,xyz,uv,valid,k,e,contact,sigma)
        if candidate is not None:
            choices.append((criterion(candidate,2+frames,len(axes)),label,candidate))
    if not choices:
        return free,dict(selected="free",candidates=0)
    score,label,result=min(choices,key=lambda x:x[0])
    return result,dict(selected=label,candidates=len(choices),criterion=score,
                       all_scores={label:score for score,label,_ in choices})


def angles_only(fit,xyz,uv,valid,k,e):
    """Estimate later rotation amounts without changing an early fitted axis."""
    axis=unit(fit["params"][:3]);pivot=fit["params"][3:6]
    relative=xyz-pivot;cross=np.cross(axis,relative);cross2=np.cross(axis,cross)
    def residual(angles):
        points=xyz[None]+np.sin(angles)[:,None,None]*cross+(1-np.cos(angles))[:,None,None]*cross2
        return np.stack([project(p,k,e) for p in points])-uv
    grid=np.linspace(-math.pi,math.pi,49)
    seed=[]
    for state in range(len(uv)):
        costs=[]
        for a in grid:
            point=xyz+math.sin(a)*cross+(1-math.cos(a))*cross2
            norms=np.linalg.norm((project(point,k,e)-uv[state])[valid[state]],axis=1)
            costs.append(huber_cost(norms))
        seed.append(grid[np.argmin(costs)])
    solved=least_squares(lambda a:residual(a)[valid].ravel(),seed,loss="soft_l1",f_scale=8.,max_nfev=50)
    return dict(fit,params=np.r_[axis,pivot,solved.x])


def global_joint_seed(xyz,uv,valid,k,e):
    """SQPnP alternative to near-reference iterative pose initialization."""
    poses=[]
    for state in range(len(uv)):
        mask=valid[state]&np.isfinite(uv[state]).all(1)
        options=[]
        if mask.sum()>=4:
            try:
                output=cv2.solvePnPGeneric(np.ascontiguousarray(xyz[mask],dtype=np.float64),
                    np.ascontiguousarray(uv[state,mask],dtype=np.float64),k,None,flags=cv2.SOLVEPNP_SQPNP)
                for rv,tv in zip(output[1],output[2]):
                    camera_rotation=cv2.Rodrigues(rv)[0];camera_translation=tv.ravel()
                    camera=xyz[mask]@camera_rotation.T+camera_translation
                    if np.any(camera[:,2]<=0):continue
                    rotation=e[:3,:3].T@camera_rotation
                    translation=e[:3,:3].T@(camera_translation-e[:3,3])
                    norm=np.linalg.norm(project(xyz[mask]@rotation.T+translation,k,e)-uv[state,mask],axis=1)
                    options.append((huber_cost(norm),rotation,translation))
            except cv2.error:
                pass
        if options:
            _,rotation,translation=min(options,key=lambda item:item[0])
            poses.append((rotation,translation,True))
        else:
            poses.append((np.eye(3),np.zeros(3),False))
    if not any(p[2] for p in poses):return None
    identity=[(np.eye(3),np.zeros(3),True)]*len(uv)
    axis,pivot,angles=initial_axis(poses,identity,xyz.mean(0))
    return fit_model("revolute",xyz,uv,valid,identity,k,e,np.r_[axis,pivot,angles])
