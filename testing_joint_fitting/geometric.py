"""Initial RGB-D contact evidence and joint refinement; no GT motion inputs."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from finetuning.joint_parame_estim.run import unit, project


def surface_points(depth, segmentation, label, k, e, maximum=1800):
    pixels = np.argwhere((segmentation == label) & np.isfinite(depth) & (depth > 0))
    if len(pixels) > maximum:
        pixels = pixels[np.linspace(0,len(pixels)-1,maximum,dtype=int)]
    if not len(pixels):
        return np.empty((0,3))
    y,x = pixels.T
    rays = np.c_[x,y,np.ones(len(x))] @ np.linalg.inv(k).T
    camera = rays * depth[y,x,None]
    return (camera-e[:3,3]) @ e[:3,:3]


def contact_evidence(context, k, e):
    """A soft connection region from nearest visible parent/child surfaces.

    Its uncertainty includes visible-point extent: the actual hinge may be
    inside the part rather than exactly on a visible surface contact.
    """
    if context is None:
        return None
    parent,child = [surface_points(context["depth"],context["segmentation"],label,k,e)
                    for label in context["pair_labels"]]
    if min(len(parent),len(child)) < 8:
        return None
    distance,index = cKDTree(parent).query(child)
    nearest = np.sort(distance)[:min(32,len(distance))]
    threshold = nearest[-1]
    chosen = distance <= threshold
    midpoints = .5*(child[chosen]+parent[index[chosen]])
    center = np.median(midpoints,axis=0)
    cloud_scale = min(np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1))) for p in (parent,child))
    floor = max(.5*cloud_scale, float(np.median(nearest)), 1e-5)
    covariance = np.cov((midpoints-center).T)+floor**2*np.eye(3)
    eig,axes = np.linalg.eigh(covariance)
    whitening = (axes/np.sqrt(eig)[None]).T
    return dict(center=center, covariance=covariance, whitening=whitening,
                gap=float(nearest[0]), uncertainty_floor=floor, points=len(midpoints))


def tangent_frame(axis):
    axis = unit(axis)
    other = np.eye(3)[np.argmin(abs(axis))]
    first = unit(np.cross(axis,other))
    return np.stack([first,np.cross(axis,first)],axis=1)


def huber_cost(norm):
    return float(np.where(norm < 8, norm**2, 16*norm-64).sum())


def refine_revolute(initial, xyz, uv, valid, k, e, *, contact=None, contact_start=False,
                    weights=None):
    """Minimal 2-axis + 2-pivot coordinates and one angle per frame.

    Pivot gauge is the plane through the initial point centroid perpendicular
    to the estimated axis. Contact evidence adds uncertainty-scaled residuals,
    not a hard requirement that the hinge lies on a visible surface.
    """
    rows,cols = np.where(valid)
    if len(rows) < 8 or initial is None:
        return initial
    p = np.asarray(initial["params"],float)
    anchor = xyz.mean(0)
    scale = max(float(np.sqrt(np.mean(np.sum((xyz-anchor)**2,axis=1)))),1e-5)
    axis0 = unit(p[:3]); frame0 = tangent_frame(axis0)
    weights = np.ones(valid.shape) if weights is None else np.broadcast_to(np.asarray(weights),valid.shape)

    def unpack(parameters):
        rotation = cv2.Rodrigues(frame0@parameters[:2])[0]
        axis = rotation@axis0
        pivot = anchor+scale*(rotation@frame0)@parameters[2:4]
        return axis,pivot,parameters[4:]

    def transformed(parameters):
        axis,pivot,angles = unpack(parameters)
        relative = xyz-pivot
        cross = np.cross(axis,relative)
        cross2 = np.cross(axis,cross)
        return relative[None]+pivot+np.sin(angles)[:,None,None]*cross+(1-np.cos(angles))[:,None,None]*cross2

    def image_errors(parameters):
        points = transformed(parameters)
        return np.stack([project(x,k,e) for x in points])-uv

    initial_x = np.r_[np.zeros(2),frame0.T@(p[3:6]-anchor)/scale,p[6:]]
    initial_norm = np.linalg.norm(image_errors(initial_x)[rows,cols],axis=1)
    # Median radial error is 1.177*sigma for an isotropic 2D Gaussian.
    sigma = max(.25,float(np.median(initial_norm))/math.sqrt(2*math.log(2)))

    def prior_errors(parameters):
        if contact is None:
            return np.empty(0)
        axis,pivot,_ = unpack(parameters)
        w = contact["whitening"]
        direction = w@axis; offset = w@(pivot-contact["center"])
        closest = offset-direction*(direction@offset)/(direction@direction)
        return sigma*closest

    def errors(parameters):
        values = image_errors(parameters)[rows,cols]*np.sqrt(weights[rows,cols,None])
        points = transformed(parameters)
        depth = (points@e[:3,:3].T+e[:3,3])[rows,cols,2]
        cheirality = np.minimum(depth/scale,0.)*100
        return np.r_[values.ravel(),prior_errors(parameters),cheirality]

    starts = [initial_x]
    if contact is not None and contact_start:
        x = initial_x.copy()
        x[2:4] = frame0.T@(contact["center"]-anchor)/scale
        starts.append(x)
    results=[]
    for start in starts:
        solved = least_squares(errors,start,loss="soft_l1",f_scale=8.,x_scale="jac",
                               max_nfev=100,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        points = transformed(solved.x)
        depth = (points@e[:3,:3].T+e[:3,3])[rows,cols,2]
        if np.isfinite(solved.x).all() and np.all(depth > 0):
            results.append((solved.cost,solved))
    if not results:
        return initial
    solved = min(results,key=lambda x:x[0])[1]
    axis,pivot,angles = unpack(solved.x)
    norms = np.linalg.norm(image_errors(solved.x)[rows,cols],axis=1)
    jac = solved.jac[:2*len(rows)]
    singular = np.linalg.svd(jac,compute_uv=False)
    # Profile out pivot/angle nuisance parameters before assessing axis
    # observability. This is a local stability estimate, not calibrated error.
    axis_jac,nuisance=jac[:,:2],jac[:,2:]
    profiled=axis_jac-nuisance@np.linalg.lstsq(nuisance,axis_jac,rcond=1e-8)[0]
    axis_singular=np.linalg.svd(profiled,compute_uv=False)
    uncertainty=min(90.,math.degrees(sigma/max(float(axis_singular[-1]),1e-12)))
    return dict(params=np.r_[axis,pivot,angles], sse=huber_cost(norms),
                rmse_px=float(np.sqrt(np.mean(norms**2))),median_px=float(np.median(norms)),n=len(norms),
                refinement=dict(evaluations=solved.nfev, converged=solved.success,
                    contact=contact is not None, contact_sigma_px=sigma,
                    image_jacobian_condition=float(singular[0]/max(singular[-1],1e-12)),
                    axis_local_uncertainty_deg=uncertainty,
                    prior_norm=float(np.linalg.norm(prior_errors(solved.x)))))
