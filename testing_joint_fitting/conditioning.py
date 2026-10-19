"""Experimental fitting rules. No true joint parameters enter this module."""
from __future__ import annotations

import copy
import math

import numpy as np

from finetuning_pair.analyze_joints import fit_joint
from finetuning.joint_parame_estim.run import modeled_points, project, select_type
from .geometric import contact_evidence, refine_revolute
from .selection import select_shape_fit, contact_cost, angles_only, global_joint_seed


VARIANTS = ("baseline", "all_states", "track_filter", "moving_tracks", "track_consensus", "bic",
            "refined", "contact_prior", "contact_restart", "contact_tracks", "contact_all_states", "contact_type",
            "shape_mdl", "shape_only", "ensemble_physics", "ensemble_medoid",
            "early_contact", "ensemble_weighted", "track_stability", "pnp_global")


def track_stat(values, valid, reducer=np.median):
    return np.array([reducer(values[valid[:,i],i]) if valid[:,i].any() else np.nan
                     for i in range(values.shape[1])])


def residuals(kind, fit, xyz, uv, k, e):
    identity = [(np.eye(3), np.zeros(3), True)] * len(uv)
    points = modeled_points(kind, fit["params"], xyz, identity)
    return np.stack([project(p, k, e) for p in points]) - uv


def rescore(result, xyz, uv, valid, k, e):
    """Score each candidate on the SAME original observations after filtering."""
    result = copy.deepcopy(result)
    for kind, fit in result["fits"].items():
        if fit is None:
            continue
        norm = np.linalg.norm(residuals(kind, fit, xyz, uv, k, e), axis=-1)[valid]
        fit.update(sse=float(np.where(norm < 8, norm**2, 16*norm-64).sum()),
                   rmse_px=float(np.sqrt(np.mean(norm**2))), median_px=float(np.median(norm)), n=len(norm))
    result["kind"], result["scores"] = select_type(result["fits"], result["motion"])
    return result


def fit_conditioned(xyz, full_uv, full_valid, k, e, *, variant="baseline", context=None):
    if variant not in VARIANTS and variant != "weighted_internal":
        raise ValueError(variant)
    if variant.startswith("ensemble_"):
        if variant == "ensemble_weighted":
            result=fit_conditioned(xyz,full_uv,full_valid,k,e,variant="weighted_internal",context=context)
            if result is not None:
                result["diagnostics"]["variant"]=variant
            return result
        hypotheses=context["hypotheses_uv"]
        if "ensemble_results" not in context:
            context["ensemble_results"]=[fit_conditioned(xyz,hypothesis,full_valid,k,e,
                variant="contact_restart",context=context) for hypothesis in hypotheses]
        options=[(i,result) for i,result in enumerate(context["ensemble_results"]) if result is not None]
        if not options:
            return None
        contact=contact_evidence(context,k,e)
        scores=[]
        for index,result in options:
            fit=result["fits"][result["kind"]]
            sigma=max(.25,fit["median_px"]/math.sqrt(2*math.log(2)))
            score=fit["sse"]/max(1,fit["n"])
            if result["kind"]=="revolute":
                score+=contact_cost(fit,contact,sigma)/max(1,fit["n"])
            scores.append(score)
        selected=int(np.argmin(scores))
        if variant=="ensemble_medoid":
            available=[j for j,(_,r) in enumerate(options) if r["kind"]=="revolute"]
            if len(available)>=2:
                scale=max(float(np.linalg.norm(xyz-xyz.mean(0),axis=1).mean()),1e-5)
                center=contact["center"] if contact is not None else xyz.mean(0)
                distances=np.zeros((len(available),len(available)))
                for a,ia in enumerate(available):
                    pa=options[ia][1]["fits"]["revolute"]["params"]
                    ua=pa[:3]/np.linalg.norm(pa[:3]);qa=pa[3:6]+ua*(ua@(center-pa[3:6]))
                    for b,ib in enumerate(available):
                        pb=options[ib][1]["fits"]["revolute"]["params"]
                        ub=pb[:3]/np.linalg.norm(pb[:3]);qb=pb[3:6]+ub*(ub@(center-pb[3:6]))
                        distances[a,b]=math.acos(np.clip(abs(ua@ub),0,1))/(math.pi/2)+np.linalg.norm(qa-qb)/scale
                selected=available[int(np.argmin(np.median(distances,axis=1)))]
        index,result=options[selected]
        result=copy.deepcopy(result)
        result["diagnostics"].update(variant=variant,hypothesis_index=index,hypothesis_scores=scores)
        return result
    states = np.array([1, 2, 3, 4]) if variant in ("all_states", "contact_all_states") else np.array([2, 3, 4])
    uv, valid = full_uv[states], full_valid[states].copy()
    base = fit_joint(xyz, uv, valid, k, e)
    if base is None:
        return None
    diag = dict(variant=variant, fitting_states=states, tracks_total=len(xyz), tracks_kept=len(xyz))
    if variant in ("track_filter", "moving_tracks"):
        if variant == "track_filter":
            fit = base["fits"]["revolute"]
            if fit is not None:
                r = np.linalg.norm(residuals("revolute", fit, xyz, uv, k, e), axis=-1)
                scores = track_stat(r,valid)
                center = np.nanmedian(scores)
                scale = 1.4826*np.nanmedian(abs(scores-center))
                keep = np.isfinite(scores) & (scores <= center+max(3., 2.5*scale))
            else:
                keep = np.ones(len(xyz), bool)
        else:
            initial = project(xyz, k, e)
            motion = track_stat(np.linalg.norm(uv-initial, axis=-1),valid)
            keep = motion >= max(2., .1*np.nanmedian(motion))
        # Keep at least four distinct points and most of their spatial support.
        spread = np.linalg.norm(xyz-xyz.mean(0), axis=-1).mean()
        if keep.sum() >= 4 and np.linalg.norm(xyz[keep]-xyz[keep].mean(0), axis=-1).mean() >= .5*spread:
            candidate = fit_joint(xyz, uv, valid & keep[None], k, e)
            if candidate is not None:
                base = rescore(candidate, xyz, uv, valid, k, e)
                diag.update(tracks_kept=int(keep.sum()), kept_indices=np.flatnonzero(keep))
    elif variant == "track_consensus":
        # Leave-one-track-out candidates; rank by a common least-trimmed track
        # cost. Selection uses predicted positions only, never true geometry.
        candidates = [base]
        for point in range(len(xyz)):
            mask = valid.copy(); mask[:, point] = False
            if (mask.sum(0) > 0).sum() < 4:
                continue
            fit = fit_joint(xyz, uv, mask, k, e)
            if fit is not None:
                candidates.append(fit)
        chosen = copy.deepcopy(base)
        for kind in chosen["fits"]:
            options = []
            for candidate in candidates:
                fit = candidate["fits"][kind]
                if fit is None:
                    continue
                norm = np.linalg.norm(residuals(kind, fit, xyz, uv, k, e), axis=-1)
                per_track = track_stat(np.minimum(norm**2, 16*norm),valid,np.mean)
                finite = np.sort(per_track[np.isfinite(per_track)])
                count = max(4, math.ceil(.75*len(finite)))
                options.append((float(np.mean(finite[:count])), fit))
            if options:
                chosen["fits"][kind] = min(options, key=lambda x:x[0])[1]
        base = rescore(chosen, xyz, uv, valid, k, e)
        diag["initializations"] = len(candidates)
    elif variant == "bic":
        n = int(2*valid.sum())
        dof = dict(fixed=0, prismatic=2+len(states), revolute=4+len(states))
        scores = {kind: n*math.log(1+fit["sse"]/n)+dof[kind]*math.log(n)
                  for kind, fit in base["fits"].items() if fit is not None}
        if base["motion"] >= 3 and base["fits"]["fixed"]["median_px"] >= 3:
            base["kind"] = min(scores, key=scores.get)
        base["scores"] = scores
    elif variant == "early_contact":
        contact=contact_evidence(context,k,e)
        early=fit_joint(xyz,full_uv[1:3],full_valid[1:3],k,e)
        if early is not None and early["fits"]["revolute"] is not None:
            fit=refine_revolute(early["fits"]["revolute"],xyz,full_uv[1:3],full_valid[1:3],k,e,
                                contact=contact,contact_start=True)
            base["fits"]["revolute"]=angles_only(fit,xyz,uv,valid,k,e)
            original_kind=base["kind"]
            base=rescore(base,xyz,uv,valid,k,e);base["kind"]=original_kind
            diag["axis_fitting_states"]=[1,2]
    elif variant in ("refined", "contact_prior", "contact_restart", "contact_tracks", "contact_all_states", "contact_type", "shape_mdl", "shape_only", "weighted_internal", "track_stability", "pnp_global"):
        original_kind = base["kind"]
        contact = contact_evidence(context,k,e) if variant != "refined" else None
        initial = base["fits"]["revolute"]
        weights=None
        if variant=="weighted_internal":
            variance=np.var(context["hypotheses_uv"][:,states],axis=0).sum(-1)
            weights=64./(64.+variance)
            diag["observation_weights"]=weights
        fit = refine_revolute(initial,xyz,uv,valid,k,e,contact=contact,weights=weights,
                              contact_start=variant in ("contact_restart", "contact_tracks", "shape_mdl", "shape_only", "weighted_internal", "track_stability", "pnp_global"))
        if variant=="pnp_global" and fit is not None:
            seed=global_joint_seed(xyz,uv,valid,k,e)
            if seed is not None:
                alternative=refine_revolute(seed,xyz,uv,valid,k,e,contact=contact,contact_start=True)
                sigma=max(.25,initial["median_px"]/math.sqrt(2*math.log(2)))
                def score(f):return f["sse"]+contact_cost(f,contact,sigma)
                diag["global_initialization_selected"]=score(alternative)<score(fit)
                if score(alternative)<score(fit):fit=alternative
        if variant=="track_stability" and fit is not None:
            axes=[]
            for point in range(len(xyz)):
                mask=valid.copy();mask[:,point]=False
                if (mask.sum(0)>0).sum()<4:
                    continue
                seeded=fit_joint(xyz,uv,mask,k,e)
                if seeded is None or seeded["fits"]["revolute"] is None:
                    continue
                trial=refine_revolute(seeded["fits"]["revolute"],xyz,uv,mask,k,e,contact=contact)
                if trial is not None:
                    axes.append(trial["params"][:3])
            errors=[math.degrees(math.acos(np.clip(abs(np.dot(a,fit["params"][:3])),0,1))) for a in axes]
            diag["leave_track_out_axis_changes_deg"]=errors
            diag["axis_track_instability_deg"]=float(np.percentile(errors,75)) if errors else 90.
        if variant in ("shape_mdl", "shape_only") and fit is not None:
            fit,selection=select_shape_fit(fit,initial,xyz,uv,valid,k,e,contact,context,
                                           always=variant=="shape_only")
            diag["shape_selection"]=selection
        if variant == "contact_tracks" and fit is not None:
            norm = np.linalg.norm(residuals("revolute",fit,xyz,uv,k,e),axis=-1)
            per_track = track_stat(norm,valid)
            weights = np.nan_to_num(np.minimum(1.,8./np.maximum(per_track,1e-6)),nan=0.)
            fit = refine_revolute(fit,xyz,uv,valid,k,e,contact=contact,weights=weights)
            diag["track_weights"] = weights
        base["fits"]["revolute"] = fit
        base = rescore(base,xyz,uv,valid,k,e)
        if variant != "contact_type":
            # Hold type decisions fixed for the geometry ablation.
            base["kind"] = original_kind
        if contact is not None:
            diag["contact"] = {key:value for key,value in contact.items() if key != "whitening"}
        if fit is not None:
            diag["refinement"] = fit.get("refinement",{})
    # The scorer always compares states 2--4, even if earlier evidence was used.
    if variant in ("all_states", "contact_all_states"):
        for kind, fit in base["fits"].items():
            if fit is not None and kind != "fixed":
                fit["params"] = np.r_[fit["params"][:6 if kind == "revolute" else 3], fit["params"][-3:]]
    base["diagnostics"] = diag
    return base
