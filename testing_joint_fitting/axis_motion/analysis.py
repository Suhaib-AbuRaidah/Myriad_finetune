"""Describe length, direction, spatial conditioning, and known anchoring."""
from __future__ import annotations

import json
import numpy as np
from scipy.stats import spearmanr

from .common import load, OUT, lengths
from testing_joint_fitting.experiment import write
from training_from_scratch.full_evaluation.metrics import distribution


def main():
    cases=load();tracks=[];pairs=[]
    previous=json.loads((OUT.parent.parent/"results/without_poke/baseline.json").read_text())
    old={r["pair_index"]:r for r in previous["rows"]}
    for case in cases:
        data=case["data"];scale=data.source_per_model_pixel
        predicted=lengths(data.uv);truth=lengths(case["gt_uv"])
        complete=data.valid.all(0)
        axis=case["gt_axis"]/np.linalg.norm(case["gt_axis"])
        radius=np.linalg.norm(np.cross(data.xyz-case["gt_pivot"],axis),axis=1) if case["gt_pivot"] is not None else np.full(len(data.xyz),np.nan)
        for i in range(len(data.xyz)):
            if not complete[i]:continue
            pd=(data.uv[:,i]-data.uv[0,i])/scale
            gd=(case["gt_uv"][:,i]-case["gt_uv"][0,i])/scale
            eligible=(np.linalg.norm(pd,axis=-1)>=1)&(np.linalg.norm(gd,axis=-1)>=1)
            dot=(pd*gd).sum(-1)
            cos=dot/np.maximum(np.linalg.norm(pd,axis=-1)*np.linalg.norm(gd,axis=-1),1e-10)
            angles=np.degrees(np.arccos(np.clip(cos[eligible],-1,1)))
            gt_length=truth["path"][i]/scale
            ade=float(np.linalg.norm(data.uv[1:,i]-case["gt_uv"][1:,i],axis=-1).mean()/scale)
            tracks.append(dict(pair_index=case["pair_index"],scene_index=case["scene_index"],track=i,
                predicted_path=predicted["path"][i]/scale,gt_path=gt_length,
                predicted_extent=predicted["extent"][i]/scale,gt_extent=truth["extent"][i]/scale,
                gt_axis_radius_world=float(radius[i]) if np.isfinite(radius[i]) else None,
                gt_axis_radius_fraction=float(radius[i]/max(float(np.nanmax(radius)),1e-10)) if np.isfinite(radius[i]) else None,
                ade_pixels=ade,ade_percent=100*ade/gt_length if gt_length>=1 else None,
                direction_error_deg=float(angles.mean()) if len(angles) else None,
                opposite_fraction=float(np.mean(dot[eligible]<0)) if eligible.any() else None))
        motion_fields={}
        centered_fields={}
        for name,positions in (("predicted",data.uv),("gt",case["gt_uv"])):
            mask=data.valid[1:];movement=(positions[1:]-positions[:1])/scale
            common=(movement*mask[...,None]).sum(1)/np.maximum(mask.sum(1)[:,None],1)
            differential=movement-common[:,None]
            motion_fields[name+"_common_motion_rms"]=float(np.sqrt(np.mean(np.sum(common**2,axis=1))))
            motion_fields[name+"_differential_motion_rms"]=float(np.sqrt(np.mean(np.sum(differential[mask]**2,axis=1))))
            centered_fields[name]=differential
        motion_fields["differential_motion_error_rms"]=float(np.sqrt(np.mean(np.sum((centered_fields["predicted"]-centered_fields["gt"])[mask]**2,axis=1))))
        pairs.append(dict(pair_index=case["pair_index"],scene_index=case["scene_index"],joint_ordinal=case["joint_ordinal"],
            predicted_median_path=float(np.median(predicted["path"])/scale),
            gt_median_path=float(np.median(truth["path"][complete])/scale) if complete.any() else None,
            axis_error_deg=old[case["pair_index"]]["forced_revolute_axis_error_deg"],
            initial_anchor_max_error_pixels=float(np.linalg.norm(data.uv[0]-case["gt_uv"][0],axis=-1).max()/scale),**motion_fields))
    bins=[0,2,5,10,20,40,float("inf")]
    grouped={}
    for field in ("predicted_path","gt_path"):
        grouped[field]=[]
        for a,b in zip(bins[:-1],bins[1:]):
            rows=[r for r in tracks if a<=r[field]<b]
            grouped[field].append(dict(range=[a,b if np.isfinite(b) else None],n=len(rows),
                ade=distribution(r["ade_pixels"] for r in rows),direction=distribution(r["direction_error_deg"] for r in rows),
                opposite_fraction=distribution(r["opposite_fraction"] for r in rows)))
    correlations={}
    for field in ("predicted_median_path","gt_median_path"):
        rows=[r for r in pairs if r[field] is not None]
        correlations[field]=float(spearmanr([r[field] for r in rows],[r["axis_error_deg"] for r in rows]).statistic)
    result=dict(pair_count=len(pairs),complete_tracks=len(tracks),length_unit="384 model pixels",bins=grouped,
        pair_length_axis_spearman=correlations,initial_anchor_max_error_pixels=max(r["initial_anchor_max_error_pixels"] for r in pairs),
        pairs=pairs,tracks=tracks)
    result["motion_field_summary"]={field:distribution(row[field] for row in pairs) for field in motion_fields}
    result["radius_groups"]=[]
    for lower,upper in ((0,.25),(.25,.5),(.5,.75),(.75,1.000001)):
        rows=[r for r in tracks if r["gt_axis_radius_fraction"] is not None and lower<=r["gt_axis_radius_fraction"]<upper]
        result["radius_groups"].append(dict(range=[lower,min(upper,1.)],n=len(rows),
            gt_path=distribution(r["gt_path"] for r in rows),direction=distribution(r["direction_error_deg"] for r in rows)))
    write(OUT/"length_analysis.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("pairs","tracks")},indent=2))


if __name__=="__main__":main()
