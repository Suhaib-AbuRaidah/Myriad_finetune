"""Summarize every development ablation, including failures and fixed cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .experiment import ROOT, write, load_cases
from training_from_scratch.full_evaluation.metrics import distribution
from finetuning.joint_parame_estim.run import modeled_points, project, from_normalized

AXIS="forced_revolute_axis_error_deg"
PIVOT="forced_revolute_kinematic_pivot_distance"


def paired(before,after,key):
    old={r["pair_index"]:r for r in before}
    entries=[(r["scene_index"],float(old[r["pair_index"]][key]),float(r[key])) for r in after
             if r.get(key) is not None and old[r["pair_index"]].get(key) is not None]
    scenes=sorted({s for s,_,_ in entries})
    sums=np.array([sum(b-a for s,a,b in entries if s==scene) for scene in scenes])
    counts=np.array([sum(s==scene for s,_,_ in entries) for scene in scenes])
    draws=np.random.default_rng(42).integers(0,len(scenes),(2000,len(scenes)))
    means=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(n=len(entries),scenes=len(scenes),mean_difference=float(sums.sum()/counts.sum()),
                scene_bootstrap_95_interval=np.percentile(means,[2.5,97.5]).tolist(),
                improved_fraction=float(np.mean([b<a for _,a,b in entries])))


def reprojection_analysis(cases,results):
    output={}
    for variant,result in results.items():
        by_index={r["pair_index"]:r for r in result["rows"]}
        errors=[]
        for case in cases:
            row=by_index[case["original"]["pair_index"]]
            if not row.get("fit_available"):
                continue
            kind=row["predicted_type"]
            parameters=np.array(row[f"fit_parameters_{kind}"])
            identity=[(np.eye(3),np.zeros(3),True)]*3
            points=modeled_points(kind,parameters,case["xyz"],identity)
            s=case["source"];k,e=s["camera_intrinsic"],s["camera_extrinsic"]
            pixels=np.stack([project(p,k,e) for p in points])
            h,w=s["initial_depth"].shape
            gt=from_normalized(case["ground_truth"],h,w,int(s["image_size"]))[2:]
            valid=case["valid"][2:]
            ade=float(np.linalg.norm(pixels-gt,axis=-1)[valid].mean()*int(s["image_size"])/min(h,w))
            errors.append(ade)
        output[variant]=distribution(errors)
    return output


def figures(out,results,quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination=out/"figures";destination.mkdir(exist_ok=True)
    variants=["baseline","refined","contact_restart","track_consensus","shape_mdl","ensemble_weighted"]
    labels=["Original","More converged","Connection prior","Track consensus","Shape candidates","Rollout weights"]
    x=np.arange(len(variants))
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    axes[0].bar(x,[results[v]["summary"]["metrics"][AXIS]["mean"] for v in variants],color="#397da8")
    axes[0].set(ylabel="Mean axis error (degrees)",ylim=(0,90),title="Orientation remains difficult")
    for offset,key,label in ((-.23,"mean","Mean"),(0,"median","Median"),(.23,"p90","90th percentile")):
        axes[1].bar(x+offset,[results[v]["summary"]["metrics"][PIVOT][key] for v in variants],width=.23,label=label)
    axes[1].set(ylabel="Pivot-to-axis distance (dataset world units)",yscale="log",title="Connection evidence reduces remote axes")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x,labels,rotation=25,ha="right",fontsize=8)
        axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
    fig.suptitle("Same 204 revolute joints in 80 development scenes; hypothesis 0")
    for extension in ("png","pdf"):fig.savefig(destination/f"comparison.{extension}",dpi=160)
    plt.close(fig)
    if quality:
        fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained")
        for key,label in (("track_instability","Leave-one-track-out stability"),
                          ("local_uncertainty","Local Jacobian stability"),
                          ("fit_residual","Lowest 2D fitting residual")):
            entries=quality[key]["risk_coverage"]
            ax.plot([100*p["coverage"] for p in entries],[p["mean_axis_error"] for p in entries],"o-",label=label)
        ax.axhline(results["contact_restart"]["summary"]["metrics"][AXIS]["mean"],ls="--",color="gray",label="All 204 joints")
        ax.set(xlabel="Retained fraction of revolute candidates",ylabel="Mean axis error (degrees)",ylim=(0,90),
               title="Stability helps rank estimates, but does not establish correctness")
        ax.set_xticks([25,50,75,100],["25% (51)","50% (102)","75% (153)","100% (204)"])
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc="lower right")
        for extension in ("png","pdf"):fig.savefig(destination/f"stability.{extension}",dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args();out=args.output
    subset=json.loads((out/"subset.json").read_text())
    evaluation=Path(subset["evaluation"])
    results={p.stem:json.loads(p.read_text()) for p in sorted((out/"without_poke").glob("*.json"))}
    baseline=results["baseline"]
    comparison={key:{metric:paired(baseline["rows"],value["rows"],metric)
                     for metric in (AXIS,PIVOT,"rotation_geodesic_deg")}
                for key,value in results.items()}
    write(out/"paired_comparisons.json",comparison)
    cases=load_cases(evaluation,set(subset["scenes"]),"without_poke")
    projections=reprojection_analysis(cases,results)
    write(out/"fitted_projection_errors.json",projections)
    quality={}
    stable=results.get("track_stability")
    if stable:
        rows=[r for r in stable["rows"] if r.get(AXIS) is not None]
        fields={"track_instability":lambda r:r["diagnostics"]["axis_track_instability_deg"],
                "local_uncertainty":lambda r:r["diagnostics"]["refinement"]["axis_local_uncertainty_deg"],
                "fit_residual":lambda r:r["fit_rmse_revolute"]}
        for name,function in fields.items():
            values=np.array([function(r) for r in rows]);truth=np.array([r[AXIS] for r in rows])
            order=np.argsort(values)
            points=[]
            for fraction in (.25,.5,.75,1.):
                selected=[rows[i] for i in order[:max(1,round(fraction*len(rows)))]]
                old={r["pair_index"]:r for r in baseline["rows"]}
                points.append(dict(coverage=fraction,n=len(selected),mean_axis_error=float(np.mean([r[AXIS] for r in selected])),
                    matched_baseline_axis_error=float(np.mean([old[r["pair_index"]][AXIS] for r in selected]))))
            quality[name]=dict(spearman_with_axis_error=float(spearmanr(values,truth).statistic),risk_coverage=points)
        accepted=[r for r in rows if fields["track_instability"](r)<=15]
        quality["track_stability_at_15_deg"]=dict(n=len(accepted),total=len(rows),
            coverage=len(accepted)/len(rows),axis_error=distribution(r[AXIS] for r in accepted),
            matched_baseline_axis=distribution({r["pair_index"]:r for r in baseline["rows"]}[r["pair_index"]][AXIS] for r in accepted))
    write(out/"quality_diagnostics.json",quality)
    figures(out,results,quality)
    inventory=json.loads((evaluation/"run.json").read_text())
    selected=set(subset["scenes"])
    counts=dict(scenes=len(selected),source_joints=sum(r["scene_index"] in selected for r in inventory["inventory"]),
                eligible_pairs=sum(pair[0] in selected for pair in inventory["pairs"]),scored_pairs=len(cases))
    write(out/"coverage.json",counts)
    lines=["# Joint-fitting development experiments", "",
        f"Frozen **80 scenes**, seed {subset['seed']}; {counts['source_joints']} source joints, "
        f"{counts['eligible_pairs']} initially eligible pairs, **{counts['scored_pairs']} scoreable pairs** "
        "(204 revolute, 12 fixed). Thirteen source joints have no initially eligible child and three eligible pairs have no scoreable future GT. "
        "The other 240 scenes were not evaluated with candidate algorithms.","",
        "Predictions are from the existing step-108000 checkpoint. No network weights were changed, no new inference ran, "
        "and the production fitter was not modified. These repeatedly inspected test1 scenes now serve as development data; "
        "the results below are not an independent test of generalization.","",
        "## Main conclusion", "",
        "A soft, uncertainty-scaled axis-line preference near the initial parent–child connection substantially improves pivot distance. "
        "None of the tested conditioning, geometry, or ensemble rules produces convincing overall axis-direction recovery from image-only predictions. "
        "Keep `contact_restart` as the simplest candidate for later held-out evaluation, not as a validated production replacement. "
        "Its joint-type choices are deliberately identical to the original fitter, so improvements cannot come from relabeling difficult joints.","",
        "Both original and candidate misclassify all 12 fixed joints in these image-only predictions. Their 85.65% overall type accuracy "
        "therefore corresponds to only 45.34% balanced accuracy; joint-type recognition remains unresolved.","",
        "![Common-cohort axis and pivot comparison](figures/comparison.png)","",
        "## All ablations", "",
        "Axis and pivot columns evaluate the revolute candidate on the **same 204 GT-revolute pairs**, including pairs classified as another type. "
        "Type accuracy and rotation errors include all 216 scored pairs; no ablation had a fit failure. Pivot distances are dataset world units. "
        "Success requires correct type, axis error ≤15°, and pivot distance ≤0.1, divided by all 204 true revolute joints.","",
        "| Variant | Axis mean ° | Pivot mean | Pivot median | Rotation mean ° | Type accuracy % | Balanced % | Success % | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order=["baseline","all_states","track_filter","moving_tracks","bic","refined","track_consensus", "contact_prior","contact_restart", "contact_tracks","contact_all_states","contact_type", "shape_mdl","shape_only","ensemble_physics","ensemble_medoid","early_contact","ensemble_weighted","track_stability","pnp_global"]
    for name in order:
        if name not in results:continue
        r=results[name];s=r["summary"];m=s["metrics"]
        seconds=r["seconds"]
        if name=="ensemble_medoid":seconds+=results["ensemble_physics"]["seconds"]
        lines.append(f"| {name} | {m[AXIS]['mean']:.2f} | {m[PIVOT]['mean']:.4f} | {m[PIVOT]['median']:.4f} | "
                     f"{m['rotation_geodesic_deg']['mean']:.2f} | {100*s['type_accuracy']:.2f} | {100*s['balanced_accuracy']:.2f} | "
                     f"{100*s['joint_success']['axis_le_15_pivot_le_0.1']:.2f} | {seconds:.1f} |")
    lines.extend(["", "Timing includes geometry scoring. Ensemble medoid timing includes building the five-fit pool; "
        "its recorded raw timing is approximately one second only because the preceding physics-selection run cached that pool. "
        "Ensemble selection changes the input hypothesis; all other variants retain hypothesis 0.","",
        "## Why each round was tested", "",
        "1. **Observation conditioning:** include state 1; reject persistent residual outliers; retain strongly moving tracks; "
        "or use leave-one-track-out consensus. Motivation: isolate drift/outliers while retaining spatial support. "
        "Outcome: no reliable axis gain; removing low-motion points can remove useful near-axis constraints.","",
        "2. **Interpretation and numerical control:** BIC-like type penalties and a converged, centered/scaled minimal axis-line fit. "
        "Motivation: complexity can explain noise, and gauge freedoms impair conditioning. Outcome: BIC lowers overall type accuracy; "
        "better numerical convergence alone can increase remote-pivot degeneracy.","",
        "3. **Physical connection evidence:** nearest visible parent/child surfaces define a soft contact region. "
        "Its covariance includes half the smaller part's visible RMS extent and the surface gap, because the true hinge can be inside an occluded part. "
        "The line is penalized by Mahalanobis distance to that region, scaled by estimated image residual noise. "
        "A second initialization puts the pivot near that region. Outcome: much better pivot tails and median, without a clear axis-direction gain.","",
        "4. **Geometry and rollout hypotheses:** well-separated principal directions in initial visible geometry supply axis candidates; "
        "a model-description-length criterion can retain the free-axis fit. A stronger forced-shape ablation checks the mechanical-alignment assumption. "
        "Five-rollout physical-residual selection and fitted-joint medoid selection use no GT errors. Outcome: forced shape creates many severe axis errors; "
        "ensemble selection does not solve orientation. Higher type accuracy from the medoid does not mean fixed-joint recognition improved.","",
        "5. **Horizon and uncertainty:** early states 1–2 set the axis/pivot, with only later angles refitted; "
        "an alternative weights observations by spread across saved rollouts. Finally, leave-one-track-out perturbations and the "
        "profiled image Jacobian assess stability. These test forecast drift and observability, not a search over arbitrary thresholds.","",
        "6. **Alternative pose initialization:** SQPnP supplies reference-world to camera poses without the original near-reference iterative "
        "pose guess. Its shared-axis candidate and the original candidate compete using the same predicted reprojection/contact objective. "
        "This tests whether the initial pose solver, rather than trajectory conditioning, traps the reconstruction.","",
        "## Paired candidate comparison", "",
        "Differences are candidate minus original; negative means improvement. Confidence intervals resample whole scenes 2,000 times, "
        "preserving correlation between joints. They describe this development comparison and do not correct for trying multiple algorithms.","",
        "| Metric | Mean difference | Scene-bootstrap 95% interval | Improved pairs % |",
        "|---|---:|---|---:|"])
    for key in (AXIS,PIVOT,"rotation_geodesic_deg"):
        item=comparison["contact_restart"][key];a,b=item["scene_bootstrap_95_interval"]
        lines.append(f"| {key} | {item['mean_difference']:.5f} | [{a:.5f}, {b:.5f}] | {100*item['improved_fraction']:.1f} |")
    lines.extend(["", "## Per-joint candidate comparison", "",
        "| Joint ordinal | Revolute n | Original axis ° | Candidate axis ° | Original pivot | Candidate pivot |",
        "|---|---:|---:|---:|---:|---:|"])
    for joint,s in baseline["per_joint"].items():
        a=s["metrics"];b=results["contact_restart"]["per_joint"][joint]["metrics"]
        lines.append(f"| {joint} | {a[AXIS]['n']} | {a[AXIS]['mean']:.2f} | {b[AXIS]['mean']:.2f} | {a[PIVOT]['mean']:.4f} | {b[PIVOT]['mean']:.4f} |")
    lines.extend(["", "## Controls", "",
        "| Inputs | Fitter | Forced axis mean ° | Pivot mean | Rotation mean ° | Type accuracy % |",
        "|---|---|---:|---:|---:|---:|"])
    for directory,label in ((out/"control_"/"without_poke","Ground-truth tracks"),(out/"with_poke","GT first-motion hint + Myriad")):
        for name in ("baseline","contact_restart","shape_mdl"):
            path=directory/f"{name}.json"
            if not path.exists():continue
            s=json.loads(path.read_text())["summary"];m=s["metrics"]
            lines.append(f"| {label} | {name} | {m[AXIS]['mean']:.3f} | {m[PIVOT]['mean']:.6f} | {m['rotation_geodesic_deg']['mean']:.3f} | {100*s['type_accuracy']:.2f} |")
    lines.extend(["", "The exact-input and first-motion-hint controls support a limitation in the image-only predicted motion evidence; "
        "they do not prove no better estimator could exist. Axis means here include all revolute candidates, unlike the original report's "
        "correctly-typed-only geometry means.","", "## Stability and the interpretation of fitted results", ""])
    if quality:
        lines.extend(["![Risk versus retained fraction](figures/stability.png)",""])
        lines.extend(["| Ranking rule, lowest first | Retained n / 204 | Candidate axis mean ° | Original on same joints ° |",
                      "|---|---:|---:|---:|"])
        for key in ("track_instability","local_uncertainty","fit_residual"):
            for point in quality[key]["risk_coverage"]:
                lines.append(f"| {key} | {point['n']} | {point['mean_axis_error']:.2f} | {point['matched_baseline_axis_error']:.2f} |")
        accepted=quality["track_stability_at_15_deg"]
        lines.extend(["",f"A 15° leave-track-out sensitivity threshold retains {accepted['n']}/204 revolute candidates "
            f"({100*accepted['coverage']:.1f}%). Their mean axis error is {accepted['axis_error']['mean']:.2f}°, versus "
            f"{accepted['matched_baseline_axis']['mean']:.2f}° for the original fitter on exactly those joints. "
            "This modest difference does not justify calling the retained axes reliably accurate."])
    lines.extend(["", "Local Jacobian uncertainty assumes an adequate model/noise approximation; leave-track-out variation measures sensitivity, "
        "not correctness. A consistently biased set of predictions can produce a stable but wrong axis. No points or joints were removed from headline "
        "scores based on these diagnostics, and no calibrated confidence probability is claimed.","",
        "## Projection back to GT image positions", "",
        "These ADE values compare the fitted 3D model reprojected at states 2–4 with GT positions, in 384-pixel model-image units. "
        "They are an additional denoising diagnostic, not the fitting objective or a replacement for joint errors.","",
        f"Unfitted hypothesis-0 trajectories have {np.mean([case['original']['common_ADE_pixels'] for case in cases]):.3f} px ADE "
        "on these same states and pairs. A more physically constrained result need not reduce its GT image error.","",
        "| Variant | Fitted-projection ADE px |", "|---|---:|"])
    for name in ("baseline","refined","contact_restart","shape_mdl","ensemble_weighted","ensemble_physics","ensemble_medoid"):
        if name in projections:lines.append(f"| {name} | {projections[name]['mean']:.3f} |")
    lines.extend(["", "## Reproduction and limits", "",
        "`subset.json` freezes scene indices before candidate scoring; per-variant JSON retains every pair, parameters, scores, timing, "
        "diagnostics and per-joint summaries. Later runs also retain exact Python source snapshots by hash under `code/`. "
        "Earlier round source hashes cover the conditioner; the current code preserves those variant rules. "
        "`paired_comparisons.json`, `quality_diagnostics.json`, and `fitted_projection_errors.json` retain machine-readable analyses.","",
        "Fitting uses initial RGB-D/camera/pair labels, predicted tracks and the existing evaluation's future GT geometric-validity masks. "
        "True joint types, axes, pivots and transforms enter scoring only; no candidate is selected by its GT error. "
        "The contact prior uses initial segmented visible geometry, not a true joint location. Pivot scoring keeps the established FK-consistent "
        "reference and raw annotation distances. There are no GT-prismatic examples here; prismatic behavior is not validated by this dataset.","",
        "Stop this development loop here: multiple independently motivated changes did not improve orientation, while contact information "
        "consistently addressed remote-pivot ambiguity. Keep the candidate isolated. A later frozen comparison on the remaining scenes is needed "
        "before replacement; this is a practical stopping point, not a claim of reaching a global optimum.", ""])
    (out/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(variants=len(results),coverage=counts,candidate=comparison["contact_restart"],quality=quality),indent=2))


if __name__=="__main__":main()
