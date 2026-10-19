"""Full fixed-cohort analysis of length conditioning and shift robustness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import OUT,summary
from testing_joint_fitting.experiment import write
from training_from_scratch.full_evaluation.metrics import distribution

CANDIDATE="frame_length_contact"


def paired(before,after,key="axis_error_with_failure_penalty_deg"):
    old={r["pair_index"]:r for r in before}
    differences=[(r["scene_index"],r[key]-old[r["pair_index"]][key]) for r in after
                 if r.get(key) is not None and old[r["pair_index"]].get(key) is not None]
    scenes=sorted({s for s,_ in differences})
    sums=np.array([sum(d for s,d in differences if s==scene) for scene in scenes])
    counts=np.array([sum(s==scene for s,d in differences) for scene in scenes])
    draws=np.random.default_rng(42).integers(0,len(scenes),(2000,len(scenes)))
    means=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(n=len(differences),mean_difference=float(sums.sum()/counts.sum()),
                interval_95=np.percentile(means,[2.5,97.5]).tolist(),
                improved_fraction=float(np.mean([d<0 for _,d in differences])))


def load_sensitivity():
    data={}
    for path in sorted((OUT/"sensitivity").glob("*/*/*.json")):
        value=json.loads(path.read_text())
        data[value["variant"],value["input"],value["pattern"],value["amplitude_model_pixels"]]=value
    return data


def sensitivity_groups(sensitivity,pairs):
    by_pair={r["pair_index"]:r for r in pairs}
    outputs=[]
    for (variant,kind,pattern,amplitude),result in sensitivity.items():
        if amplitude!=1. or pattern!="per_track":continue
        for field in ("gt_median_path","predicted_median_path"):
            bins=[0,2,5,10,20,40,float("inf")]
            for low,high in zip(bins[:-1],bins[1:]):
                rows=[r for r in result["rows"] if by_pair[r["pair_index"]].get(field) is not None and
                      low<=by_pair[r["pair_index"]][field]<high]
                outputs.append(dict(variant=variant,input=kind,field=field,
                    range=[low,high if np.isfinite(high) else None],n=len(rows),
                    axis=distribution(r.get("axis_error_deg") for r in rows),
                    drift=distribution(r.get("axis_drift_deg") for r in rows)))
    return outputs


def figures(variants,length,sensitivity,risk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=OUT/"figures";output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    entries=length["bins"]["predicted_path"]
    x=np.arange(len(entries));labels=[f"{r['range'][0]}–{r['range'][1]}" if r["range"][1] else f"≥{r['range'][0]}" for r in entries]
    axes[0].bar(x,[r["direction"]["mean"] for r in entries],color="#477caa")
    axes[0].set_xticks(x,labels);axes[0].set(xlabel="Predicted point path (model pixels)",ylabel="Mean image-motion direction error (°)",ylim=(0,100),title="Long tracks predict image direction better")
    for i,r in enumerate(entries):axes[0].text(i,r["direction"]["mean"]+2,f"n={r['direction']['n']}",ha="center",fontsize=8)
    values=[variants[n]["summary"]["axis"]["mean"] for n in ("original","longest4","length_soft","frame_bias",CANDIDATE)]
    axes[1].bar(np.arange(5),values,color=["#777777","#c57855","#477caa","#477caa","#3d9970"])
    axes[1].set_xticks(np.arange(5),["Original","Longest 4","Length weights","Frame offsets","Combined"],rotation=20,ha="right")
    axes[1].set(ylabel="Mean axis error (°)",ylim=(0,90),title="Track selection alone does not solve 3D orientation")
    for i,value in enumerate(values):axes[1].text(i,value+1,f"{value:.2f}°",ha="center",fontsize=9)
    for axis in axes:axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
    for ext in ("png","pdf"):fig.savefig(output/f"length_and_axis.{ext}",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    for ax,pattern,title in zip(axes,("common_x","per_track","jitter"),("Shared future x offset","Persistent per-track offsets","Frame-by-frame jitter")):
        for name,label in (("original","Original"),(CANDIDATE,"Combined candidate"),("bias_track_2px","Per-track offset model")):
            values=sorted((amplitude,result["summary"]["axis"]["mean"]) for (variant,kind,p,a),result in sensitivity.items()
                          if variant==name and kind=="ground_truth" and ((p==pattern and a>0) or p=="none")
                          for amplitude in [a])
            if values:ax.plot([x for x,_ in values],[y for _,y in values],"o-",label=label,ms=4)
        ax.set(xlabel="Perturbation magnitude (384-image pixels)",ylabel="Mean axis error (°)",title=title)
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Controlled GT trajectories: a perturbation of 1 pixel gives exactly 1 pixel future ADE")
    for ext in ("png","pdf"):fig.savefig(output/f"shift_sensitivity.{ext}",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),layout="constrained")
    for name,label in (("original","Original"),(CANDIDATE,"Combined candidate")):
        ax.plot([r["threshold_model_pixels"] for r in risk],[r[name]["axis"]["mean"] for r in risk],"o-",label=label)
    ax.set(xlabel="Minimum median predicted path per joint (model pixels)",ylabel="Mean axis error on retained joints (°)",
           title="Long-motion coverage changes the evaluation cohort",ylim=(0,90))
    ax.legend();ax.grid(alpha=.2)
    previous_count=None
    for r in risk:
        if r["n"]!=previous_count:
            ax.annotate(f"{r['n']}/204",(r["threshold_model_pixels"],r[CANDIDATE]["axis"]["mean"]),xytext=(0,-15),textcoords="offset points",ha="center",fontsize=8)
        previous_count=r["n"]
    for ext in ("png","pdf"):fig.savefig(output/f"length_coverage.{ext}",dpi=160)
    plt.close(fig)


def main():
    variants={p.stem:json.loads(p.read_text()) for p in sorted((OUT/"variants").glob("*.json"))}
    controls={p.stem:json.loads(p.read_text()) for p in sorted((OUT/"controls").glob("*.json"))}
    length=json.loads((OUT/"length_analysis.json").read_text())
    sensitivity=load_sensitivity();base=variants["original"];candidate=variants[CANDIDATE]
    comparisons={name:paired(base["rows"],result["rows"]) for name,result in variants.items()}
    pairs={r["pair_index"]:r for r in length["pairs"]}
    risk=[]
    for threshold in (0,2,5,10,20,40):
        indices={index for index,row in pairs.items() if row["predicted_median_path"]>=threshold}
        risk.append(dict(threshold_model_pixels=threshold,n=len(indices),coverage=len(indices)/204,
            **{name:summary([r for r in variants[name]["rows"] if r["pair_index"] in indices]) for name in ("original",CANDIDATE)}))
    grouped=sensitivity_groups(sensitivity,length["pairs"])
    worst={}
    for kind in ("ground_truth","predicted"):
        result=sensitivity.get(("original",kind,"per_track",1.))
        if result:
            clean={r["pair_index"]:r for r in sensitivity["original",kind,"none",0.]["rows"]}
            worst[kind]=[dict(r,median_gt_path_model_pixels=pairs[r["pair_index"]]["gt_median_path"],
                                  median_predicted_path_model_pixels=pairs[r["pair_index"]]["predicted_median_path"],
                                  axis_error_before_shift_deg=clean[r["pair_index"]].get("axis_error_deg"))
                         for r in sorted(result["rows"],key=lambda r:r.get("axis_drift_deg",-1),reverse=True)[:10]]
    per_joint={str(j):{name:summary([r for r in variants[name]["rows"] if r["joint_ordinal"]==j]) for name in ("original",CANDIDATE)} for j in (1,2,3,4)}
    aggregates=dict(candidate=CANDIDATE,variants={name:r["summary"] for name,r in variants.items()},paired=comparisons,
        controls={name:r["summary"] for name,r in controls.items()},per_joint=per_joint,risk_coverage=risk,
        sensitivity=[dict(variant=v,input=k,pattern=p,amplitude_model_pixels=a,summary=r["summary"],axis_drift=r["axis_drift"]) for (v,k,p,a),r in sensitivity.items()])
    write(OUT/"summary.json",aggregates);write(OUT/"sensitivity_by_length.json",grouped);write(OUT/"worst_shift_cases.json",worst)
    figures(variants,length,sensitivity,risk)
    lines=["# Axis orientation: trajectory length and shift sensitivity","",
        "**80 frozen development scenes; 204 scoreable GT-revolute pairs.** Fixed joints and classification are outside this experiment. "
        "Every estimator is scored on all 204 pairs, including those the original classifier called fixed or prismatic. "
        "The same primary Myriad rollout is used; covariance variants additionally consult the other saved rollouts for uncertainty, not GT selection. "
        "The remaining 240 scenes, production fitter, prior experiment results, model weights and inference archives were not changed.","",
        "## Findings","",
        "The length hypothesis is supported at the level of image-motion direction and across joints. "
        "However, keeping only the longest tracks within a joint does not improve its 3D axis: spatial/depth support is lost, "
        "and common projected movement can be large without strongly constraining 3D rotation. Absolute-pixel fitting already gives "
        "a large penalty to the wrong direction of a long displacement. Extra length weights are not a substitute for correct motion geometry.","",
        f"The frozen candidate `{CANDIDATE}` combines bounded path-length weights, Gaussian-regularized per-frame common "
        "localization corrections, and a stronger soft initial contact-region constraint. The nuisance corrections are shared across points in a frame; "
        "they cannot move each point independently and have a 2-model-pixel prior standard deviation. Length weights are bounded/saturating "
        "and normalized to preserve total data weight. No candidate or point is selected using GT errors.","",
        "![Trajectory length and orientation results](figures/length_and_axis.png)","",
        "## Main paired comparison","",
        "| Metric | Original | Candidate |", "|---|---:|---:|"]
    for label,key,stat in (("Axis mean °","axis","mean"),("Axis median °","axis","median"),("Axis 90th percentile °","axis","p90"),("Rotation-transform error mean °","rotation","mean"),("Pivot-to-axis distance mean, world units","pivot","mean")):
        lines.append(f"| {label} | {base['summary'][key][stat]:.4f} | {candidate['summary'][key][stat]:.4f} |")
    for label,key in (("Axis within 10° %","success_10_deg"),("Axis within 15° %","success_15_deg")):
        lines.append(f"| {label} | {100*base['summary'][key]:.2f} | {100*candidate['summary'][key]:.2f} |")
    item=comparisons[CANDIDATE];low,high=item["interval_95"]
    lines.extend(["",f"Paired mean axis change: **{item['mean_difference']:.3f}°** (candidate minus original); "
        f"scene-bootstrap 95% interval **[{low:.3f}, {high:.3f}]°**; improved on {100*item['improved_fraction']:.1f}% of pairs. "
        "Resampling uses 2,000 scene-cluster draws. This interval is not corrected for trying many configurations; "
        "the small development improvement needs a frozen held-out comparison before replacement. The absolute remaining error is still substantial.","",
        "## Direction versus trajectory length","",
        "Lengths/ADE use 384-image pixels. A track is included here only when all five GT positions are valid. "
        "Direction compares displacements from the observed state; point-states with either displacement below 1 pixel are excluded "
        "from angular/opposite-direction calculations. Opposite fractions are averaged within track, then over tracks. "
        "These GT-dependent diagnostics do not enter estimation.","",
        "| Predicted path px | Complete tracks | Direction-scored tracks | Mean direction error ° | Opposite motion % | Mean absolute ADE px |",
        "|---|---:|---:|---:|---:|---:|"])
    for r in length["bins"]["predicted_path"]:
        low,high=r["range"];label=f"{low}–{high}" if high else f"≥{low}"
        lines.append(f"| {label} | {r['n']} | {r['direction']['n']} | {r['direction']['mean']:.2f} | {100*r['opposite_fraction']['mean']:.2f} | {r['ade']['mean']:.2f} |")
    lines.extend(["", "Grouping by the **true** path instead makes the small-absolute-error issue explicit:","",
        "| GT path px | Complete tracks | Mean absolute ADE px | Mean direction error ° |",
        "|---|---:|---:|---:|"])
    for r in length["bins"]["gt_path"]:
        low,high=r["range"];label=f"{low}–{high}" if high else f"≥{low}"
        lines.append(f"| {label} | {r['n']} | {r['ade']['mean']:.2f} | {r['direction']['mean']:.2f} |")
    lines.extend(["",f"Across pairs, Spearman correlation between median predicted path and original axis error is "
        f"{length['pair_length_axis_spearman']['predicted_median_path']:.3f}; using true path, "
        f"{length['pair_length_axis_spearman']['gt_median_path']:.3f}. These are associations, not evidence that length alone fixes a particular joint.","",
        "The additional radius diagnostic uses distance to the true 3D axis, normalized by the maximum radius among the joint's sampled points. "
        "It tests the near-axis explanation without feeding the true axis into a fitter. A point can be far from a visible connection but close to "
        "the axis line, and projection can make a physically large motion look small. `length_analysis.json` retains the radius groups and decomposes "
        "common image movement from movement differences between points; the latter carries shape-change information that a long common displacement alone lacks.","",
        "## Controlled small shifts","",
        "Observed state 0 remains exact and unchanged. Perturbations affect states 1–4; a magnitude of 1 means "
        "exactly 1 pixel at 384×384, converted to source-camera pixels before fitting. On GT inputs this gives exactly 1 pixel future ADE. "
        "Shared x/y offsets are common to all tracks; per-track offsets have independent deterministic random directions but persist over time; "
        "jitter has independent fixed-magnitude random directions at every future point-state. Seeds are fixed per pair and reused across magnitudes/estimators. "
        "All saved rollouts receive the same perturbation if an uncertainty estimator uses them, so ensemble covariance is not handed an unshifted reference.","",
        "![Controlled shift sensitivity](figures/shift_sensitivity.png)","",
        "| GT-input perturbation | Magnitude px | Original axis mean ° | Candidate axis mean ° | Per-track-offset model mean ° |",
        "|---|---:|---:|---:|---:|"])
    for pattern in ("common_x","per_track","jitter"):
        for amplitude in (.5,1.,2.,4.):
            values=[]
            for name in ("original",CANDIDATE,"bias_track_2px"):
                result=sensitivity.get((name,"ground_truth",pattern,amplitude))
                values.append(f"{result['summary']['axis']['mean']:.3f}" if result else "pending")
            lines.append(f"| {pattern} | {amplitude:g} | {' | '.join(values)} |")
    lines.extend(["", "The original baseline additionally covers 0.25 pixels and shared-y offsets. Candidate comparisons focus on "
        "0.5/1/2/4 pixels for shared-x, persistent per-track offsets and jitter. Axis drift is the unsigned angle "
        "between the perturbed and unperturbed fitted axes; it measures stability, not correctness. Full absolute errors and drift distributions "
        "for GT and actual predicted inputs are retained in `summary.json` and per-condition JSON.","",
        "Actual predicted trajectories, with an added 1-pixel perturbation:","",
        "| Perturbation | Original mean axis change ° | Candidate mean change ° | Per-track-offset mean change ° |",
        "|---|---:|---:|---:|"])
    for pattern in ("common_x","per_track","jitter"):
        values=[]
        for name in ("original",CANDIDATE,"bias_track_2px"):
            result=sensitivity.get((name,"predicted",pattern,1.))
            values.append(f"{result['axis_drift']['mean']:.3f}" if result else "pending")
        lines.append(f"| {pattern} | {' | '.join(values)} |")
    lines.extend(["", "A lower change does not guarantee a more accurate axis; an estimator can be consistently wrong. "
        "The fixed-cohort absolute errors above must be read alongside this stability table.","",
        "## Most sensitive 1-pixel cases","",
        "| Inputs | Pair | Joint | Median GT path px | Axis change ° | Axis error before ° | Axis error after ° |", "|---|---:|---:|---:|---:|---:|---:|"])
    for kind,rows in worst.items():
        for r in rows[:5]:
            path=r['median_gt_path_model_pixels'];label=f"{path:.3f}" if path is not None else "no complete track"
            lines.append(f"| {kind} | {r['pair_index']} | {r['joint_ordinal']} | {label} | {r['axis_drift_deg']:.2f} | {r['axis_error_before_shift_deg']:.2f} | {r['axis_error_deg']:.2f} |")
    lines.extend(["", "## All estimator configurations", "",
        "No joints are removed. Failures would receive a 90° penalty in the primary mean. These runs had no failed fits. "
        "`longest4/6` retain four/six tracks only when there are enough valid correspondences; otherwise they explicitly fall back to all tracks. "
        "`spatial4` balances length with the initial 3D design determinant. Covariance variants use five rollouts; others use one.","",
        "| Variant | Axis mean ° | Median ° | 90th percentile ° | Within 15° % | Paired mean change ° | Observed CPU seconds/pair |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for name,result in sorted(variants.items(),key=lambda x:x[1]["summary"]["axis_with_failure_penalty"]["mean"]):
        s=result["summary"]
        lines.append(f"| {name} | {s['axis_with_failure_penalty']['mean']:.3f} | {s['axis']['median']:.3f} | {s['axis']['p90']:.3f} | "
                     f"{100*s['success_15_deg']:.2f} | {comparisons[name]['mean_difference']:.3f} | {result['seconds']/204:.4f} |")
    lines.extend(["", "CPU times are measured per task while several single-thread CPU workers run concurrently; they are not batch wall time.","",
        "## Logic and feedback by round", "",
        "1. Path/extent weighting, longest-track subsets and geometry-preserving subset selection test whether small tracks corrupt orientation. "
        "Long-only subsets hurt on average; bounded weighting is safer but insufficient.","",
        "2. Future temporal increments cancel persistent future offsets, while regularized shared/per-track offsets explicitly model localization bias. "
        "Increment-only fitting discards useful absolute constraints and worsens orientation; differencing also amplifies independent jitter. "
        "Simple per-track offset tolerance does not solve the actual image-only errors.","",
        "3. Per-frame common offsets and pairwise spatial residuals test whether common translation dominates the axis signal. "
        "Contact-strength controls distinguish this from remote-pivot degeneracy. An instantaneous rotational-flow initializer checks whether "
        "better use of calibrated initial geometry helps; retaining its linearized axis or refining it did not improve overall orientation.","",
        "4. Combine the mechanisms that address separate observed problems: bounded path reliability, small common frame offsets, and contact conditioning. "
        "Also test correlated uncertainty from saved rollouts; it did not beat the single-rollout combined candidate.","",
        "5. Weight by the initial rotational projection Jacobian instead of noisy predicted length. This is a geometry-derived expected-motion cue, "
        "but its standalone and combined variants did not improve on the selected candidate.","",
        "6. Controlled shifts show that per-frame common corrections and persistent per-track corrections solve different error structures. "
        "Combining both with length/contact conditioning (`two_way_conditioned`) slightly worsens real-prediction orientation relative to the "
        "simpler selected candidate. Keep the per-track model as a separate diagnostic rather than claiming one model fixes every type of shift.","",
        "The process stops after these independent tests rather than tuning thresholds to squeeze the development mean. "
        "The result supports better conditioning, but does not establish that axis orientation is reliably recovered.","",
        "## Per-joint results", "",
        "| Joint ordinal | n | Original axis mean ° | Candidate axis mean ° | Original within 15° % | Candidate within 15° % |",
        "|---|---:|---:|---:|---:|---:|"])
    for joint,group in per_joint.items():
        a,b=group["original"],group[CANDIDATE]
        lines.append(f"| {joint} | {a['n']} | {a['axis']['mean']:.3f} | {b['axis']['mean']:.3f} | {100*a['success_15_deg']:.2f} | {100*b['success_15_deg']:.2f} |")
    lines.extend(["", "## Length-based coverage, not an estimator improvement", "",
        "![Length versus coverage](figures/length_coverage.png)","",
        "| Minimum median predicted path px | Retained joints / 204 | Original mean axis ° | Candidate mean axis ° |",
        "|---|---:|---:|---:|"])
    for r in risk:lines.append(f"| {r['threshold_model_pixels']} | {r['n']} | {r['original']['axis']['mean']:.3f} | {r[CANDIDATE]['axis']['mean']:.3f} |")
    lines.extend(["", "This table changes the joint cohort. It must not be presented as a gain on all 204 joints. "
        "Long predicted motion can help identify favorable conditions, but even long trajectories can give wrong axes.","",
        "## Clean-input controls", "",
        "| Algorithm | Axis mean ° | Median ° | 90th percentile ° |", "|---|---:|---:|---:|"])
    for name,result in controls.items():
        s=result["summary"]["axis"];lines.append(f"| {name} | {s['mean']:.6f} | {s['median']:.6f} | {s['p90']:.6f} |")
    lines.extend(["", "Reversing a rotation can preserve the unsigned axis orientation while changing its angle/direction. "
        "Rotation-transform errors are retained separately. Axis-point distances are perpendicular distances to the established FK-consistent reference line; "
        "a pivot's coordinate along its axis remains unidentifiable.","",
        "## Reproduction and boundaries", "",
        "`manifest.json` freezes the same 80 scenes and 204 revolute pairs before this round. `variants/`, `controls/`, and `sensitivity/` "
        "retain every fitted parameter vector, failure, timing, and scorer result. Exact source snapshots are stored by hash under `code/`. "
        "Fitting never receives true axes, future 3D poses, GT lengths or GT future positions outside explicitly labelled GT-input controls. "
        "The correspondence masks retain the pre-existing benchmark's GT geometric validity; deployment must obtain its own validity. "
        "Candidate hyperparameters are development choices, not validated universal settings. No new Myriad inference ran and no GPU was needed.","",
        "`candidate.estimate_revolute_motion` exposes the isolated combined method. It assumes a revolute joint and uses source-camera pixel coordinates, "
        "initial RGB-D/camera/pair labels and predicted tracks. It returns the axis line, time-dependent world transforms and diagnostics. "
        "The production implementation remains unchanged.",""])
    (OUT/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(variants=len(variants),sensitivity_conditions=len(sensitivity),candidate=candidate["summary"],paired=comparisons[CANDIDATE]),indent=2))


if __name__=="__main__":main()
