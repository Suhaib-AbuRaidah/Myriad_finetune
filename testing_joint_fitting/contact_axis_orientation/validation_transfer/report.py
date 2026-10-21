"""Report the frozen algorithm's transfer to disjoint validation objects."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing_joint_fitting.axis_motion.common import summary
from testing_joint_fitting.experiment import write
from training_from_scratch.full_evaluation.metrics import distribution
from .evaluate import ROOT, freeze_protocol


def paired_bootstrap(old, new, metric, cluster="object_id"):
    before = {row["pair_index"]:row for row in old}
    groups = {}
    for row in new:
        a, b = before[row["pair_index"]].get(metric), row.get(metric)
        if a is None or b is None: continue
        groups.setdefault(row[cluster], []).append(b-a)
    totals = np.array([np.sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    indices = np.random.default_rng(20261021).integers(0, len(totals), size=(10000, len(totals)))
    draws = totals[indices].sum(1)/counts[indices].sum(1)
    return dict(mean_change=float(totals.sum()/counts.sum()),
        interval_95=np.quantile(draws, [.025, .975]), cluster=cluster, groups=len(groups))


def grouped(rows, key):
    return {str(value):summary([row for row in rows if row[key]==value]) for value in sorted({r[key] for r in rows})}


def main():
    out = ROOT/"results"
    cohort = json.loads((out/"cohort.json").read_text())
    protocol = freeze_protocol()
    variants = {p.stem:json.loads(p.read_text()) for p in sorted((out/"variants").glob("*.json"))}
    controls = {p.stem:json.loads(p.read_text()) for p in sorted((out/"controls").glob("*.json"))}
    assert set(variants)==set(protocol["configs"]) and set(controls)==set(protocol["configs"])
    candidate, baseline = variants["frozen_candidate"], variants["rgbd_position"]
    for item in [*variants.values(), *controls.values()]:
        assert item["algorithm_sources"] == protocol["algorithm_sources"]
        assert [row["pair_index"] for row in item["rows"]] == cohort["pair_indices"]
    per_object = {name:grouped(item["rows"], "object_id") for name,item in variants.items()}
    per_joint = {name:grouped(item["rows"], "joint_ordinal") for name,item in variants.items()}
    macro = {name:{metric:float(np.mean([s[metric]["mean"] for s in objects.values() if s[metric]["mean"] is not None]))
        for metric in ("axis_with_failure_penalty", "pivot", "rotation")} for name,objects in per_object.items()}
    objects = sorted(per_object["frozen_candidate"])
    overlap = json.loads((ROOT/"object_inventory.json").read_text())
    assert not overlap["overlap"]["val_objects_shared_with_development"]
    before = {row["pair_index"]:row for row in baseline["rows"]}
    improved = sum(row["axis_error_with_failure_penalty_deg"] < before[row["pair_index"]]["axis_error_with_failure_penalty_deg"] for row in candidate["rows"])
    obj_improved = sum(per_object["frozen_candidate"][obj]["axis_with_failure_penalty"]["mean"] <
        per_object["rgbd_position"][obj]["axis_with_failure_penalty"]["mean"] for obj in objects)
    development = json.loads((ROOT.parent/"results/selected.json").read_text())
    intervals = {metric:paired_bootstrap(baseline["rows"], candidate["rows"], metric)
        for metric in ("axis_error_with_failure_penalty_deg", "pivot_distance", "rotation_error_deg")}
    trajectory = {metric:distribution(row["trajectory_metrics"].get(metric) for row in candidate["rows"])
        for metric in ("ADE_1_pixels", "ADE_percent_full_path", "minADE_1", "gt_path_pixels")}
    control_before = {row["pair_index"]:row for row in controls["rgbd_position"]["rows"]}
    introduced_control_errors = [dict(pair_index=row["pair_index"], object_id=row["object_id"],
        joint_ordinal=row["joint_ordinal"], old_axis_error_deg=control_before[row["pair_index"]]["axis_error_deg"],
        frozen_axis_error_deg=row["axis_error_deg"], frozen_rmse_px=row["rmse_px"])
        for row in controls["frozen_candidate"]["rows"] if row["axis_error_with_failure_penalty_deg"]>15
        and control_before[row["pair_index"]]["axis_error_with_failure_penalty_deg"]<=15]
    failure_objects = {}
    for row in candidate["rows"]:
        if row["axis_error_with_failure_penalty_deg"]>15:
            failure_objects[row["object_id"]] = failure_objects.get(row["object_id"], 0)+1
    smaller_errors = distribution(row["axis_error_deg"] for row in candidate["rows"]
        if row["axis_error_with_failure_penalty_deg"]<=15)
    result = dict(protocol=protocol, coverage={key:cohort[key] for key in
        ("scenes", "source_joints", "eligible_pairs", "eligible_revolute_pairs", "excluded_source_revolute_pairs",
         "skipped_revolute", "scoreable_revolute_pairs", "scoreable_scenes", "objects")},
        summary={name:item["summary"] for name,item in variants.items()},
        controls={name:item["summary"] for name,item in controls.items()},
        per_object=per_object, per_joint=per_joint, macro_object_means=macro,
        paired_object_bootstrap=intervals, improved_pairs=improved, improved_objects=obj_improved,
        trajectory_metrics=trajectory, development_summary=development["summary"],
        introduced_control_axis_errors=introduced_control_errors, prediction_failure_objects=failure_objects,
        large_axis_errors=[row for row in candidate["rows"] if row["axis_error_with_failure_penalty_deg"]>15])
    write(out/"summary.json", result)
    lines = ["# Frozen contact-axis fitting on validation objects", "",
        "## Object split and scope", "",
        "The 80-scene fitting development subset covers all 14 test1 objects. The remaining **240** test1 scenes "
        "also contain those same 14 objects. They provide additional poses/configurations, not additional object identities.", "",
        "Validation contains 12 object IDs with **zero overlap** with test1/development and with the 57 training object IDs. "
        "The requested evaluation uses **50 validation scenes**, selected by seeded round-robin object coverage: "
        "four scenes per object, plus one additional scene for two objects. Every eligible joint in those scenes "
        "was enumerated; none was chosen from its fitting score. The remaining 270 validation scenes were not inferred.", "",
        f"The 50 scenes contain {cohort['source_joints']} source joints, {cohort['eligible_pairs']} initially eligible "
        f"parent–child pairs, and {cohort['eligible_revolute_pairs']} eligible revolute pairs. "
        f"The axis comparison scores {cohort['scoreable_revolute_pairs']} revolute pairs across {cohort['scoreable_scenes']} scenes. "
        f"There are {len(cohort['skipped_revolute'])} revolute pairs with unscoreable future trajectories and "
        f"{cohort['excluded_source_revolute_pairs']} source revolute joints excluded by the original initial-image eligibility rule.", "",
        "Fixed joints retain inference outputs but are outside this assumed-revolute fitting experiment, consistent "
        "with development. Source-joint exclusions are listed in the inference manifest; skipped examples remain "
        "explicit in `results/cohort.json`.", "",
        "## Frozen protocol", "",
        "The exact development checkpoint is used: step 108000, SHA256 "
        f"`{protocol['checkpoint_sha256']}`. Original uncropped images, 384 model resolution, 16 diffusion steps, "
        "five hypotheses, hypothesis zero, without-poke mode, and fitting states 2/3/4 match the development evaluation. "
        "Missing original-image DINO features were computed once with the same frozen encoder and stored in the existing "
        "feature cache. The model ran in FP32. No weights were retrained.", "",
        "The recommended strength-128 fitting configuration and source hashes were recorded in `protocol.json` before "
        "validation fitting. No thresholds, prior strengths, point selection rules or candidates were tuned on validation. "
        "GT tracks are used only in the explicitly separate diagnostic controls, and GT joint parameters only for scoring.", "",
        "## Main paired results", "",
        "| Fitter | Pairs | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot distance | Mean rotation ° | Mean reprojection RMSE px | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("original", "rgbd_position", "mesh_position", "frozen_candidate"):
        item = variants[name]; s = item["summary"]
        rmse = distribution(row.get("rmse_px") for row in item["rows"])["mean"]
        lines.append(f"| {name} | {s['n']} | {s['axis_with_failure_penalty']['mean']:.3f} | {s['axis']['median']:.3f} | "
            f"{100*s['success_15_deg']:.1f}% | {s['pivot']['mean']:.5f} | {s['rotation']['mean']:.3f} | {rmse:.3f} | {s['failures']} |")
    lines += ["", "Pivot distance is measured in dataset world units. Reprojection is against the predicted tracks "
        "in original 512-image pixels, not ADE against GT. All fitters use exactly the same pairs, anchors, masks "
        "and predictions. `mesh_position` isolates richer mesh inputs from adding the normal-based orientation prior.", "",
        "![Validation comparison](results/generalization.png)", "",
        "## Comparison with fitting development", "",
        "| Dataset | Scenes | Distinct objects | Pairs | Candidate mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Mean rotation ° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, scenes, obj_count, s in (("Test1 development", 80, 14, development["summary"]),
            ("Validation transfer", 50, len(objects), candidate["summary"])):
        lines.append(f"| {label} | {scenes} | {obj_count} | {s['n']} | {s['axis_with_failure_penalty']['mean']:.3f} | "
            f"{s['axis']['median']:.3f} | {100*s['success_15_deg']:.1f}% | {s['pivot']['mean']:.5f} | {s['rotation']['mean']:.3f} |")
    lines += ["", "## Per-object comparison", "",
        f"Mean axis error improves for **{obj_improved}/{len(objects)} objects** and **{improved}/{len(candidate['rows'])} pairs** "
        "relative to the old visible-RGB-D contact-position prior.", "",
        "| Object ID | Pairs | Old mean axis ° | Frozen mean axis ° | Frozen median axis ° | Old pivot | Frozen pivot | Frozen rotation ° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for obj in objects:
        old, new = per_object["rgbd_position"][obj], per_object["frozen_candidate"][obj]
        lines.append(f"| {obj} | {new['n']} | {old['axis_with_failure_penalty']['mean']:.3f} | {new['axis_with_failure_penalty']['mean']:.3f} | "
            f"{new['axis']['median']:.3f} | {old['pivot']['mean']:.5f} | {new['pivot']['mean']:.5f} | {new['rotation']['mean']:.3f} |")
    lines += ["", "Equal-object macro means (each object counts once):", "",
        "| Fitter | Axis ° | Pivot distance | Rotation ° |", "|---|---:|---:|---:|"]
    for name in ("rgbd_position", "mesh_position", "frozen_candidate"):
        m = macro[name]
        lines.append(f"| {name} | {m['axis_with_failure_penalty']:.3f} | {m['pivot']:.5f} | {m['rotation']:.3f} |")
    lines += ["", "## Per-joint comparison", "",
        "| Joint ordinal | Pairs | Old mean axis ° | Frozen mean axis ° | Axis ≤15° | Old pivot | Frozen pivot | Frozen rotation ° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for joint in per_joint["frozen_candidate"]:
        old, new = per_joint["rgbd_position"][joint], per_joint["frozen_candidate"][joint]
        lines.append(f"| {joint} | {new['n']} | {old['axis_with_failure_penalty']['mean']:.3f} | {new['axis_with_failure_penalty']['mean']:.3f} | "
            f"{100*new['success_15_deg']:.1f}% | {old['pivot']['mean']:.5f} | {new['pivot']['mean']:.5f} | {new['rotation']['mean']:.3f} |")
    lines += ["", "## Object-level uncertainty", "",
        "Paired bootstrap intervals below resample **objects**, preserving each object's scenes and joints. "
        "There are only 12 held-out object identities; intervals are descriptive, not a broad category-generalization guarantee.", ""]
    for metric, interval in intervals.items():
        low, high = interval["interval_95"]
        lines.append(f"- {metric}: mean candidate-minus-baseline change {interval['mean_change']:.6f}; 95% interval [{low:.6f}, {high:.6f}].")
    lines += ["", "## GT-trajectory controls", "",
        "These runs test geometry-prior bias on the same cohort with GT trajectories. They never select a new configuration.", "",
        "| Fitter | Mean axis ° | Maximum axis ° | Mean pivot | Mean rotation ° |", "|---|---:|---:|---:|---:|"]
    for name in ("original", "rgbd_position", "mesh_position", "frozen_candidate"):
        s = controls[name]["summary"]
        lines.append(f"| {name} | {s['axis_with_failure_penalty']['mean']:.5f} | {s['axis']['max']:.5f} | {s['pivot']['mean']:.7f} | {s['rotation']['mean']:.5f} |")
    lines += ["", "**The controls expose prior bias on new objects.** The frozen prior introduces "
        f"{len(introduced_control_errors)} axis errors above 15° on pairs for which the previous position prior "
        "has an accurate axis. These failures persist with GT trajectories, so they cannot be attributed only to "
        "Myriad prediction noise. A dominant contact-normal direction is not always a reliable hinge direction, "
        "and the fixed prior can overpower good trajectory evidence. This is a reason to keep the candidate experimental.", "",
        "| Pair | Object ID | Joint ordinal | Previous GT-track axis ° | Frozen GT-track axis ° | Frozen RMSE px |",
        "|---|---|---:|---:|---:|---:|"]
    for row in introduced_control_errors:
        lines.append(f"| {row['pair_index']} | {row['object_id']} | {row['joint_ordinal']} | "
            f"{row['old_axis_error_deg']:.5f} | {row['frozen_axis_error_deg']:.3f} | {row['frozen_rmse_px']:.3f} |")
    lines += ["", "## Model trajectory quality on this cohort", "",
        "The fitting algorithm does not change the Myriad tracks. These diagnostics help separate model motion errors "
        "from geometric fitting errors; they are identical across fitters.", ""]
    for metric, values in trajectory.items():
        lines.append(f"- {metric}: mean {values['mean']:.5f}, median {values['median']:.5f}, n={values['n']}.")
    lines += ["", "## Interpretation and limits", "",
        "This split checks transfer of the frozen fitting rules to different object identities. The objects remain "
        "robotic arms and can share procedural shape families. It does not establish transfer to unrelated object categories.", "",
        "The neural checkpoint was selected using validation metrics during its original training. This is an independent "
        "object split for fitting development; it is not an untouched end-to-end model-selection test set.", "",
        "The strongest fitter requires full initial mesh samples and their normals. It never receives future mesh states "
        "or true articulation parameters. Any eventual RGB-D-only deployment needs its own check.", "",
        "Axis direction alone does not determine motion: angle magnitude, angle sign and pivot still affect the recovered "
        "transformation. Large errors and fit failures are retained in the per-pair JSON results. "
        f"There are {len(result['large_axis_errors'])} candidate pairs above 15° axis error.", "",
        f"All {len(result['large_axis_errors'])} large predicted-track axis errors are concentrated in "
        f"{len(failure_objects)} object identities. The other {smaller_errors['n']} pairs have "
        f"{smaller_errors['mean']:.3f}° mean axis error. "
        "The higher validation mean versus development is driven by these specific contact-geometry failures. "
        "No validation failures were used to retune or replace the frozen configuration.", "",
        "Frozen inputs, object membership, exclusions, per-pair fit parameters, controls and source hashes are saved "
        "beside this report. The aborted full-320 setup is recorded separately; only the selected 50 scenes received "
        "completed inference in this evaluation."]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for name, label in (("rgbd_position", "Previous contact prior"), ("mesh_position", "Mesh position only"), ("frozen_candidate", "Frozen orientation prior")):
        errors = np.sort([r["axis_error_with_failure_penalty_deg"] for r in variants[name]["rows"]])
        axes[0].plot(errors, np.arange(1, len(errors)+1)/len(errors), lw=2, label=label)
    axes[0].set(xlabel="Validation axis error (degrees)", ylabel="Fraction of pairs", xlim=(0, 90), ylim=(0, 1.01))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    x = np.arange(len(objects))
    axes[1].bar(x-.18, [per_object["rgbd_position"][obj]["axis_with_failure_penalty"]["mean"] for obj in objects], .36, label="Previous")
    axes[1].bar(x+.18, [per_object["frozen_candidate"][obj]["axis_with_failure_penalty"]["mean"] for obj in objects], .36, label="Frozen")
    axes[1].set(xlabel="Validation object (table order)", ylabel="Mean axis error (degrees)", xticks=x, xticklabels=np.arange(1, len(objects)+1))
    axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    joints = list(per_joint["frozen_candidate"]); x = np.arange(len(joints))
    axes[2].bar(x-.18, [per_joint["rgbd_position"][j]["axis_with_failure_penalty"]["mean"] for j in joints], .36, label="Previous")
    axes[2].bar(x+.18, [per_joint["frozen_candidate"][j]["axis_with_failure_penalty"]["mean"] for j in joints], .36, label="Frozen")
    axes[2].set(xlabel="Joint ordinal", ylabel="Mean axis error (degrees)", xticks=x, xticklabels=joints)
    axes[2].legend(); axes[2].grid(axis="y", alpha=.2)
    fig.savefig(out/"generalization.png", dpi=180); plt.close(fig)
    print(json.dumps(dict(summary={name:item["summary"] for name,item in variants.items()},
        improved_objects=obj_improved, objects=len(objects), improved_pairs=improved, macro=macro)))


if __name__ == "__main__": main()
