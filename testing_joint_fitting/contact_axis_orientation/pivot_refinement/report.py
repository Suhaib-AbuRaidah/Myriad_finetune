"""Generate full comparisons and a standalone scientific result plot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from testing_joint_fitting.axis_motion.common import summary
from testing_joint_fitting.experiment import write
from .run import ROOT, OUT


def paired(old, new, metric, cluster="scene_index"):
    before = {r["pair_index"]: r for r in old}
    groups = {}
    for r in new:
        a, b = before[r["pair_index"]].get(metric), r.get(metric)
        if a is not None and b is not None:
            groups.setdefault(r[cluster], []).append(b-a)
    values = list(groups.values())
    sums = np.array([np.sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    rng = np.random.default_rng(20261025)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    changes = np.concatenate(values)
    return dict(mean_change=float(changes.mean()), improved=int((changes < 0).sum()),
        worsened=int((changes > 0).sum()), interval_95=np.quantile(means, [.025, .975]),
        cluster=cluster, replications=10000)


def table_row(name, item):
    s = item["summary"]
    return (f"| {name} | {s['axis']['mean']:.4f} | {s['axis']['median']:.4f} | "
        f"{100*s['success_15_deg']:.2f}% | {s['pivot']['mean']:.6f} | {s['pivot']['median']:.6f} | "
        f"{s['rotation']['mean']:.4f} | {item['mean_rmse_px']:.4f} | {s['failures']} |")


HEADER = ["| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]


def main():
    selected = json.loads((OUT/"selected.json").read_text())
    name = selected["variant"]
    lines = ["# Soft contact-center pivot refinement", "", "Completed 25 October 2026.", "",
        f"Selected `{name}` from {selected['compared_variants']} configurations on the original "
        "80 development scenes / 204 revolute pairs. Axis orientation strength remains 128. "
        "All predictions are the existing uncropped hypothesis-zero trajectories, without a poke, states 2–4. "
        "No neural inference or training was rerun.", ""]
    comparisons = {}
    plot_data = []
    for cohort, title in (("development", "Development: 80 scenes / 204 pairs"),
                           ("validation", "Frozen transfer: 50 scenes / 134 pairs")):
        folder = OUT/cohort
        if not folder.exists():
            continue
        old = json.loads((folder/"variants/orientation_baseline.json").read_text())
        new = json.loads((folder/"variants"/f"{name}.json").read_text())
        control_old = json.loads((folder/"controls/orientation_baseline.json").read_text())
        control_new = json.loads((folder/"controls"/f"{name}.json").read_text())
        assert [r["pair_index"] for r in old["rows"]] == [r["pair_index"] for r in new["rows"]]
        cluster = "object_id" if cohort == "validation" else "scene_index"
        statistics = {metric: paired(old["rows"], new["rows"], metric, cluster)
            for metric in ("pivot_distance", "axis_error_deg", "rotation_error_deg")}
        reduction = 100*(1-new["summary"]["pivot"]["mean"]/old["summary"]["pivot"]["mean"])
        comparisons[cohort] = dict(baseline=old["summary"], selected=new["summary"],
            pivot_reduction_percent=reduction, paired=statistics, groups={})
        lines += [f"## {title}", "", *HEADER, table_row("Previous orientation fitter", old),
            table_row(name, new), "", f"Mean pivot error falls **{reduction:.2f}%**. "
            f"Pivot improves on {statistics['pivot_distance']['improved']}/{len(new['rows'])} pairs; "
            f"{statistics['pivot_distance']['worsened']} worsen. All pairs remain in the comparison.", ""]
        for metric, s in statistics.items():
            low, high = s["interval_95"]
            lines.append(f"- {metric}: paired mean change {s['mean_change']:.6f}; descriptive 95% "
                f"{'object' if cluster == 'object_id' else 'scene'}-cluster bootstrap interval [{low:.6f}, {high:.6f}].")
        lines += ["", "GT-track controls (separate diagnostic fits, not predicted-track performance):", "",
            *HEADER, table_row("Previous, GT tracks", control_old), table_row("Selected, GT tracks", control_new), "",
            f"Maximum control axis error: {control_old['summary']['axis']['max']:.4f}° → "
            f"{control_new['summary']['axis']['max']:.4f}°. "
            "Soft centering still introduces measurable geometry bias on clean tracks.", ""]
        for field, label in (("joint_ordinal", "joint"), ("object_id", "object")):
            lines += [f"### Per-{label} results", "",
                f"| {label.title()} | Pairs | Previous pivot | Selected pivot | Previous axis ° | Selected axis ° | Selected rotation ° |",
                "|---|---:|---:|---:|---:|---:|---:|"]
            groups = {}
            for value in sorted({r[field] for r in new["rows"]}):
                a = summary([r for r in old["rows"] if r[field] == value])
                b = summary([r for r in new["rows"] if r[field] == value])
                groups[str(value)] = dict(baseline=a, selected=b)
                lines.append(f"| {value} | {b['n']} | {a['pivot']['mean']:.6f} | {b['pivot']['mean']:.6f} | "
                    f"{a['axis']['mean']:.4f} | {b['axis']['mean']:.4f} | {b['rotation']['mean']:.4f} |")
            comparisons[cohort]["groups"][field] = groups
            lines.append("")
        plot_data.append((cohort, old, new))
    lines += ["## What changed", "",
        "The original broad contact-position prior allowed noisy predicted motion to displace the axis line. "
        "The new estimator increases that prior's strength and estimates the center using the selected patch rule. "
        "The arithmetic centroid also avoids the coordinate dependence of a coordinate-wise median. "
        "Orientation evidence remains a separate fixed 64-pair patch; changing the position patch cannot change its normal matrix.", "",
        "The existing whitened perpendicular residual is retained:", "", "```text",
        "sigma = max(0.25, seed_median_reprojection_px / sqrt(2 log 2))",
        "effective_strength = position_strength * sigma / hypot(sigma, noise_reference_px)",
        "d = W axis; b = W (pivot - contact_center)",
        "r_position = effective_strength * sigma * (b - d (d.T b)/(d.T d))", "```", "",
        "Without adaptation, noise_reference_px is zero and effective strength equals nominal strength. "
        "At the 0.25-pixel sigma floor, a 1-pixel reference multiplies the nominal strength by about 0.243. "
        "For large sigma the multiplier approaches one. This scale is a heuristic based only on the initial fit, "
        "not a calibrated estimate of model error. Low-residual coherent but wrong tracks can still mislead it.", "",
        "The same robust soft-L1 position and pixel losses, quadratic orientation loss, multistart optimizer, "
        "axis/pivot/angle joint fit and objective-based candidate selection are retained. "
        "No post-fit hard snapping or ground-truth candidate selection is used. "
        "The returned pivot gauge remains the anchor-centroid plane. Moving only along the axis does not affect "
        "pivot distance or the transforms; the improvement comes from moving the infinite line.", "",
        "Selected configuration:", "", "```json", json.dumps(selected["config"], indent=2), "```", "",
        "## Selection and all configurations", "",
        selected["selection"], "", "Guards fixed before completion of the adaptive sweep:", "",
        "```json", json.dumps(selected["guards"], indent=2), "```", "",
        "These are development acceptance limits, not guarantees for new data. "
        "The clean-track pivot guard deliberately allows some bias (0.002 world units); "
        "the full control change is reported above. Larger patches and stronger forces are retained here even when they regress.", "",
        "| Configuration | Mean axis ° | Mean pivot | Rotation ° | GT-track mean pivot | GT-track max axis ° | Eligible / rejection |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for variant in selected["decisions"]:
        item = json.loads((OUT/"development/variants"/f"{variant}.json").read_text())
        control = json.loads((OUT/"development/controls"/f"{variant}.json").read_text())
        s, c = item["summary"], control["summary"]
        checks = selected["decisions"][variant]
        status = "yes" if checks["eligible"] else ", ".join(k for k, v in checks["checks"].items() if not v)
        lines.append(f"| {variant} | {s['axis']['mean']:.4f} | {s['pivot']['mean']:.6f} | {s['rotation']['mean']:.4f} | "
            f"{c['pivot']['mean']:.6f} | {c['axis']['max']:.4f} | {status} |")
    lines += ["", "## Limits and verification", "",
        "Pivot distances use dataset world units and the transform-consistent scoring reference. "
        "Axis errors are sign-invariant; reprojection is against predicted tracks in source-image pixels. "
        "The larger reprojection residual reflects a changed geometry/motion tradeoff; it is not improved trajectory prediction. "
        "Angle recovery remains limited, as the rotation errors show.", "",
        "The repeatedly used development cohort is used for global selection. Scene bootstrap intervals do not account for this search. "
        "The selected configuration was frozen before fitting the existing 50-scene validation archive; "
        "its 12 object identities do not overlap development's 14 objects. Validation was evaluated previously for the old fitter, "
        "so this is a transfer check on an existing archive, not a newly untouched benchmark. "
        "Object bootstrap uses just 12 validation clusters. No validation scores change the selected configuration.", "",
        "The full initial mesh points/normals and existing visibility masks are still required; "
        "RGB-D-only performance is untested. A contact centroid need not lie on a real hinge for every geometry. "
        "No GT joint parameters, future geometry or future transforms enter the estimator. "
        "GT tracks enter only separately labeled controls. Cohort definition uses GT revolute labels, as before.", "",
        "`results/audit.json` independently checks every saved current-sweep fit's axis, pivot, polar-projected SO(3) rotation, "
        "reprojection and positive scored depth. Immutable code snapshots are hash-verified. "
        "Synthetic tests cover exact recovery, noisy-track displacement, pivot gauge, rigid-frame centroids, missing geometry, "
        "and the existing orientation invariants. Production code, the prior frozen prototype, training and original artifacts remain intact.", "",
        "![Pivot and axis comparison](results/comparison.png)", ""]
    write(OUT/"comparison.json", dict(selection=selected, comparisons=comparisons))
    (ROOT/"REPORT.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(len(plot_data), 2, figsize=(11, 4*len(plot_data)), squeeze=False, constrained_layout=True)
    for axrow, (cohort, old, new) in zip(axes, plot_data):
        for axis, metric, label in zip(axrow, ("pivot_distance", "axis_error_deg"), ("Pivot distance (world units)", "Axis error (degrees)")):
            for item, legend in ((old, "Previous"), (new, "Refined")):
                values = np.sort([r[metric] for r in item["rows"] if r.get(metric) is not None])
                axis.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=legend)
            axis.set(xlabel=label, ylabel="Fraction of pairs", title=cohort.title())
            axis.grid(alpha=.25)
            axis.legend()
            axis.set_xlim(0, .12 if metric == "pivot_distance" else 5.)
    fig.savefig(OUT/"comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({c: dict(pivot_reduction_percent=v["pivot_reduction_percent"],
        selected=v["selected"]) for c, v in comparisons.items()}))


if __name__ == "__main__":
    main()
