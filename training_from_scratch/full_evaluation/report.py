"""Rebuild tables, distributions, plots, and a readable report from records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from finetuning_pair.evaluate import write_json
from .metrics import KINDS, METRICS, clean, distribution, summarize


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def bootstrap_mean(rows, key, draws=2000):
    grouped = {}
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            grouped.setdefault(row["scene_index"], []).append(value)
    if not grouped:
        return dict(n=0, scenes=0, mean=None, ci95=None)
    sums = np.array([sum(values) for values in grouped.values()])
    counts = np.array([len(values) for values in grouped.values()])
    indices = np.random.default_rng(42).integers(len(sums), size=(draws, len(sums)))
    means = sums[indices].sum(1) / counts[indices].sum(1)
    return dict(n=int(counts.sum()), scenes=len(counts), mean=float(sums.sum()/counts.sum()),
                ci95=np.percentile(means, [2.5, 97.5]).tolist())


def build_summary(manifest, rows):
    summary = {}
    for mode in manifest["modes"]:
        subset = [row for row in rows if row["mode"] == mode]
        overall = summarize(subset, source_count=manifest["all_source_joints"])
        overall["confidence_intervals"] = {key: bootstrap_mean(subset, key) for key in (
            "ADE_1_pixels", "ADE_percent_full_path", "axis_error_deg", "kinematic_pivot_distance")}
        def group(key, values, *, inventory_key=None):
            return {str(value): summarize([r for r in subset if r[key] == value],
                source_count=sum(r[inventory_key] == value for r in manifest["inventory"]) if inventory_key else None)
                for value in sorted(values)}
        per_joint = group("joint_index", {r["joint_index"] for r in manifest["inventory"]}, inventory_key="joint_index")
        per_depth = group("chain_depth", {r["chain_depth"] for r in manifest["inventory"]}, inventory_key="chain_depth")
        per_object = group("object_id", {r["object_id"] for r in subset})
        per_scene = group("scene_index", range(len(manifest["scenes"])), inventory_key="scene_index")
        per_type = group("gt_type", KINDS)
        summary[mode] = dict(overall=overall, per_joint=per_joint, per_chain_depth=per_depth,
                             per_object=per_object, per_scene=per_scene, per_gt_type=per_type)
    return clean(summary)


def common_point_pairs(out, manifest, indexed):
    """Use the same source point slots as well as the same future states."""
    pairs=[]
    scale=manifest['train_config']['image_size']-1
    for i,a in indexed['without_poke'].items():
        b=indexed['with_poke'].get(i)
        if b is None or a.get('skipped') or b.get('skipped'):
            continue
        with np.load(out/'predictions'/f'{i:04d}_without_poke.npz',allow_pickle=False) as x, np.load(out/'predictions'/f'{i:04d}_with_poke.npz',allow_pickle=False) as y:
            _,ia,ib=np.intersect1d(x['point_indices'],y['point_indices'],return_indices=True)
            if not len(ia):
                continue
            gt=x['ground_truth'][:,ia].astype(float)
            np.testing.assert_allclose(gt,y['ground_truth'][:,ib],rtol=0,atol=0)
            valid=x['valid'][:,ia]&y['valid'][:,ib]
            da=np.linalg.norm((x['predictions'][0][:,ia]-gt)*scale,axis=-1)
            db=np.linalg.norm((y['predictions'][0][:,ib]-gt)*scale,axis=-1)
        if not valid[2:].any():
            continue
        aa,bb=dict(a),dict(b)
        aa['common_ADE_pixels']=float(da[2:][valid[2:]].mean())
        bb['common_ADE_pixels']=float(db[2:][valid[2:]].mean())
        length=np.linalg.norm(np.diff(gt,axis=0)*scale,axis=-1).sum(0)
        keep=valid.all(0)&(length>=manifest['min_path_pixels'])&(a['gt_type']!='fixed')
        aa['common_ADE_percent_full_path']=float(np.mean(100*da[2:].mean(0)[keep]/length[keep])) if keep.any() else None
        bb['common_ADE_percent_full_path']=float(np.mean(100*db[2:].mean(0)[keep]/length[keep])) if keep.any() else None
        pairs.append((aa,bb))
    return pairs


def matched_comparisons(manifest, rows, out=None):
    if not {"without_poke", "with_poke"}.issubset(manifest["modes"]):
        return {}
    indexed = {mode: {r["pair_index"]: r for r in rows if r["mode"] == mode} for mode in manifest["modes"]}
    common=common_point_pairs(out,manifest,indexed) if out is not None else None
    result = {}
    for metric in ("common_ADE_pixels", "common_ADE_percent_full_path", "axis_error_deg",
                   "kinematic_pivot_distance", "forced_revolute_axis_error_deg", "rotation_geodesic_deg"):
        paired = [(row, indexed["with_poke"][i]) for i, row in indexed["without_poke"].items()
                  if i in indexed["with_poke"] and row.get(metric) is not None and indexed["with_poke"][i].get(metric) is not None]
        if common is not None and metric.startswith('common_ADE'):
            paired=[(a,b) for a,b in common if a.get(metric) is not None and b.get(metric) is not None]
        differences = [dict(scene_index=a["scene_index"], delta=b[metric]-a[metric]) for a, b in paired]
        result[metric] = dict(without=distribution(a[metric] for a, b in paired),
                              with_poke=distribution(b[metric] for a, b in paired),
                              difference=bootstrap_mean(differences, "delta"))
    return result


def normalization_sensitivity(manifest, rows):
    result=[]
    for mode in manifest['modes']:
        for threshold in sorted({.1,manifest['min_path_pixels'],5.,10.}):
            pair_values=[]; tracks=0
            for row in rows:
                if row['mode']!=mode or row.get('skipped') or row['gt_type']=='fixed':
                    continue
                values=[100*t['ADE_pixels']/t['path_pixels'] for t in row.get('track_metrics',[])
                        if t['complete'] and t['path_pixels']>=threshold]
                if values:
                    pair_values.append(float(np.mean(values))); tracks+=len(values)
            result.append(dict(mode=mode,min_path_pixels=threshold,tracks=tracks,**distribution(pair_values)))
    return result


def normalized_stationary_comparison(out, manifest, rows, sensitivity):
    """Compare normalized errors on exactly the same complete moving tracks."""
    collected={(r['mode'],r['min_path_pixels']):[] for r in sensitivity}
    scale=manifest['train_config']['image_size']-1
    for row in rows:
        if row.get('skipped') or row['gt_type']=='fixed':
            continue
        with np.load(out/'predictions'/f"{row['pair_index']:04d}_{row['mode']}.npz",allow_pickle=False) as a:
            gt=a['ground_truth'].astype(float); valid=a['valid']; start=int(a['known_states'])
        length=np.linalg.norm(np.diff(gt,axis=0)*scale,axis=-1).sum(0)
        baseline=np.linalg.norm((gt[start:]-gt[start-1])*scale,axis=-1).mean(0)
        model=np.asarray([t['ADE_pixels'] if t['complete'] else 0 for t in row['track_metrics']])
        for mode,threshold in collected:
            if mode!=row['mode']:
                continue
            keep=valid.all(0)&(length>=threshold)
            if keep.any():
                m=float(np.mean(100*model[keep]/length[keep])); b=float(np.mean(100*baseline[keep]/length[keep]))
                collected[mode,threshold].append(dict(pair_index=row['pair_index'],joint_index=row['joint_index'],model=m,stationary=b))
                if threshold==manifest['min_path_pixels']:
                    row['stationary_ADE_percent_full_path']=b
                    row['improves_normalized_stationary']=m<b
    result=[]
    for s in sensitivity:
        values=collected[s['mode'],s['min_path_pixels']]
        base=distribution(v['stationary'] for v in values)
        model=distribution(v['model'] for v in values)
        s['stationary_mean']=base['mean']
        s['fraction_beating_normalized_stationary']=float(np.mean([v['model']<v['stationary'] for v in values])) if values else None
        result.append(dict(mode=s['mode'],min_path_pixels=s['min_path_pixels'],model=model,stationary=base,
            fraction_beating_baseline=s['fraction_beating_normalized_stationary'],
            per_joint={str(j):dict(model=distribution(v['model'] for v in values if v['joint_index']==j),
                                  stationary=distribution(v['stationary'] for v in values if v['joint_index']==j))
                       for j in sorted({v['joint_index'] for v in values})}))
    return result


def generate_report(out):
    out = Path(out)
    manifest = json.loads((out / "run.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted((out / "records").glob("*.json"))]
    expected_keys = {(i, mode) for i in range(len(manifest["pairs"])) for mode in manifest["modes"]}
    actual = {(r["pair_index"], r["mode"]) for r in rows}
    if actual != expected_keys or len(rows) != len(actual):
        raise ValueError(f"Incomplete or duplicate evaluation: {len(actual)} records; expected {len(expected_keys)}")
    for row in rows:
        if not row.get("skipped") and not (out / "predictions" / f"{row['pair_index']:04d}_{row['mode']}.npz").exists():
            raise ValueError("A scored record is missing its persisted predictions")
    summary = build_summary(manifest, rows)
    inference_coverage = {mode: dict(
        attempted=sum(r['mode']==mode for r in rows),
        predictions_saved=sum(r['mode']==mode and (out/'predictions'/f"{r['pair_index']:04d}_{mode}.npz").exists() for r in rows),
        scoreable=sum(r['mode']==mode and not r.get('skipped') for r in rows)) for mode in manifest['modes']}
    write_json(out/'inference_coverage.json',inference_coverage)
    comparisons = matched_comparisons(manifest, rows, out)
    sensitivity=normalization_sensitivity(manifest,rows)
    normalized_baseline=normalized_stationary_comparison(out,manifest,rows,sensitivity)
    write_json(out/'normalized_stationary_comparison.json',normalized_baseline)
    write_json(out/'normalization_sensitivity.json',sensitivity)
    write_csv(out/'normalization_sensitivity.csv',sensitivity)
    controls = [dict(r["ground_truth_control"], joint_index=r["joint_index"]) for r in rows if r.get("ground_truth_control")]
    control = summarize(controls)
    references = [r["kinematic_reference"] for r in rows if r["mode"] == manifest["modes"][0] and r.get("kinematic_reference", {}).get("observable")]
    consistency = dict(observable_revolute_pairs=len(references),
        stored_pivot_off_axis_over_1e_4=sum(r["stored_pivot_axis_offset"] > 1e-4 for r in references),
        stored_pivot_offset=distribution(r["stored_pivot_axis_offset"] for r in references),
        stored_axis_error_deg=distribution(r["stored_axis_error_deg"] for r in references),
        transform_residual=distribution(r["transform_residual"] for r in references))
    write_json(out / "summary.json", summary)
    write_json(out / "matched_mode_comparison.json", clean(comparisons))
    write_json(out / "ground_truth_control.json", control)
    write_json(out / "dataset_geometry_consistency.json", clean(consistency))
    write_csv(out / "per_pair.csv", [{k:v for k,v in r.items() if k != "track_metrics"} for r in rows])
    write_csv(out / "per_track.csv", [dict(pair_index=r["pair_index"], mode=r["mode"], scene_index=r["scene_index"],
        joint_index=r["joint_index"], gt_type=r["gt_type"], **track) for r in rows for track in r.get("track_metrics", [])])
    write_csv(out / "excluded_source_joints.csv", [r for r in manifest["inventory"] if not r["eligible"]])
    for group in ("per_joint", "per_chain_depth", "per_object", "per_scene", "per_gt_type"):
        flattened = []
        for mode, modes in summary.items():
            for key, result in modes[group].items():
                item = dict(mode=mode, group=key, scored=result["scored"], source=result["source_count"],
                    fitted=result["fitted"], type_accuracy=result["type_accuracy"], balanced_accuracy=result["balanced_accuracy"])
                for metric, stats in result["metrics"].items():
                    for field in ("n", "mean", "median", "p90", "p95"):
                        item[f"{metric}_{field}"] = stats[field]
                flattened.append(item)
        write_csv(out / f"{group}_summary.csv", flattened)
    ranked = {}
    for mode in manifest["modes"]:
        ranked[mode] = {}
        for metric in ("ADE_1_pixels", "ADE_percent_full_path", "axis_error_deg", "kinematic_pivot_distance"):
            ordered = sorted([r for r in rows if r["mode"] == mode and r.get(metric) is not None], key=lambda r:r[metric])
            def case(row):
                return {key: row[key] for key in ("pair_index", "scene_index", "joint_index", "object_id", metric)}
            ranked[mode][metric] = dict(best=[case(r) for r in ordered[:10]], worst=[case(r) for r in ordered[-10:][::-1]])
    write_json(out / "ranked_cases.json", clean(ranked))
    render_plots(out, summary, rows)
    render_examples(out, rows)
    write_report(out, manifest, summary, control, consistency, comparisons, ranked, inference_coverage, sensitivity,normalized_baseline)
    write_json(out / "status.json", dict(status="complete", scenes=len(manifest["scenes"]),
        eligible_pairs=len(manifest["pairs"]), source_joints=manifest["all_source_joints"],
        records=len(rows), expected=len(expected_keys), checkpoint_step=manifest["checkpoint_step"]))
    print(json.dumps(dict(event="complete", output=str(out), checkpoint_step=manifest["checkpoint_step"],
                         scenes=len(manifest["scenes"]), records=len(rows))), flush=True)


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}g}"


def write_report(out, manifest, summary, control, consistency, comparisons, ranked, inference_coverage, sensitivity=None,normalized_baseline=None):
    lines = [f"# Full {manifest['split']} evaluation: FP32 MYRIAD", "",
        f"Checkpoint: `{manifest['checkpoint_source']}`; **step {manifest['checkpoint_step']:,}**. "
        f"SHA-256: `{manifest['checkpoint_sha256']}`. The hard-linked `checkpoint.pt` freezes this version without duplicating its data.", "",
        f"All **{len(manifest['scenes'])} scenes** and **{len(manifest['pairs'])} eligible parent/child pairs** were enumerated. "
        f"The source contains {manifest['all_source_joints']} joint instances; "
        f"{sum(not r['eligible'] for r in manifest['inventory'])} lack an initially visible valid-depth child point inside the crop. "
        "Their exclusions are recorded explicitly. No random scene subset is used.", "",
        f"FP32 inference; cached frozen DINO features; {manifest['ensemble_size']} hypotheses; "
        f"{manifest['diffusion_steps']} flow steps; seed {manifest['seed']} + pair index; "
        f"model image size {manifest['train_config']['image_size']} pixels. Both GPUs process disjoint pairs, with no change to per-pair seeds.", "",
        "## Main results", "",
        "Without-poke is the primary image-only result: initial RGB, pair masks, and initial point locations. "
        "With-poke additionally receives the ground-truth first motion and is a separate diagnostic. "
        "Hypothesis 0 supplies primary trajectory and joint-estimation metrics; best-of-five uses ground truth for oracle selection.", "",
        "| Mode | Scored / eligible | ADE px | FDE px | ADE / GT path % (pairs) | minADE_1 | minADE_5 | Type accuracy | Axis error deg (n) | Kinematic pivot distance (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, groups in summary.items():
        s, m = groups["overall"], groups["overall"]["metrics"]
        lines.append(f"| {mode} | {s['scored']}/{len(manifest['pairs'])} | {fmt(m['ADE_1_pixels']['mean'])} | {fmt(m['FDE_1_pixels']['mean'])} | "
            f"{fmt(m['ADE_percent_full_path']['mean'])} ({m['ADE_percent_full_path']['n']}) | {fmt(m['minADE_1']['mean'])} | {fmt(m['minADE_5']['mean'])} | "
            f"{fmt(100*s['type_accuracy'])}% | {fmt(m['axis_error_deg']['mean'])} ({m['axis_error_deg']['n']}) | "
            f"{fmt(m['kinematic_pivot_distance']['mean'])} ({m['kinematic_pivot_distance']['n']}) |")
    for mode, counts in inference_coverage.items():
        lines += ["", f"**Inference coverage, {mode}:** {counts['attempted']} pairs visited, "
            f"{counts['predictions_saved']} prediction archives, {counts['scoreable']} scoreable pairs. "
            "Inputs with no valid future GT retain predictions where observed points are available; their error is undefined. "
            "A with-poke input additionally needs points valid at both observed states."]
    for mode, groups in summary.items():
        s = groups["overall"]; m = s["metrics"]
        base, ade = m["stationary_ADE_pixels"]["mean"], m["ADE_1_pixels"]["mean"]
        relation = "lower" if ade < base else "higher"
        lines += ["", f"**{mode}:** ADE is {fmt(100*abs(ade/base-1))}% {relation} than the stationary baseline "
            f"({fmt(base)} px); {fmt(100*s['beats_stationary_fraction'])}% of pairs improve on its squared-error baseline. "
            f"There are {s['fit_failures']} geometry-fit failures. Balanced type accuracy is {fmt(100*s['balanced_accuracy'])}%. "
            f"Median/p90 axis errors are {fmt(m['axis_error_deg']['median'])}°/{fmt(m['axis_error_deg']['p90'])}°; "
            f"median/p90 kinematic pivot distances are {fmt(m['kinematic_pivot_distance']['median'])}/{fmt(m['kinematic_pivot_distance']['p90'])} world units.", ""]
        lines += ["Mean-metric 95% intervals resample whole scenes (2,000 draws), preserving correlation among joints:", "",
                  "| Metric | Mean | 95% interval | Pairs / scenes |", "|---|---:|---:|---:|"]
        for key, ci in s["confidence_intervals"].items():
            lines.append(f"| {key} | {fmt(ci['mean'])} | {' to '.join(fmt(v) for v in ci['ci95']) if ci['ci95'] else 'N/A'} | {ci['n']} / {ci['scenes']} |")
    lines += ["", "## Interpretation", ""]
    for item in normalized_baseline or []:
        if item['min_path_pixels']!=manifest['min_path_pixels']:
            continue
        relation='higher' if item['model']['mean']>item['stationary']['mean'] else 'lower'
        lines += [f"- **{item['mode']}, normalized stationary comparison:** mean ADE/GT-path is "
            f"{fmt(item['model']['mean'])}% for the model versus {fmt(item['stationary']['mean'])}% for stationary prediction "
            f"on the same {item['model']['n']} moving-pair cohort. The model's mean relative error is {relation}; "
            f"{fmt(100*item['fraction_beating_baseline'])}% of those pairs improve on the normalized baseline. "
            "Averaging error/path ratios emphasizes short trajectories differently from averaging pixel errors, "
            "so a gain in absolute pixel ADE can coexist with worse mean normalized ADE."]
    for mode,groups in summary.items():
        s=groups['overall']; support=s['gt_types']; total=sum(support.values())
        majority=max(support,key=support.get)
        lines += [f"- **{mode}, type imbalance:** always predicting {majority} would achieve "
            f"{fmt(100*support[majority]/total)}% accuracy on the scored cohort. The estimated type accuracy is "
            f"{fmt(100*s['type_accuracy'])}% and balanced accuracy is {fmt(100*s['balanced_accuracy'])}%. "
            f"Fixed-joint recall is {s['confusion']['fixed']['fixed']}/{support['fixed']}; "
            f"revolute recall is {s['confusion']['revolute']['revolute']}/{support['revolute']}. "
            "Type accuracy alone can conceal poor detection of the minority class."]
    if comparisons.get('common_ADE_pixels',{}).get('without',{}).get('mean'):
        c=comparisons['common_ADE_pixels']; a=c['without']['mean']; b=c['with_poke']['mean']
        lines += [f"- **Observed-motion benefit:** on identical point slots and states 2–4, ADE changes from "
            f"{fmt(a)} to {fmt(b)} px ({fmt(100*(1-b/a))}% lower with the GT hint). "
            "This is a diagnostic of conditioning on true first motion; it is distinct from image-only inference."]
    lines += ["- **Joint order:** compare pixel and relative errors together. A lower pixel ADE on a joint with "
        "smaller ground-truth motion can coexist with a higher path-normalized or axis error. "
        "The per-joint table, type support, path-length sensitivity, and confidence intervals retain these distinctions.",
        "- **Geometry:** compare mean, median and tails, and read the correctly typed cohort sizes. "
        "Ground-truth-track controls below reveal estimator limits; the stored-pivot consistency check separates reference discrepancies from prediction error.", "",
        "## Per-joint results", "",
        "Joint ordinal = dataset joint_index + 1: joint 1 is index 0, joint 2 is index 1, etc. "
        "These labels pool different objects. Tree depth is reported separately because joint ID and depth can differ in branched trees.", "",
        "| Mode / joint | Scored / source | GT revolute / fixed | ADE px | ADE / GT path % | Type accuracy | Axis mean / median deg (n) | Kinematic pivot mean / median (n) | Rotation error deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, groups in summary.items():
        for joint, s in groups["per_joint"].items():
            m = s["metrics"]
            lines.append(f"| {mode} / joint {int(joint)+1} | {s['scored']}/{s['source_count']} | {s['gt_types']['revolute']}/{s['gt_types']['fixed']} | {fmt(m['ADE_1_pixels']['mean'])} | "
                f"{fmt(m['ADE_percent_full_path']['mean'])} | {fmt(100*s['type_accuracy']) if s['type_accuracy'] is not None else 'N/A'}% | "
                f"{fmt(m['axis_error_deg']['mean'])}/{fmt(m['axis_error_deg']['median'])} ({m['axis_error_deg']['n']}) | "
                f"{fmt(m['kinematic_pivot_distance']['mean'])}/{fmt(m['kinematic_pivot_distance']['median'])} ({m['kinematic_pivot_distance']['n']}) | "
                f"{fmt(m['rotation_geodesic_deg']['mean'])} |")
    lines += ["", "## Metric definitions and coverage", "",
        "- Pixel ADE is mean Euclidean position error over valid predicted future point-states; FDE scores the last state. "
        "Without-poke scores states 1–4, with-poke scores states 2–4. common_ADE_pixels scores states 2–4 in both. "
        "Model pixels use the 384-pixel resized crop (normalized differences × 383). ADE_source_pixels converts errors back to the source image scale.",
        "- The repository's minADE is mean **squared** Euclidean error in [-1,1] coordinates, not Euclidean ADE. "
        "minADE_5 chooses one complete hypothesis with the lowest masked mean squared error; oracle_5_ADE_pixels scores that same hypothesis in pixels.",
        "- ADE_percent_full_path = 100 × per-track ADE / sum of GT segment lengths across states 0–4, "
        "then averaged over eligible tracks within each pair and equally over pairs. Values can exceed 100%. "
        f"A track must have valid positions at all five states, belong to a non-fixed GT joint, and have GT path length ≥ {manifest['min_path_pixels']} model pixels. "
        "Short/fixed/incomplete tracks keep absolute scores and are excluded only from percentages. "
        "ADE_percent_path_weighted uses the ratio of sums within a pair; ADE_percent_forecast_path uses only segments after the last observed state.",
        "- Joint type is estimated from predicted trajectories by the existing calibrated RGB-D/PnP and robust reprojection fitter, "
        "rather than a joint-classification head in MYRIAD. Candidate fixed/prismatic/revolute models use no GT type, axis, pivot, or future pose in fitting. "
        "The fitter uses initial depth/camera calibration and GT future geometric-validity masks; its 3-source-pixel stationary guard and unpenalized robust SSE can affect type decisions.",
        "- Geometry uses hypothesis 0 and only predicted states 2–4 in both modes, with a stationary parent. "
        "Axis orientation error is acos(abs(dot(estimated axis, GT axis))) in degrees [0,90]; axis sign is immaterial. "
        "rotation_geodesic_deg is the SO(3) rotation error, and angle_magnitude_mae_deg measures principal rotation amount error. "
        "Fixed/prismatic predictions contribute identity rotation, so rotation errors include type mistakes.",
        "- Main axis/pivot metrics are conditional on correctly typed GT revolute joints. Always read their n with type recall. "
        "Forced-revolute metrics also score the fitted revolute candidate for misclassified GT revolute joints. "
        "Fixed joints have no identifiable axis/pivot error. Unobservable relative rotations have no transform-derived pivot reference.",
        "- Pivot distance is perpendicular distance from the GT pivot to the estimated axis line, invariant to choosing another point along the axis. "
        "Raw Euclidean pivot-to-pivot distance is not identifiable along a rotation axis. Distances remain in dataset world units; no meter/cm conversion is assumed. "
        "Both stored-annotation and transform-consistent kinematic pivot distances are reported.",
        "- Overall means weight each eligible pair equally. Type accuracy includes fit failures as incorrect among scored pairs; "
        "source coverage also exposes initially ineligible joints. Group summaries retain n/mean/median/std/p90/p95/min/max.", ""]
    for mode, groups in summary.items():
        s = groups["overall"]
        lines += [f"**Percentage coverage, {mode}:** {s['tracks']['normalized_tracks']} normalized tracks / "
            f"{s['tracks']['tracks_total']} total. Excluded: {s['tracks']['fixed_tracks_excluded']} fixed, "
            f"{s['tracks']['short_path_tracks_excluded']} short moving paths, {s['tracks']['incomplete_tracks']} incomplete tracks.", ""]
    if sensitivity:
        lines += ["### Sensitivity to short GT paths", "",
            "The same per-track normalization is recomputed with several minimum GT path lengths. "
            "Absolute metrics always retain these tracks. This shows how small denominators affect percentage means.", "",
            "| Mode | Minimum GT path px | Pairs | Tracks | Mean ADE % | Stationary mean % | Median ADE % | p90 ADE % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sensitivity:
            lines.append(f"| {r['mode']} | {r['min_path_pixels']} | {r['n']} | {r['tracks']} | "
                         f"{fmt(r['mean'])} | {fmt(r.get('stationary_mean'))} | {fmt(r['median'])} | {fmt(r['p90'])} |")
        lines.append("")
    lines += ["## Joint-type confusion and geometry control", ""]
    for mode, groups in summary.items():
        s = groups["overall"]
        lines += [f"### {mode}", "", "| GT / predicted | fixed | prismatic | revolute | Fit failures | Recall |",
                  "|---|---:|---:|---:|---:|---:|"]
        for kind in KINDS:
            lines.append(f"| {kind} | " + " | ".join(str(s['confusion'][kind][p]) for p in KINDS) +
                f" | {s['fit_failures_by_type'][kind]} | {fmt(s['recall'][kind])} |")
    cm = control["metrics"]
    lines += ["", f"The identical fitter on GT tracks classifies {fmt(100*control['type_accuracy'])}% of joints correctly. "
        f"Control axis error mean/median: {fmt(cm['axis_error_deg']['mean'])}°/{fmt(cm['axis_error_deg']['median'])}°. "
        f"Control kinematic pivot distance mean/median: {fmt(cm['kinematic_pivot_distance']['mean'])}/{fmt(cm['kinematic_pivot_distance']['median'])} "
        f"(n={cm['kinematic_pivot_distance']['n']}). This measures fitter/convention limits separately from model error.", "",
        f"Among {consistency['observable_revolute_pairs']} observable GT revolute pairs, "
        f"{consistency['stored_pivot_off_axis_over_1e_4']} stored pivots are more than 1e-4 world units from the transform-consistent axis line. "
        f"Median/p90 annotation offset: {fmt(consistency['stored_pivot_offset']['median'])}/{fmt(consistency['stored_pivot_offset']['p90'])}. "
        "Use kinematic_pivot_distance for consistency with the target-generating transforms, and retain pivot_distance to expose the annotation discrepancy.", "",
        "## Matched motion-hint comparison", "",
        "Trajectory comparisons below use the same source point slots, joint pairs, valid point-states, and states 2–4. "
        "Axis/pivot rows use only pairs scoreable in both modes; "
        "type-conditional cohorts can differ from the overall tables. Negative differences favor with-poke.", "",
        "| Metric | Matched pairs | Without | With | With − without | 95% scene-bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|"]
    for key, item in comparisons.items():
        ci = item['difference']
        lines.append(f"| {key} | {ci['n']} | {fmt(item['without']['mean'])} | {fmt(item['with_poke']['mean'])} | "
            f"{fmt(ci['mean'])} | {' to '.join(fmt(v) for v in ci['ci95']) if ci['ci95'] else 'N/A'} |")
    lines += ["", "## Detailed metric distributions", ""]
    for mode, groups in summary.items():
        lines += [f"### {mode}", "", "| Metric | n | Mean | Median | p90 | p95 | Max |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, stats in groups["overall"]["metrics"].items():
            lines.append(f"| {key} | {stats['n']} | " + " | ".join(fmt(stats[x]) for x in ("mean", "median", "p90", "p95", "max")) + " |")
    lines += ["", "## Artifacts", "",
        "- `summary.json`: overall, per-joint, per-depth, per-object, per-scene and per-GT-type distributions and confusion matrices.",
        "- `per_pair.csv`, `per_track.csv`, `per_*_summary.csv`: detailed numeric tables, with exclusions and denominator coverage.",
        "- `ground_truth_control.json`, `dataset_geometry_consistency.json`, `matched_mode_comparison.json`: geometry controls and paired comparisons.",
        "- `normalized_stationary_comparison.json`, `normalization_sensitivity.csv`: stationary baseline on the identical normalized cohort and sensitivity to minimum GT path length.",
        "- `predictions/`: all five predicted trajectories, GT, validity masks, points and exact conditioned images for every scored pair/mode.",
        "- `sources/`: initial RGB-D/segmentation, calibration, transforms and joint metadata once per source scene.",
        "- `records/`: complete per-pair metrics and fitted joint parameters; `ranked_cases.json`: best/worst examples.",
        "- `figures/`: distribution, per-joint, confusion, and motion-error plots as PNG/PDF; `examples/`: representative best/median/worst rollouts.",
        "- `run.json`, `status.json`, `worker_*.log`: provenance, complete coverage checks, seeds, and progress.", ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def render_plots(out, summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = out / "figures"
    target.mkdir(exist_ok=True)
    labels=dict(ADE_1_pixels='Trajectory ADE (model pixels)',ADE_percent_full_path='ADE / GT path (%)',
                axis_error_deg='Axis orientation error (degrees)',kinematic_pivot_distance='Pivot-to-axis distance (world units)',
                rotation_geodesic_deg='Rotation geodesic error (degrees)',FDE_1_pixels='Final displacement error (model pixels)')
    mode_labels={'without_poke':'Image only','with_poke':'GT motion hint'}
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(target / f"{name}.png", dpi=170)
        fig.savefig(target / f"{name}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    keys = ("ADE_1_pixels", "ADE_percent_full_path", "axis_error_deg", "kinematic_pivot_distance", "rotation_geodesic_deg", "FDE_1_pixels")
    for mode in summary:
        for ax, key in zip(axes.flat, keys):
            values = sorted(r[key] for r in rows if r["mode"] == mode and r.get(key) is not None)
            if values:
                ax.plot(values, np.arange(1, len(values)+1)/len(values), label=f"{mode_labels[mode]} (n={len(values)})")
            ax.set(xlabel=labels[key], ylabel="Cumulative pair fraction", ylim=(0, 1.02))
            if key in ("kinematic_pivot_distance", "ADE_percent_full_path"):
                ax.set_xscale("symlog", linthresh=.01 if key.startswith("kinematic") else 1)
            if key == "axis_error_deg":
                ax.set_xlim(0, 90)
            ax.grid(alpha=.25); ax.legend(fontsize=7)
    save(fig, "error_distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for mode, groups in summary.items():
        joints = groups["per_joint"]
        for ax, key in zip(axes.flat, keys):
            ax.plot([int(j)+1 for j in joints], [joints[j]["metrics"][key]["mean"] for j in joints], "o-", label=mode_labels[mode])
            ax.set(xlabel="Joint ordinal (dataset index + 1)", ylabel=f"Mean {labels[key]}",xticks=[int(j)+1 for j in joints])
            ax.grid(alpha=.25); ax.legend(fontsize=7)
    save(fig, "per_joint_metrics")
    fig, axes = plt.subplots(1, len(summary), figsize=(6*len(summary), 5), squeeze=False)
    for ax, (mode, groups) in zip(axes.flat, summary.items()):
        counts = np.array([[groups["overall"]["confusion"][g][p] for p in KINDS] for g in KINDS])
        ax.imshow(counts, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(counts[i, j]), ha="center", va="center",color='white' if counts[i,j]>counts.max()/2 else 'black')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=KINDS, yticklabels=KINDS,
               xlabel="Predicted joint type", ylabel="GT joint type", title=f"{mode_labels[mode]} (fit failures: {groups['overall']['fit_failures']})")
    save(fig, "joint_type_confusion")
    fig, axes = plt.subplots(1, len(summary), figsize=(7*len(summary), 5), squeeze=False)
    for ax, mode in zip(axes.flat, summary):
        subset = [r for r in rows if r['mode'] == mode and r.get('gt_path_pixels') is not None]
        points = ax.scatter([r['gt_path_pixels'] for r in subset], [r['ADE_1_pixels'] for r in subset],
                            c=[r['joint_ordinal'] for r in subset], s=12, alpha=.6)
        ax.set(xscale="symlog", yscale="symlog", xlabel="Mean complete GT track path (model pixels)",
               ylabel="ADE (model pixels)", title=mode)
        fig.colorbar(points, ax=ax, label="Joint ordinal"); ax.grid(alpha=.2)
    save(fig, "motion_vs_error")


def render_examples(out, rows):
    import matplotlib.pyplot as plt
    target = out / "examples"
    target.mkdir(exist_ok=True)
    for mode in sorted({r['mode'] for r in rows}):
        ordered = sorted([r for r in rows if r['mode'] == mode and not r.get('skipped')], key=lambda r:r['ADE_1_pixels'])
        if not ordered:
            continue
        for name, row in (("best", ordered[0]), ("median", ordered[len(ordered)//2]), ("worst", ordered[-1])):
            with np.load(out / "predictions" / f"{row['pair_index']:04d}_{mode}.npz", allow_pickle=False) as a:
                image = ((a['image'].transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)
                gt, pred, valid = a['ground_truth'], a['predictions'][0], a['valid']
            size = image.shape[0]-1
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax in axes:
                ax.imshow(image); ax.axis('off')
            for track in range(gt.shape[1]):
                axes[0].plot(gt[:,track,0][valid[:,track]]*size, gt[:,track,1][valid[:,track]]*size, '-o', ms=3)
                axes[1].plot(pred[:,track,0]*size, pred[:,track,1]*size, '-o', ms=3)
            axes[0].set_title('Ground truth'); axes[1].set_title('Prediction: hypothesis 0')
            fig.suptitle(f"{mode} / {name}: scene {row['scene_index']}, joint {row['joint_ordinal']} | ADE {row['ADE_1_pixels']:.2f} px")
            fig.tight_layout(); fig.savefig(target / f"{mode}_{name}.png", dpi=160); plt.close(fig)
