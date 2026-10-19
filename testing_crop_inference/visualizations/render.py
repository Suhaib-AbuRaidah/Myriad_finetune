"""Render saved crop-evaluation predictions; no inference or fitting is run.

python -B testing_crop_inference/visualizations/render.py
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/myriad_crop_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

BLUE, PURPLE, GREEN = "#0072B2", "#A238B5", "#009E73"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def records(root):
    variants = {}
    for variant in ("original", "crop_10"):
        rows = [json.loads(p.read_text()) for p in sorted((root/"records").glob(f"*_{variant}.json"))]
        variants[variant] = {r["pair_index"]: r for r in rows if not r["skipped"]}
    assert set(variants["original"]) == set(variants["crop_10"])
    return variants


def choose(rows):
    candidates = [i for i, c in rows["crop_10"].items() if c["gt_type"] == "revolute"
        and c.get("forced_revolute_axis_error_deg") is not None
        and rows["original"][i].get("forced_revolute_axis_error_deg") is not None
        and c["complete_tracks"] >= 6 and (c.get("gt_path_pixels") or 0) >= 10]
    inside = [i for i in candidates if rows["crop_10"][i]["gt_outside_crop_fraction"] == 0]
    def delta(i):
        return rows["crop_10"][i]["ADE_1_pixels"] - rows["original"][i]["ADE_1_pixels"]
    def median(ids):
        return sorted(ids, key=lambda i: (delta(i), i))[len(ids)//2]
    selected = [dict(pair_index=median([i for i in inside if delta(i) > 0]),
                     label="Typical regression", selection="Median ADE increase among eligible worsened pairs"),
                dict(pair_index=median([i for i in inside if delta(i) < 0]),
                     label="Typical improvement", selection="Median ADE change among eligible improved pairs")]
    for joint in (1, 2, 3, 4):
        ids = [i for i in inside if rows["crop_10"][i]["joint_ordinal"] == joint]
        if ids:
            selected.append(dict(pair_index=median(ids), label=f"Joint {joint}: representative example",
                selection=f"Median ADE change among eligible joint-{joint} pairs"))
    outside = [i for i in candidates if rows["crop_10"][i]["gt_outside_crop_fraction"] > 0]
    if outside:
        selected.append(dict(pair_index=median(outside), label="GT motion leaves the crop",
            selection="Median ADE change among eligible pairs with original-valid future GT outside the crop"))
    return dict(eligibility="GT revolute; both revolute candidates available; >=6 complete tracks; mean GT path >=10 original-view pixels",
                inside_examples="Except the explicitly labelled boundary example, every originally valid future GT remains inside the crop",
                seed_or_tuning="No manual case picking, new model runs, or algorithm changes", examples=selected)


def overview(root, out, rows):
    s = json.loads((root/"summary.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Original image vs parent–child crop — full test1 evaluation", fontsize=18, y=.98)
    fig.text(.5, .935, "320 scenes · 877 scored pairs · same checkpoint and existing joint fitter · lower errors are better",
             ha="center", color="#444444")
    specs = [("Trajectory ADE", "ADE_1_pixels", "Original-view pixels", "mean"),
             ("Axis orientation error", "forced_revolute_axis_error_deg", "Degrees · 825 matched revolute pairs", "mean"),
             ("Pivot distance", "forced_revolute_kinematic_pivot_distance", "World units · median · 825 matched pairs", "median")]
    for ax, (title, key, unit, stat) in zip(axs[0], specs):
        values = [s["all_pairs"][v]["metrics"][key][stat] for v in ("original", "crop_10")]
        bars = ax.bar([0, 1], values, color=[BLUE, PURPLE], width=.57)
        ax.set_xticks([0, 1], ["Original", "Cropped"])
        ax.set(title=title, ylabel=unit, ylim=(0, max(values)*1.24))
        ax.bar_label(bars, labels=[f"{v:.3f}" if max(values)<1 else f"{v:.2f}" for v in values], padding=5, fontsize=13)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    ax = axs[1, 0]
    ids = sorted(rows["original"])
    x, y = [np.array([rows[v][i]["ADE_1_pixels"] for i in ids]) for v in ("original", "crop_10")]
    ax.scatter(x, y, s=11, c=np.where(y>x, PURPLE, BLUE), alpha=.45, edgecolor="none")
    limit = max(x.max(), y.max())*1.04
    ax.plot([0, limit], [0, limit], color="#555", ls="--", lw=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Original ADE (pixels)", ylabel="Cropped ADE (pixels)",
           title="Each dot is one pair")
    ax.text(.04, .94, "Above diagonal = crop worse", transform=ax.transAxes, va="top", fontsize=10)
    ax.grid(alpha=.15)
    ax = axs[1, 1]
    joints = sorted(s["by_joint"], key=int)
    positions = np.arange(len(joints))
    for offset, variant, color, label in [(-.18, "original", BLUE, "Original"), (.18, "crop_10", PURPLE, "Cropped")]:
        vals = [s["by_joint"][j][variant]["metrics"]["forced_revolute_axis_error_deg"]["mean"] for j in joints]
        ax.bar(positions+offset, vals, width=.35, color=color, label=label)
    ax.set_xticks(positions, [f"Joint {j}" for j in joints])
    ax.set(title="Axis error by joint", ylabel="Mean orientation error (degrees)", ylim=(0, 75))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax = axs[1, 2]
    values = [100*s["all_pairs"][v]["type_accuracy"] for v in ("original", "crop_10")]
    bars = ax.bar([0, 1], values, color=[BLUE, PURPLE], width=.57)
    ax.set_xticks([0, 1], ["Original", "Cropped"])
    ax.set(title="Joint-type accuracy", ylabel="Percent · higher is better", ylim=(0, 105))
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=5, fontsize=13)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.5, .025, "All pixel errors are scored AFTER inverse cropping. Axis/pivot: revolute candidate for each GT-revolute pair, including type mistakes.\n"
             "Pivot metric: perpendicular GT kinematic-pivot-to-predicted-axis distance; units follow the dataset.",
             ha="center", fontsize=10, color="#444")
    fig.subplots_adjust(top=.87, bottom=.13, hspace=.45, wspace=.34)
    save(fig, out/"metrics_overview")


def tracks(ax, positions, valid, color, *, predicted=False):
    for track in range(positions.shape[1]):
        xy = np.where(valid[:, track, None], positions[:, track], np.nan)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.65, ls="--" if predicted else "-",
                path_effects=[pe.Stroke(linewidth=3.3, foreground="white", alpha=.9), pe.Normal()], zorder=4 if predicted else 3)
        ax.scatter(xy[0, 0], xy[0, 1], s=20, facecolors="white", edgecolors="#333", linewidths=.7, zorder=7)
        if valid[-1, track]:
            ax.scatter(*xy[-1], s=23, marker="^" if predicted else "s", c=color, edgecolors="white", linewidths=.5, zorder=6)


def example(root, out, rows, selection):
    index = selection["pair_index"]
    a, b = rows["original"][index], rows["crop_10"][index]
    with np.load(root/"predictions"/f"{index:04d}_original.npz", allow_pickle=False) as saved:
        original = {key: saved[key] for key in ("image", "ground_truth", "predictions", "valid")}
    with np.load(root/"predictions"/f"{index:04d}_crop_10.npz", allow_pickle=False) as saved:
        cropped = {key: saved[key] for key in ("image", "cropped_ground_truth", "cropped_predictions", "predictions", "valid")}
    np.testing.assert_array_equal(original["valid"], cropped["valid"])
    valid = original["valid"]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.7))
    title = f'{selection["label"]} · pair {index} · joint {a["joint_ordinal"]} · zoom {b["zoom"]:.2f}×'
    fig.suptitle(title, fontsize=17, y=.98)
    fig.text(.5, .91, f'Original: ADE {a["ADE_1_pixels"]:.2f} px · axis {a["forced_revolute_axis_error_deg"]:.1f}° · type {a.get("predicted_type", "failed")}     |     '
        f'Cropped: ADE {b["ADE_1_pixels"]:.2f} px · axis {b["forced_revolute_axis_error_deg"]:.1f}° · type {b.get("predicted_type", "failed")}',
        ha="center", fontsize=12)
    for ax, data in zip(axs, (original, cropped, original)):
        ax.imshow(np.moveaxis((data["image"]+1)/2, 0, -1).clip(0, 1), alpha=.65 if ax is axs[2] else 1)
        ax.set_aspect("equal")
    gt, pred_old, pred_new = original["ground_truth"]*383, original["predictions"][0]*383, cropped["predictions"][0]*383
    tracks(axs[0], gt, valid, GREEN)
    tracks(axs[0], pred_old, valid, BLUE, predicted=True)
    tracks(axs[1], cropped["cropped_ground_truth"]*383, valid, GREEN)
    tracks(axs[1], cropped["cropped_predictions"][0]*383, valid, PURPLE, predicted=True)
    tracks(axs[2], gt, valid, GREEN)
    tracks(axs[2], pred_old, valid, BLUE, predicted=True)
    tracks(axs[2], pred_new, valid, PURPLE, predicted=True)
    with np.load(root/"predictions"/f"{index:04d}_original.npz", allow_pickle=False) as saved:
        source_path = Path(str(saved["source_path"]))
    with np.load(source_path, allow_pickle=False) as source:
        height, width = source["depth"].shape[-2:]
    side = min(height, width)
    top, left = round((height-side)/2), round((width-side)/2)
    crop = b["crop"]
    # Rectangle denotes crop support edges, rather than pixel-center positions.
    x0 = (crop["left"]-left)*384/side-.5
    y0 = (crop["top"]-top)*384/side-.5
    axs[0].add_patch(Rectangle((x0, y0), crop["side"]*384/side, crop["side"]*384/side,
                    fill=False, edgecolor=PURPLE, lw=1.7, ls=":"))
    for ax in axs[:2]:
        ax.set(xlim=(-.5, 383.5), ylim=(383.5, -.5))
        ax.set_xticks([]); ax.set_yticks([])
    points = np.concatenate([array[valid] for array in (gt, pred_old, pred_new)])
    low, high = points.min(0), points.max(0)
    center = (low+high)/2
    view_side = max(48., float((high-low).max())*1.25)
    axs[2].set(xlim=(center[0]-view_side/2, center[0]+view_side/2),
               ylim=(center[1]+view_side/2, center[1]-view_side/2),
               xlabel="Original-view x (pixels)", ylabel="Original-view y (pixels)")
    for ax, label in zip(axs, ("Original input + prediction", "Cropped input + prediction", "Both predictions in the SAME coordinates")):
        ax.set_title(label, fontsize=11, pad=10)
    handles = [Line2D([], [], c=GREEN, lw=2, label="Ground truth"),
               Line2D([], [], c=BLUE, lw=2, ls="--", label="Original-image prediction"),
               Line2D([], [], c=PURPLE, lw=2, ls="--", label="Cropped-image prediction"),
               Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#333", ls="", label="Observed initial point")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .025, "Hypothesis 0 · GT endpoint: square; predicted endpoint: triangle · ADE always measured in original-view pixels.\n"
        f'Axis error uses each revolute candidate; predicted types are shown above. GT future states outside crop: {100*b["gt_outside_crop_fraction"]:.1f}%.',
        ha="center", fontsize=9, color="#444")
    fig.subplots_adjust(left=.025, right=.985, top=.82, bottom=.2, wspace=.23)
    name = f"pair_{index:04d}"
    save(fig, out/name)
    return dict(**selection, joint=a["joint_ordinal"], zoom=b["zoom"], image=f"{name}.png",
                original_ADE=a["ADE_1_pixels"], cropped_ADE=b["ADE_1_pixels"])


def axes_plot(out, rows, selections):
    chosen = [s for s in selections["examples"] if s["label"].startswith("Joint ")]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Fitted joint-axis orientations in world coordinates", fontsize=17, y=.98)
    fig.text(.5, .94, "Green = GT · Blue = original input · Purple = cropped input", ha="center")
    for slot, selection in enumerate(chosen):
        index = selection["pair_index"]
        a, b = rows["original"][index], rows["crop_10"][index]
        ax = fig.add_subplot(2, 2, slot+1, projection="3d")
        gt = np.asarray(a["gt_axis_world"], float)
        gt /= np.linalg.norm(gt)
        for vector, color, name in ((gt, GREEN, "GT"),
                (a["forced_revolute_axis_world"], BLUE, "Original"),
                (b["forced_revolute_axis_world"], PURPLE, "Crop")):
            direction = np.asarray(vector, float)
            direction /= np.linalg.norm(direction)
            direction *= 1 if direction @ gt >= 0 else -1
            ax.quiver(0, 0, 0, *direction, color=color, linewidth=2.7, arrow_length_ratio=.13)
            ax.text(*(direction*1.07), name, color=color, fontsize=9)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([-.5, 0, .5])
        ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.15, 1.15), xlabel="World x", ylabel="World y", zlabel="World z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-55)
        ax.set_title(f'Joint {a["joint_ordinal"]} · pair {index}\nAxis error: {a["forced_revolute_axis_error_deg"]:.1f}° → {b["forced_revolute_axis_error_deg"]:.1f}°', fontsize=11)
    fig.text(.5, .025, "Orientation only: directions are drawn from a common origin; pivot positions are NOT shown.\n"
             "Axes are sign-aligned to GT for display. These are revolute candidates, including cases with another predicted joint type.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.12, wspace=.05, hspace=.22)
    save(fig, out/"axis_orientations")


def gallery(out, rendered):
    def embed(name):
        return "data:image/png;base64,"+base64.b64encode((out/name).read_bytes()).decode()
    sections = [f'<section><h2>Overall metrics — full dataset</h2><img src="{embed("metrics_overview.png")}" alt="Overall crop versus original metrics"></section>']
    for r in rendered:
        sections.append(f'<section class="example" data-joint="{r["joint"]}"><h2>{html.escape(r["label"])}</h2>'
            f'<p>Selection: {html.escape(r["selection"])}. Pair {r["pair_index"]}; zoom {r["zoom"]:.2f}×.</p>'
            f'<img src="{embed(r["image"])}" alt="Trajectory comparison for pair {r["pair_index"]}"></section>')
    sections.append(f'<section><h2>3D axis directions</h2><img src="{embed("axis_orientations.png")}" alt="World axis directions"></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Myriad: original vs cropped inputs</title><style>
body{margin:0;background:#f3f5f8;color:#1a2430;font:16px/1.5 system-ui,sans-serif}main{max-width:1550px;margin:35px auto;padding:0 18px}h1{margin-bottom:8px}h2{font-size:21px}p{max-width:1050px;color:#465365}section{background:white;border:1px solid #dce1e8;border-radius:12px;padding:15px 20px;margin:24px 0}img{display:block;width:100%;height:auto}select{padding:8px 12px;border:1px solid #aeb8c5;border-radius:6px;font:inherit}header{position:sticky;top:0;background:#f3f5f8ed;backdrop-filter:blur(8px);padding:12px 0;border-bottom:1px solid #dce1e8;z-index:2}.legend{font-weight:600}.blue{color:#0072B2}.purple{color:#A238B5}.green{color:#009E73}
</style><main><h1>Myriad: original vs cropped inputs</h1><p>Checkpoint step 108,000; all 320 test1 scenes, 877 scored pairs. No new model inference or fitting was performed to make these figures. Aggregate plots use every scored pair; examples follow recorded, deterministic selection rules.</p>
<p class="legend"><span class="green">Green: ground truth</span> · <span class="blue">Blue: original-image prediction</span> · <span class="purple">Purple: cropped-image prediction</span></p><p>In every example, the right panel uses the same original-image coordinate system for both methods. Input images retain the model’s blue parent and orange child overlays. Curves use hypothesis 0.</p>
<header><label>Filter examples by joint: <select id="joint"><option value="all">All joints</option><option value="1">Joint 1</option><option value="2">Joint 2</option><option value="3">Joint 3</option><option value="4">Joint 4</option></select></label></header>'''
    page += "\n".join(sections)
    page += '''<script>document.querySelector('#joint').addEventListener('change',e=>{document.querySelectorAll('.example').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.joint!==e.target.value)})</script></main></html>'''
    (out/"index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("testing_crop_inference/results"))
    args = parser.parse_args()
    out = args.results/"visualizations"
    out.mkdir(exist_ok=True)
    rows = records(args.results)
    selections = choose(rows)
    (out/"selection.json").write_text(json.dumps(selections, indent=2)+"\n")
    overview(args.results, out, rows)
    rendered = [example(args.results, out, rows, s) for s in selections["examples"]]
    axes_plot(out, rows, selections)
    gallery(out, rendered)
    (out/"README.md").write_text("# Saved-evaluation visualizations\n\nOpen `index.html` for the self-contained gallery, with a joint filter. Each figure\nis also available as PNG and PDF. Green is GT, blue is original-input prediction,\nand purple is cropped-input prediction. Input role overlays remain blue for the\nparent and orange for the child. No new inference or joint fitting is performed.\n\nAll example ADE values use original-view pixels. Original and cropped input\npanels have different fields of view; the third panel compares both predictions\nin identical original coordinates. Only originally valid GT states are drawn;\nfuture targets outside the crop remain scored. White circles indicate supplied\ninitial positions, squares GT endpoints, and triangles predicted endpoints.\n\n`selection.json` records exact cohort/selection rules and pair IDs. Examples\nillustrate outcomes and do not replace the full-dataset aggregate plots.\nThe axis figure displays directions only from a common origin, not pivot points.\n\nReproduce from the repository root, in myriad_finetune:\n\n```bash\npython -B testing_crop_inference/visualizations/render.py\n```\n")
    print(json.dumps(dict(gallery=str(out/"index.html"), examples=rendered), indent=2))


if __name__ == "__main__":
    main()
