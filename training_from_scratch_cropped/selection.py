"""Revolute-only validation sampled to match eligible population marginals."""
from collections import Counter
import json
import numpy as np

from .traversal import digest, inventory

KINDS = {0: "revolute", 1: "prismatic", 4: "fixed"}
RULES = dict(version=2, split="val", count=100, joint_types=["revolute"], minimum_initial_points=4,
             minimum_fitting_observations=8, fitting_states=[2, 3, 4],
             algorithm="proportional_object_quotas_marginal_deficits_scene_diversity_v2",
             distribution_reference="eligible_scoreable_revolute_validation_pairs",
             object_coverage="at_least_one_per_object_when_count_permits",
             quantile_bins=4, predictions_used=False)
FIELDS = ("object_id", "category", "joint_ordinal", "chain_depth", "joint_type",
          "motion_bin", "size_bin", "magnification_bin", "viewpoint")
RULES["stratification_fields"] = list(FIELDS)


def selection_spec(dataset, seed):
    return dict(rules=RULES, seed=seed, inventory=inventory(dataset)["sha256"],
                sample=dict(points=dataset.points_per_link, image_size=dataset.image_size,
                            crop_padding=dataset.crop_padding, overlay_alpha=dataset.overlay_alpha,
                            fps_seed=dataset.seed))


def select_validation(dataset, path, *, seed=42):
    if dataset.training or dataset.scene_dir.parent.name != "val":
        raise ValueError("Checkpoint selection requires validation split")
    spec = selection_spec(dataset,seed)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["spec"] != spec or saved["sha256"] != digest({k:v for k,v in saved.items() if k != "sha256"}):
            raise ValueError("Validation manifest or source/configuration changed")
        validate_cohort(saved)
        return saved
    rows, excluded = [], Counter()
    by_scene = {}
    for i, (scene, pair) in enumerate(dataset.examples):
        by_scene.setdefault(scene, []).append((i, pair))
    for scene, examples in by_scene.items():
        with np.load(dataset.files[scene], allow_pickle=False) as a:
            meta = json.loads(str(a["metadata_json"].item()))
            js, parents, children = a["joint_indices"], a["joint_parent_link_indices"], a["joint_child_link_indices"]
            types, q = a["joint_types"], a["joint_positions"]
            # joint_positions is indexed by movable_joint_indices, whereas
            # joint_indices also contains fixed edges.
            movable = a["movable_joint_indices"]
            motion_by_joint = joint_motion(js, movable, q)
            axes = a["joint_axes_world"][0]
            seg = a["link_segmentation"][0]
            depths = {int(x):0 for x in set(a["link_indices"]) - set(children)}
            while len(depths) < len(a["link_indices"]):
                for p,c in zip(parents, children):
                    if int(p) in depths:
                        depths[int(c)] = depths[int(p)]+1
        for index, pair in examples:
            jp = int(np.flatnonzero(js == pair[2])[0])
            kind = KINDS.get(int(types[jp]), "other")
            if kind != "revolute":
                excluded[f"joint_type_{kind}"] += 1
                continue
            if not np.isfinite(axes[jp]).all() or np.linalg.norm(axes[jp]) < 1e-12:
                excluded["invalid_ground_truth_axis"] += 1
                continue
            sample = dataset[index]
            pv = sample["point_valid"].numpy() & sample["source_position_valid"][0].numpy()
            valid = sample["source_position_valid"].numpy()[:, pv]
            if pv.sum() < 4 or valid[2:5].sum() < 8 or not valid[1:].any():
                excluded["insufficient_correspondences"] += 1
                continue
            crop = sample["crop_bounds"].tolist()
            obj, camera = meta.get("object", {}), meta.get("camera", {})
            rows.append(dict(index=index, scene=scene, path=str(dataset.files[scene].resolve()),
                parent=pair[0], child=pair[1], joint_index=pair[2], joint_ordinal=jp+1,
                chain_depth=depths[pair[1]], joint_type=kind,
                object_id=str(obj.get("object_id", "unknown")), category=str(obj.get("category", "unknown")),
                motion=motion_by_joint[pair[2]], part_size=float((seg == pair[1]).mean()),
                magnification=float(min(seg.shape)/crop[2]),
                viewpoint=str((camera.get("azimuth_deg"), camera.get("elevation_deg")))))
    if len(rows) < 100:
        raise ValueError(f"Need exactly 100 unique scoreable revolute validation pairs; only {len(rows)} available")
    edges = {}
    for source, target in (("motion", "motion_bin"), ("part_size", "size_bin"), ("magnification", "magnification_bin")):
        boundaries = np.unique(np.quantile([r[source] for r in rows], [.25, .5, .75]))
        edges[target] = boundaries.tolist()
        for row in rows:
            row[target] = int(np.searchsorted(boundaries, row[source], side="right"))
    chosen, quotas = choose_representative(rows, count=100, seed=seed)
    scenes = Counter(r["scene"] for r in chosen)
    objects = Counter(r["object_id"] for r in chosen)
    coverage = {f: dict(available=dict(Counter(str(r[f]) for r in rows)),
                       selected=dict(Counter(str(r[f]) for r in chosen))) for f in FIELDS}
    constraints = [f"{f}: only {len(coverage[f]['available'])} distinct available values"
                   for f in FIELDS if f != "joint_type" and len(coverage[f]["available"]) < 2]
    if len(objects) < len({r["object_id"] for r in rows}):
        constraints.append("100 slots cannot cover every available object identity")
    if max(scenes.values()) > 1:
        constraints.append("Some scenes repeat to satisfy object proportions or limited scene coverage")
    distribution = distribution_report(rows, chosen)
    manifest = dict(spec=spec, pairs=chosen, eligible=len(rows), excluded=dict(excluded),
                    quantile_edges=edges, coverage=coverage, constraints=constraints,
                    object_quotas=quotas, distribution=distribution,
                    scene_count=len(scenes), object_count=len(objects),
                    axis_cohort=[r["index"] for r in chosen])
    validate_cohort(manifest)
    manifest["sha256"] = digest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)
    return manifest


def joint_motion(joint_indices, movable_indices, positions):
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != len(movable_indices):
        raise ValueError("joint_positions must be indexed by movable_joint_indices")
    mapping = {int(j):float(abs(positions[-1,i]-positions[0,i])) for i,j in enumerate(movable_indices)}
    return {int(j):mapping.get(int(j),0.) for j in joint_indices}


def validate_cohort(manifest):
    pairs = manifest["pairs"]
    indices = [r["index"] for r in pairs]
    if (len(indices) != 100 or len(set(indices)) != 100
            or any(r["joint_type"] != "revolute" for r in pairs)
            or manifest["axis_cohort"] != indices):
        raise ValueError("Validation requires exactly 100 unique revolute pairs, all in the axis cohort")


def proportional_quotas(population, count, rng):
    """Largest remainder rounding, with one per object when slots permit."""
    keys = sorted(population)
    total = sum(population.values())
    expected = {k:count*population[k]/total for k in keys}
    quota = {k:int(np.floor(expected[k])) for k in keys}
    priority = dict(zip(keys, rng.random(len(keys))))
    order = sorted(keys,key=lambda k:(expected[k]-quota[k],priority[k]),reverse=True)
    for k in order[:count-sum(quota.values())]:
        quota[k] += 1
    if len(keys) <= count:
        for k in keys:
            if quota[k] == 0:
                donor = max((v for v in keys if quota[v] > 1),
                            key=lambda v:(quota[v]-expected[v],priority[v]))
                quota[donor] -= 1
                quota[k] = 1
    return quota


def choose_representative(rows, *, count, seed):
    """Match population frequencies instead of giving rare groups equal weight.

    Exact proportional object quotas prevent arbitrary concentration. Prefer
    unused scenes, then minimize the incremental Pearson discrepancy from the
    desired marginal counts. All decisions use source metadata, never errors.
    """
    if len(rows) < count or count < 1:
        raise ValueError("Insufficient population for requested validation count")
    rng = np.random.default_rng(seed)
    quotas = proportional_quotas(Counter(r["object_id"] for r in rows),count,rng)
    priority = rng.random(len(rows))
    population = {f:Counter(r[f] for r in rows) for f in FIELDS}
    targets = {f:{v:count*n/len(rows) for v,n in population[f].items()} for f in FIELDS}
    counts = {f:Counter() for f in FIELDS}
    scenes, objects, chosen = Counter(), Counter(), []
    remaining = set(range(len(rows)))
    for _ in range(count):
        candidates = [i for i in remaining if objects[rows[i]["object_id"]] < quotas[rows[i]["object_id"]]]
        def rank(i):
            r = rows[i]
            cost = sum((2*counts[f][r[f]]+1-2*targets[f][r[f]])/max(targets[f][r[f]],1.)
                       for f in FIELDS)
            return (-scenes[r["scene"]],-cost,priority[i])
        i = max(candidates,key=rank)
        remaining.remove(i)
        r = rows[i]
        chosen.append(r)
        scenes[r["scene"]] += 1
        objects[r["object_id"]] += 1
        for f in FIELDS:
            counts[f][r[f]] += 1
    return chosen,quotas


def distribution_report(population, selected):
    """Expose representativeness and missing strata, rather than assume it."""
    result = {}
    for f in FIELDS:
        available = Counter(str(r[f]) for r in population)
        chosen = Counter(str(r[f]) for r in selected)
        expected = {v:len(selected)*n/len(population) for v,n in available.items()}
        result[f] = dict(expected_selected=expected,
            available_fraction={v:n/len(population) for v,n in available.items()},
            selected_fraction={v:chosen[v]/len(selected) for v in available},
            total_variation_distance=float(.5*sum(abs(chosen[v]/len(selected)-n/len(population))
                                                  for v,n in available.items())),
            uncovered_values=[v for v in available if chosen[v] == 0])
    for f in ("motion","part_size","magnification"):
        result[f] = dict(quantiles=[0.,.25,.5,.75,1.],
            available_quantiles=np.quantile([r[f] for r in population],[0,.25,.5,.75,1]).tolist(),
            selected_quantiles=np.quantile([r[f] for r in selected],[0,.25,.5,.75,1]).tolist())
    return result


def main(argv=None):
    """Build/review the real validation cohort without loading a model."""
    import argparse
    from pathlib import Path
    import torch
    from .dataset import ArticulatedDataset
    from .paths import ROOT, artifact_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("data/Robotic_arm"))
    parser.add_argument("--manifest",type=Path,default=ROOT/"manifests"/"validation_100_revolute.json")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--image-size",type=int,default=384)
    parser.add_argument("--crop-padding",type=float,default=.1)
    parser.add_argument("--overlay-alpha",type=float,default=.85)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    dataset = ArticulatedDataset(args.data_root,split="val",training=False,seed=args.seed,
        image_size=args.image_size,crop_padding=args.crop_padding,overlay_alpha=args.overlay_alpha)
    manifest = select_validation(dataset,artifact_path(args.manifest),seed=args.seed)
    print(json.dumps(dict(manifest=str(args.manifest),pairs=len(manifest["pairs"]),
        axis_cohort=len(manifest["axis_cohort"]),eligible_revolute_pairs=manifest["eligible"],
        scenes=manifest["scene_count"],objects=manifest["object_count"],excluded=manifest["excluded"],
        total_variation={f:manifest["distribution"][f]["total_variation_distance"] for f in FIELDS},
        constraints=manifest["constraints"]),sort_keys=True),flush=True)
    return manifest


if __name__ == "__main__":
    main()
