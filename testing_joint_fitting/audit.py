"""Verify development coverage, baseline parity, parameter frames, and inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .experiment import ROOT, write
from training_from_scratch.full_evaluation.metrics import summarize
from finetuning.joint_parame_estim.run import modeled_points


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def main():
    out=ROOT/"results"
    subset=json.loads((out/"subset.json").read_text())
    evaluation=Path(subset["evaluation"])
    baseline=json.loads((out/"without_poke/baseline.json").read_text())
    original={r["pair_index"]:r for r in baseline["rows"]}
    keys=("forced_revolute_axis_error_deg","forced_revolute_kinematic_pivot_distance","rotation_geodesic_deg")
    maxima={key:0. for key in keys}
    for index,row in original.items():
        previous=json.loads((evaluation/"records"/f"{index:04d}_without_poke.json").read_text())
        assert row["predicted_type"]==previous["predicted_type"]
        for key in keys:
            if row.get(key) is not None:
                maxima[key]=max(maxima[key],abs(row[key]-previous[key]))
    assert all(value<1e-10 for value in maxima.values()),maxima
    checked=[]
    frozen_type={"refined","contact_prior","contact_restart","contact_tracks","shape_mdl","shape_only",
                 "early_contact","ensemble_weighted","track_stability","pnp_global"}
    for path in sorted((out/"without_poke").glob("*.json")):
        result=json.loads(path.read_text());rows=result["rows"]
        assert len(rows)==216
        assert {r["pair_index"] for r in rows}==set(original)
        assert len({r["scene_index"] for r in rows})==80
        assert summarize(rows)==result["summary"],path
        assert result["summary"]["metrics"][keys[0]]["n"]==204
        for row in rows:
            if path.stem in frozen_type:
                assert row["predicted_type"]==original[row["pair_index"]]["predicted_type"]
            for kind in ("revolute","prismatic"):
                parameters=row.get(f"fit_parameters_{kind}")
                if parameters is not None:
                    assert len(parameters)==(9 if kind=="revolute" else 6)
                    assert abs(np.linalg.norm(parameters[:3])-1)<1e-6
                    assert np.isfinite(parameters).all()
        checked.append(path.stem)
    candidate=json.loads((out/"without_poke/contact_restart.json").read_text())
    stability=json.loads((out/"without_poke/track_stability.json").read_text())
    for a,b in zip(candidate["rows"],stability["rows"]):
        for key in keys:
            assert a.get(key)==b.get(key)
    files={evaluation/"run.json"}
    cameras={}
    candidate_rows={r["pair_index"]:r for r in candidate["rows"]}
    minimum_camera_depth=float("inf")
    for index in original:
        for mode in ("without_poke","with_poke"):
            for directory,suffix in (("records","json"),("predictions","npz")):
                path=evaluation/directory/f"{index:04d}_{mode}.{suffix}"
                if path.exists():files.add(path)
        with np.load(evaluation/"predictions"/f"{index:04d}_without_poke.npz",allow_pickle=False) as a:
            source_file=evaluation/str(a["source_file"])
            files.add(source_file)
            if source_file not in cameras:
                with np.load(source_file,allow_pickle=False) as source:cameras[source_file]=source["camera_extrinsic"]
            row=candidate_rows[index];kind=row["predicted_type"]
            points=modeled_points(kind,np.array(row[f"fit_parameters_{kind}"]),a["reference_world"],
                                  [(np.eye(3),np.zeros(3),True)]*3)
            e=cameras[source_file]
            depth=(points@e[:3,:3].T+e[:3,3])[:,:,2][a["valid"][2:]]
            minimum_camera_depth=min(minimum_camera_depth,float(depth.min()))
    assert minimum_camera_depth>0
    manifest={str(path.relative_to(evaluation)):dict(size=path.stat().st_size,sha256=sha256(path)) for path in sorted(files)}
    target=out/"inputs_sha256.json"
    if target.exists():
        assert json.loads(target.read_text())==manifest,"Saved input file changed"
    else:
        write(target,manifest)
    result=dict(passed=True,scenes=80,scored_pairs=216,forced_revolute_cohort=204,variants=checked,
                baseline_maximum_absolute_difference=maxima,unchanged_type_variants=sorted(frozen_type),
                input_files_hashed=len(manifest),input_hashes="inputs_sha256.json",
                candidate_minimum_valid_camera_depth=minimum_camera_depth,
                stability_diagnostic_keeps_candidate_outputs_identical=True)
    write(out/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
