"""Independent metric and cohort audit for the frozen 50-scene transfer run."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from testing_joint_fitting.experiment import write
from .evaluate import ROOT, DEFAULT_EVALUATION, load_validation, frozen_sources, freeze_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    output = ROOT/"results"
    cases, cohort = load_validation(args.evaluation, output)
    cases = {case["pair_index"]:case for case in cases}
    protocol = freeze_protocol()
    assert frozen_sources() == protocol["algorithm_sources"]
    overlap = json.loads((ROOT/"object_inventory.json").read_text())["overlap"]
    assert not overlap["val_objects_shared_with_test"] and not overlap["val_objects_shared_with_train"]
    errors = dict(axis=0., pivot=0., rotation=0., reprojection=0.)
    total = 0; nonpositive_depths = 0; failure_count = 0
    for folder in ("variants", "controls"):
        for path in sorted((output/folder).glob("*.json")):
            result = json.loads(path.read_text())
            assert result["algorithm_sources"] == protocol["algorithm_sources"]
            assert {row["pair_index"] for row in result["rows"]} == set(cases)
            for row in result["rows"]:
                total += 1
                if row["failed"]:
                    assert row["axis_error_with_failure_penalty_deg"] == 90.
                    failure_count += 1; continue
                case = cases[row["pair_index"]]; data = case["data"]
                parameters = np.asarray(row["params"])
                assert np.isfinite(parameters).all() and len(parameters)==9
                axis = parameters[:3]/np.linalg.norm(parameters[:3])
                truth = case["gt_axis"]/np.linalg.norm(case["gt_axis"])
                angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, truth)), abs(axis@truth)))
                errors["axis"] = max(errors["axis"], abs(angle-row["axis_error_deg"]))
                if case["gt_pivot"] is not None:
                    offset = case["gt_pivot"]-parameters[3:6]
                    distance = np.linalg.norm(offset-(offset@axis)*axis)
                    errors["pivot"] = max(errors["pivot"], abs(distance-row["pivot_distance"]))
                rotations, image_squared = [], []
                for slot in range(3):
                    rotation = cv2.Rodrigues(axis*parameters[6+slot])[0]
                    u, _, vt = np.linalg.svd(case["gt_rotations"][slot])
                    proper = u@np.diag([1., 1., np.linalg.det(u@vt)])@vt
                    rotations.append(np.degrees(np.arccos(np.clip((np.trace(rotation@proper.T)-1)/2, -1., 1.))))
                    points = (data.xyz-parameters[3:6])@rotation.T+parameters[3:6]
                    camera = points@data.e[:3, :3].T+data.e[:3, 3]
                    mask = data.valid[slot+2]
                    nonpositive_depths += int(np.sum(camera[mask, 2]<=0))
                    homogeneous = camera@data.k.T
                    uv = homogeneous[:, :2]/homogeneous[:, 2:]
                    target = case["gt_uv"][slot+2] if folder=="controls" else data.uv[slot+2]
                    image_squared.extend(np.sum((uv[mask]-target[mask])**2, axis=1))
                errors["rotation"] = max(errors["rotation"], abs(float(np.mean(rotations))-row["rotation_error_deg"]))
                errors["reprojection"] = max(errors["reprojection"], abs(float(np.sqrt(np.mean(image_squared)))-row["rmse_px"]))
    assert total == 8*len(cases)
    assert errors["axis"]<1e-5 and errors["pivot"]<1e-10 and errors["rotation"]<.001 and errors["reprojection"]<1e-8, errors
    assert nonpositive_depths == 0, nonpositive_depths
    result = dict(status="passed", scenes=50, pairs=len(cases), audited_rows=total, failures=failure_count,
        maximum_metric_disagreement=errors, nonpositive_scored_depths=nonpositive_depths,
        algorithm_hashes_unchanged=True, validation_object_ids_disjoint_from_test_and_train=True,
        inference_audit=json.loads((args.evaluation/"audit.json").read_text()))
    write(output/"audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
