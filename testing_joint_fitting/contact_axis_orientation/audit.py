"""Independently recompute reported metrics and check geometric validity."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from .data import load, OUT
from testing_joint_fitting.experiment import write


def main():
    cases = {case["pair_index"]: case for case in load()}
    residuals = dict(axis=0., pivot=0., rotation=0., unit_axis=0.)
    negative_depth = 0; total = 0
    counts = {}
    for folder in ("variants", "controls"):
        for path in sorted((OUT/folder).glob("*.json")):
            result = json.loads(path.read_text())
            assert len(result["rows"]) == 204
            assert {row["pair_index"] for row in result["rows"]} == set(cases)
            for row in result["rows"]:
                if row["failed"]: continue
                case = cases[row["pair_index"]]; data = case["data"]
                p = np.asarray(row["params"], float)
                assert np.isfinite(p).all() and len(p) == 9
                axis = p[:3]/np.linalg.norm(p[:3])
                truth = case["gt_axis"]/np.linalg.norm(case["gt_axis"])
                angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(axis, truth)), abs(axis@truth)))
                residuals["axis"] = max(residuals["axis"], abs(angle-row["axis_error_deg"]))
                residuals["unit_axis"] = max(residuals["unit_axis"], abs(np.linalg.norm(p[:3])-1))
                if case["gt_pivot"] is not None:
                    offset = case["gt_pivot"]-p[3:6]
                    distance = np.linalg.norm(offset-(offset@axis)*axis)
                    residuals["pivot"] = max(residuals["pivot"], abs(distance-row["pivot_distance"]))
                errors = []
                for slot in range(3):
                    rotation = cv2.Rodrigues(axis*p[6+slot])[0]
                    # Polar-project float32-derived GT rotations before using
                    # the independent trace-angle definition of SO(3) distance.
                    u, _, vt = np.linalg.svd(case["gt_rotations"][slot])
                    truth_rotation = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
                    relative = rotation@truth_rotation.T
                    cosine = np.clip((np.trace(relative)-1)/2, -1, 1)
                    errors.append(np.rad2deg(np.arccos(cosine)))
                    points = (data.xyz-p[3:6])@rotation.T+p[3:6]
                    depth = (points@data.e[:3, :3].T+data.e[:3, 3])[:, 2]
                    negative_depth += int(np.sum(depth[data.valid[slot+2]] <= 0))
                residuals["rotation"] = max(residuals["rotation"], abs(np.mean(errors)-row["rotation_error_deg"]))
                total += 1
            counts[str(path.relative_to(OUT))] = len(result["rows"])
    assert residuals["axis"] < 1e-5, residuals
    assert residuals["pivot"] < 1e-10, residuals
    assert residuals["rotation"] < .001, residuals
    assert residuals["unit_axis"] < 1e-6, residuals
    assert negative_depth == 0, negative_depth
    repo = Path(__file__).resolve().parents[2]
    dependencies = ("testing_joint_fitting/geometric.py", "testing_joint_fitting/experiment.py",
        "testing_joint_fitting/axis_motion/common.py", "finetuning_pair/analyze_joints.py",
        "finetuning/joint_parame_estim/run.py", "training_from_scratch/full_evaluation/metrics.py")
    result = dict(audited_fits=total, files=counts, maximum_metric_disagreement=residuals,
        nonpositive_scored_depths=negative_depth,
        dependencies={name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in dependencies})
    write(OUT/"audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__": main()
