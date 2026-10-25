"""Independent saved-fit checks for every development and transfer configuration."""
import hashlib
import json

import cv2
import numpy as np

from testing_joint_fitting.experiment import write
from ..data import load
from .run import OUT, snapshot


def audit_cases(cases, cohort):
    cases = {c["pair_index"]: c for c in cases}
    maxima = dict(axis_deg=0., pivot=0., rotation_deg=0., reprojection_px=0., unit_axis=0.)
    total = failures = 0
    minimum_depth = float("inf")
    files = {}
    for folder in ("variants", "controls"):
        for path in sorted((OUT/cohort/folder).glob("*.json")):
            item = json.loads(path.read_text())
            assert len(item["rows"]) == len(cases)
            assert {r["pair_index"] for r in item["rows"]} == set(cases)
            assert item["control"] == (folder == "controls")
            assert item["code_sha256"] == snapshot()
            for row in item["rows"]:
                if row["failed"]:
                    failures += 1
                    assert row["axis_error_with_failure_penalty_deg"] == 90.
                    continue
                c = cases[row["pair_index"]]
                d = c["data"]
                p = np.asarray(row["params"], float)
                assert p.shape == (9,) and np.isfinite(p).all()
                a = p[:3]/np.linalg.norm(p[:3])
                gt = c["gt_axis"]/np.linalg.norm(c["gt_axis"])
                axis = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, gt)), abs(a@gt)))
                maxima["axis_deg"] = max(maxima["axis_deg"], abs(axis-row["axis_error_deg"]))
                maxima["unit_axis"] = max(maxima["unit_axis"], abs(np.linalg.norm(p[:3])-1))
                if c["gt_pivot"] is not None:
                    offset = c["gt_pivot"]-p[3:6]
                    pivot = np.linalg.norm(offset-(offset@a)*a)
                    maxima["pivot"] = max(maxima["pivot"], abs(pivot-row["pivot_distance"]))
                uv = c["gt_uv"] if item["control"] else d.uv
                rotations, squared = [], []
                for index, state in enumerate((2, 3, 4)):
                    r = cv2.Rodrigues(a*p[6+index])[0]
                    u, _, vt = np.linalg.svd(c["gt_rotations"][index])
                    truth = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
                    relative = r@truth.T
                    rotations.append(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
                    world = (d.xyz-p[3:6])@r.T+p[3:6]
                    camera = world@d.e[:3, :3].T+d.e[:3, 3]
                    image = camera@d.k.T
                    image = image[:, :2]/image[:, 2:]
                    valid = d.valid[state]
                    minimum_depth = min(minimum_depth, float(camera[valid, 2].min()))
                    squared.extend(np.sum((image[valid]-uv[state, valid])**2, axis=1))
                maxima["rotation_deg"] = max(maxima["rotation_deg"],
                    abs(np.mean(rotations)-row["rotation_error_deg"]))
                maxima["reprojection_px"] = max(maxima["reprojection_px"],
                    abs(np.sqrt(np.mean(squared))-row["rmse_px"]))
                total += 1
            files[str(path.relative_to(OUT))] = len(item["rows"])
    assert maxima["axis_deg"] < 1e-5, maxima
    assert maxima["pivot"] < 1e-10, maxima
    assert maxima["rotation_deg"] < .001, maxima
    assert maxima["reprojection_px"] < 1e-7, maxima
    assert maxima["unit_axis"] < 1e-6, maxima
    assert minimum_depth > 0
    return dict(fits=total, failures=failures, maximum_metric_disagreement=maxima,
        minimum_scored_depth=minimum_depth, files=files)


def main():
    snapshots = []
    for directory in (OUT/"code").iterdir():
        files = {str(p.relative_to(directory)): p.read_text() for p in directory.rglob("*.py")}
        assert hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == directory.name
        snapshots.append(directory.name)
    result = dict(status="passed", snapshots_verified=snapshots,
        development=audit_cases(load(), "development"))
    if (OUT/"validation/protocol.json").exists():
        from .transfer import load_cases
        cases, _ = load_cases()
        result["validation"] = audit_cases(cases, "validation")
    write(OUT/"audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
