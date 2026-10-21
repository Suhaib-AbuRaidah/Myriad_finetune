"""Scoring-only contact diagnostics; these outputs never select fitted candidates."""
import json
import numpy as np
from finetuning.joint_parame_estim.run import axis_error
from testing_joint_fitting.experiment import write
from training_from_scratch.full_evaluation.metrics import distribution
from .data import load, OUT
from .geometry import mesh_contact


def main():
    cases = load()
    result = {}
    for count, symmetric in ((16, False), (32, False), (64, False), (128, False), (32, True), (64, True)):
        rows = []
        for case in cases:
            data = case["data"]
            contact = mesh_contact(data.parent, data.child, count, symmetric)
            errors = [axis_error(contact["axes"][:, i], case["gt_axis"]) for i in range(3)]
            normals = np.r_[data.parent_normals, data.child_normals]
            rows.append(dict(pair_index=case["pair_index"], scene_index=case["scene_index"],
                joint_ordinal=case["joint_ordinal"], errors=errors,
                oracle_closest_error=min(errors), linearity=contact["linearity"], planarity=contact["planarity"],
                eigenvalues=contact["eigenvalues"],
                # Merely checks unit normals; no direction is picked using these errors.
                normal_length_median=float(np.median(np.linalg.norm(normals, axis=1)))))
        name = f"count{count}_symmetric{int(symmetric)}"
        result[name] = dict(rows=rows, summary=dict(
            normal=distribution(row["errors"][0] for row in rows),
            middle=distribution(row["errors"][1] for row in rows),
            major=distribution(row["errors"][2] for row in rows),
            oracle_closest=distribution(row["oracle_closest_error"] for row in rows),
            linearity=distribution(row["linearity"] for row in rows),
            planarity=distribution(row["planarity"] for row in rows)))
        print(json.dumps(dict(geometry=name, summary=result[name]["summary"])), flush=True)
    write(OUT/"geometry_diagnostics.json", result)


if __name__ == "__main__": main()
