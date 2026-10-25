"""Select globally on development, subject to explicit preservation/bias guards."""
import json

from testing_joint_fitting.experiment import write
from .run import OUT, snapshot
from .solver import VARIANTS

# Fixed before the adaptive sweep completed. These apply to complete cohorts,
# never to a per-pair choice of fitted parameters.
GUARDS = dict(max_axis_mean_increase_deg=.25, min_axis_success_change=0.,
    max_rotation_mean_increase_deg=1., max_control_pivot_mean=.002,
    max_control_axis_deg=5., max_control_rotation_mean_increase_deg=1., failures=0)


def select():
    variants = {name: json.loads((OUT/"development/variants"/f"{name}.json").read_text())
        for name in VARIANTS}
    controls = {name: json.loads((OUT/"development/controls"/f"{name}.json").read_text())
        for name in VARIANTS}
    code = snapshot()
    for item in [*variants.values(), *controls.values()]:
        assert item["code_sha256"] == code
        assert item["summary"]["n"] == 204
    baseline = variants["orientation_baseline"]["summary"]
    control_baseline = controls["orientation_baseline"]["summary"]
    decisions = {}
    for name, item in variants.items():
        s, c = item["summary"], controls[name]["summary"]
        checks = dict(
            failures=s["failures"] == c["failures"] == 0,
            axis=s["axis"]["mean"] <= baseline["axis"]["mean"]+GUARDS["max_axis_mean_increase_deg"],
            axis_success=s["success_15_deg"] >= baseline["success_15_deg"],
            rotation=s["rotation"]["mean"] <= baseline["rotation"]["mean"]+GUARDS["max_rotation_mean_increase_deg"],
            control_pivot=c["pivot"]["mean"] <= GUARDS["max_control_pivot_mean"],
            control_axis=c["axis"]["max"] <= GUARDS["max_control_axis_deg"],
            control_rotation=c["rotation"]["mean"] <= control_baseline["rotation"]["mean"]+GUARDS["max_control_rotation_mean_increase_deg"])
        decisions[name] = dict(eligible=all(checks.values()), checks=checks)
    eligible = [name for name in variants if decisions[name]["eligible"]]
    name = min(eligible, key=lambda n: variants[n]["summary"]["pivot"]["mean"])
    result = dict(variant=name, config=VARIANTS[name], code_sha256=code,
        guards=GUARDS, decisions=decisions, compared_variants=len(variants),
        selection="Minimum complete-cohort development mean pivot error among candidates passing all guards; no validation selection or per-pair truth choice.",
        summary=variants[name]["summary"], control_summary=controls[name]["summary"])
    target = OUT/"selected.json"
    if target.exists():
        assert json.loads(target.read_text()) == result, "Selection already frozen"
    else:
        write(target, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    select()
