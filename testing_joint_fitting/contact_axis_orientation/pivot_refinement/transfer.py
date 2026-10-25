"""Evaluate the selected refinement once on the existing 50-scene validation archive."""
import argparse
import hashlib
import json
from pathlib import Path

from testing_joint_fitting.experiment import write
from ..validation_transfer.evaluate import load_validation, DEFAULT_EVALUATION
from .run import ROOT, OUT, evaluate, snapshot


def load_cases():
    return load_validation(DEFAULT_EVALUATION, OUT/"validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selection_path = OUT/"selected.json"
    selected = json.loads(selection_path.read_text())
    assert selected["code_sha256"] == snapshot()
    protocol = dict(variant=selected["variant"], config=selected["config"],
        code_sha256=selected["code_sha256"],
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256((DEFAULT_EVALUATION/"run.json").read_bytes()).hexdigest(),
        evaluation=str(DEFAULT_EVALUATION), scenes=50, expected_pairs=134,
        tuning="No validation tuning. Selection frozen before loading these cases; validation was previously reported for the old fitter.")
    target = OUT/"validation/protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    else:
        write(target, protocol)
    cases, cohort = load_cases()
    assert len(cases) == 134 and cohort["scoreable_scenes"] == 50
    variants = dict(orientation_baseline=dict(baseline=True))
    variants[selected["variant"]] = selected["config"]
    for control in (False, True):
        evaluate(cases, variants, control=control, workers=args.workers, cohort="validation")
    assert selected["code_sha256"] == snapshot()


if __name__ == "__main__":
    main()
