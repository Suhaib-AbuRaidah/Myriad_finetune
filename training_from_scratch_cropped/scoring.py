"""Axis-only checkpoint selection, including all GT-revolute failures."""
import math
import numpy as np

DEFINITION = dict(version=1, metric="mean_axis_error_deg", range=[0,90],
    formula="degrees(acos(clip(abs(dot(unit(predicted_revolute_axis_world),unit(gt_axis_world))),0,1)))",
    cohort="fixed manifest GT-revolute pairs regardless of predicted type",
    failure_penalty_deg=90, aggregation="arithmetic mean over complete cohort",
    improvement="strictly smaller; exact ties retain existing best", tie_breakers=[])


def axis_error(predicted, truth):
    try:
        p, g = np.asarray(predicted, float), np.asarray(truth, float)
        if p.shape != (3,) or g.shape != (3,) or not np.isfinite([p,g]).all() or min(np.linalg.norm(p), np.linalg.norm(g)) < 1e-12:
            return 90.
        return math.degrees(math.acos(np.clip(abs(np.dot(p/np.linalg.norm(p), g/np.linalg.norm(g))), 0, 1)))
    except (TypeError, ValueError):
        return 90.


def checkpoint_score(rows, cohort):
    lookup = {r["index"]:r for r in rows}
    if not cohort or len(lookup) != len(rows) or not set(cohort).issubset(lookup):
        raise ValueError("Missing/duplicate validation predictions or empty axis cohort")
    errors = [axis_error(lookup[i].get("forced_revolute_axis_world"), lookup[i].get("gt_axis_world")) for i in cohort]
    return math.fsum(errors)/len(errors)


class BestState:
    def __init__(self, state=None):
        self.score, self.history = None, []
        if state:
            if state["definition"] != DEFINITION:
                raise ValueError("Axis checkpoint scoring policy changed")
            self.score, self.history = state["score"], state["history"]

    def consider(self, score, step):
        if not math.isfinite(score) or not 0 <= score <= 90:
            raise ValueError("Invalid axis score")
        improved = self.score is None or score < self.score
        self.history.append(dict(step=step, score=score, improved=improved))
        if improved:
            self.score = score
        return improved

    def state_dict(self):
        return dict(definition=DEFINITION, score=self.score, history=self.history)
