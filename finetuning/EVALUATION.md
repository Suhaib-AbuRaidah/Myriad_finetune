# Full-split evaluation and presentation figures

From the repository root, activate `myriad_finetune`, then run:

```bash
python -m finetuning.evaluate \
  --checkpoint finetuning/runs/articulated/best.pt \
  --data-root data/Robotic_arm --split test1 \
  --device cuda:0 --amp-dtype bf16 \
  --ensemble-size 5 --diffusion-steps 16 --seed 42 \
  --cases 10 --output-dir finetuning/runs/test1_presentation
```

Both poke modes run over every scene by default. Add
`--parallel-devices cuda:0 cuda:1` to use separate GPUs concurrently.
Use `--max-scenes 2` and a **different output folder** for a smoke test.

The evaluator snapshots the adapter checkpoint before sampling. Rerunning the
same command resumes completed scenes from that snapshot; changed model/data
settings require a different output folder. It calls the existing
`evaluate_validation` function and does not modify the original `myriad/` code.

```text
test1_presentation/
  checkpoint.pt                  Frozen fine-tuned checkpoint
  run.json                       Settings, checkpoint identity, source hashes
  REPORT.md                      Metrics, ranked scenes, artifact links
  summary.json                   Full distributions and both-mode comparison
  paired_comparison.csv          Identical future-state/point comparison
  plots/                         Summary PNGs and vector PDFs
  with_poke/
    summary.json
    scene_metrics.csv
    selection.json
    results/scene_.../            Every evaluated scene: results.npz + metrics.json
    visualizations/top_10/       Ten lowest minADE_1 scores
    visualizations/failures/     Ten highest minADE_1 scores
  without_poke/                  Same layout, independently ranked
```

Each selected case contains `overview.png`, `final_state.png`, `rollout.gif`,
`rollout.mp4`, `oracle_best5.png`, `results.npz`, and `metrics.json`. The primary
visualizations show **prediction 1**, which is the prediction used for ranking.
Best-of-five selection is ground-truth-dependent and is labeled as an oracle.

## Change the visualization without rerunning inference

```bash
python -m finetuning.evaluate --render-only \
  --output-dir finetuning/runs/test1_presentation
```

This path loads no model and requires no GPU. Customize
`finetuning/evaluation_visuals.py`, then run that command again, or write a new
renderer using the saved arrays:

```python
import json
import numpy as np

with np.load("path/to/a/case/results.npz", allow_pickle=False) as saved:
    predictions = saved["predictions"]       # [hypotheses, states, points, 2]
    ground_truth = saved["ground_truth"]     # [states, points, 2]
    valid = saved["valid"]                   # [states, points], Boolean
    images = saved["rgb_frames"]             # [states, H, W, 3], uint8
    links = saved["link_nodes"]              # [points]
    known_states = int(saved["known_states"])
    metadata = json.loads(str(saved["metadata_json"].item()))
```

Coordinates use normalized image pixel centers: `xy_pixels = xy * (H - 1)`.
Saved predictions preserve out-of-frame values. `image` and `final_image`
retain float32 reference/final inputs in `[-1,1]`; `rgb_frames` provides all
rendered states for display. Those future-state images were not model inputs.
Masks, visibility, original point indices, link labels, observed positions,
state fractions/times, hypothesis scores, and source metadata are also saved.

## Metric definitions

The project's `minADE_1` is the masked mean **squared** Euclidean error of the
first prediction, scaled to `[-1,1]` coordinates. It is not traditional ADE in
pixels. `minADE_5` picks the lowest-error complete trajectory from the first
five hypotheses. With five samples, `minADE_N` equals `minADE_5`.

Pixel ADE is mean Euclidean distance; FDE is final-state mean Euclidean
distance; RMSE is root mean squared Euclidean distance. Headline metrics give
each scene equal weight. The stationary baseline repeats the last observed
positions. Median/p90/p95 describe the distribution of scene scores.

Native evaluation scores with-poke scenes from state 2 onward, and without-poke
scenes from state 1 onward. The paired comparison rescoring uses the same
future states, the intersection of point IDs, and identical validity masks.
Geometrically valid occluded points are included, matching default validation.
"Failure cases" means the worst-ranked examples, not a thresholded failure rate.
