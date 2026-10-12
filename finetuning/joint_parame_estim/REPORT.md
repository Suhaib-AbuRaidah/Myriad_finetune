# Joint-parameter estimation on test1: no complexity penalty

The estimator processed all **320 test1 scenes** using the saved fine-tuned MYRIAD predictions. The identified base is held stationary, and each joint's fitted transform is composed along the known parent-child chain. Both poke modes use hypothesis 0 and model-predicted states 2–4. Test1 contains 919 joints: 874 revolute, 45 fixed, and no prismatic examples.

For each evaluated joint, the estimator fits fixed, prismatic, and revolute motion. It now selects the candidate with the **lowest robust reprojection sum of squared errors (SSE), with no complexity penalty**. The existing small-motion guard still selects fixed if relative motion or the fixed fit's median error is below 3 pixels. The `score_*` columns in the per-joint CSV files are unpenalized robust SSE values.

| Metric | Without poke | With poke |
|:--|--:|--:|
| Joints estimated / all joints | 879 / 919 (95.6%) | 871 / 919 (94.8%) |
| Joint-type accuracy on estimated joints ↑ | 89.9% | 91.3% |
| Correct type / all joints ↑ | 86.0% | 86.5% |
| Revolute recall ↑ | 94.7% (790 / 834) | 96.2% (795 / 826) |
| Fixed recall ↑ | 0.0% (0 / 45) | 0.0% (0 / 45) |
| Revolute joints called prismatic ↓ | 43 | 30 |
| Fixed joints called prismatic / revolute ↓ | 0 / 45 | 4 / 41 |
| Mean axis orientation error, correctly typed revolute ↓ | 55.24° (n=790) | 28.13° (n=795) |
| Median axis orientation error ↓ | 56.90° | 21.49° |
| 90th-percentile axis orientation error ↓ | 84.18° | 63.44° |
| Mean pivot-to-axis distance, correctly typed revolute ↓ | 0.653 (n=790) | 0.176 (n=795) |
| Median pivot-to-axis distance ↓ | 0.234 | 0.0660 |
| 90th-percentile pivot-to-axis distance ↓ | 1.454 | 0.345 |
| Median selected reprojection RMSE ↓ | 7.61 px | 7.40 px |
| Median predicted stationary-base drift ↓ | 7.24 px | 2.18 px |

Compared with the previous complexity-penalized run, type accuracy rose from **77.7% to 89.9%** without poke and from **88.6% to 91.3%** with poke. Revolute→prismatic mistakes fell from **148 to 43** and **54 to 30**, respectively. Removing the penalty did **not** recover fixed joints: fixed recall was previously 0/45 without poke and 2/45 with poke, and is now 0/45 in both modes. A flexible moving model often fits erroneous fixed-child tracks more closely than the true fixed model. Overall accuracy is strongly influenced by the large revolute majority; balanced accuracy is 47.4% without poke and 48.1% with poke across the two present classes.

There are no true prismatic joints in test1, so prismatic recall cannot be measured. Axis and pivot errors are computed only for *correctly classified revolute joints*; the two modes and the previous penalized evaluation therefore have different axis-error cohorts. The small changes in their means are not a controlled comparison on identical joints.

Figures are in [`figures/`](figures/): mean error comparison, error CDFs, joint-type confusion matrices, a stationary-base drift CDF, and [good, typical, and failure overlays](figures/examples/) for each mode. Each figure is saved as PNG and PDF. Per-joint axes, pivots, model residuals, raw selection scores, and classifications are in [`results/without_poke/per_joint.csv`](results/without_poke/per_joint.csv) and [`results/with_poke/per_joint.csv`](results/with_poke/per_joint.csv). The machine-readable summaries are in [`results/summary.json`](results/summary.json). The figures can be regenerated from the saved results with `python -m finetuning.joint_parame_estim.run --report-only`.

The axis metric is the acute angle between estimated and ground-truth directions. The pivot metric is the perpendicular distance from the ground-truth joint origin to the estimated axis line, invariant to sliding the chosen pivot point along that line. Distances are in dataset world-coordinate units. The 2D fitting targets are MYRIAD predictions; the saved `position_valid` mask, derived from the dataset's future geometry, selects which point-state pairs are fitted. This benchmark visibility assumption would need to be replaced for an observation-only deployment.
