# Joint-parameter estimation on test1

The estimator processed all **320 test1 scenes** using the saved predictions
from the fine-tuned MYRIAD checkpoint. Both the with-poke and without-poke
runs use only model-predicted states 2–4 and hypothesis 0. The test split has
919 joints: 874 revolute and 45 fixed; it has no prismatic examples.

| Metric ↓ unless noted | Without poke | With poke |
|:--|--:|--:|
| Joints with an estimate / all joints | 879 / 919 (95.6%) | 871 / 919 (94.8%) |
| Joint-type accuracy, estimated joints ↑ | 78.6% | 87.7% |
| Correct type / all joints ↑ | 75.2% | 83.1% |
| Balanced type accuracy, fixed + revolute ↑ | 41.4% | 47.3% |
| Revolute recall ↑ | 82.9% (691 / 834) | 92.4% (763 / 826) |
| Fixed recall ↑ | 0.0% (0 / 45) | 2.2% (1 / 45) |
| Mean axis orientation error, correct revolute type | 56.24° (n=691) | 26.85° (n=763) |
| Median axis orientation error | 58.42° | 20.64° |
| 90th percentile axis orientation error | 85.39° | 58.62° |
| Mean pivot-to-axis distance, correct revolute type | 0.426 (n=691) | 0.182 (n=763) |
| Median pivot-to-axis distance | 0.154 | 0.0554 |
| 90th percentile pivot-to-axis distance | 1.017 | 0.220 |
| Median reprojection RMSE | 7.18 px | 7.49 px |
| Reprojection failures over 50 px | 3 | 5 |

The poke-conditioned tracks yield substantially better estimated axes and
pivots. These geometry means are **conditional on correct joint-type
classification**, and the two columns contain different numbers of joints;
they should not be read as a paired improvement for every joint. Fixed-joint
recognition is weak, so the relatively high overall type accuracy mainly
reflects the many revolute joints. Of the 45 fixed joints, 43 are called
revolute and 2 prismatic without poke; 32 are called revolute and 12
prismatic with poke. There are no true prismatic joints, so prismatic recall
cannot be reported.

Two with-poke fits have extreme reprojection errors above one million pixels;
they remain in the raw CSV for audit. This makes mean reprojection RMSE
unrepresentative; the median, 90th percentile, and failure counts are more
useful for this run. Estimates for 40 without-poke and 48 with-poke joints
could not be made because the required link poses or point correspondences
were insufficient. These omissions count against `correct type / all joints`.

As a geometry sanity check, replacing predictions with ground-truth 2D tracks
on the first ten test1 scenes gave 27/29 correct joint types. Among 24 correctly
classified revolute joints, the median orientation error was 0.000035° and
the median pivot-to-axis distance was 0.00000016 world units. This small
check verifies the projection and joint-coordinate conventions; it is not a
model result.

The figures are in [`figures/`](figures/): mean comparison, error CDFs, type
confusion matrices, and [good, typical, and failure axis overlays](figures/examples/)
for each mode, each as PNG and PDF. The overlays project the estimated and
ground-truth 3D axes onto the reference image; their titles show both axis
orientation error and pivot distance. Per-joint records include
estimated and ground-truth axes and pivots, type decisions, and all three
model residuals in [`results/without_poke/per_joint.csv`](results/without_poke/per_joint.csv)
and [`results/with_poke/per_joint.csv`](results/with_poke/per_joint.csv).
Machine-readable summaries are in [`results/summary.json`](results/summary.json).
Run `python -m finetuning.joint_parame_estim.run --report-only` to regenerate the figures
and summary from saved CSV files without model inference or joint refitting.

The axis error uses the acute angle between the two axis directions. The
pivot metric is the perpendicular distance from the ground-truth joint origin
to the estimated axis line, so arbitrary shifts *along* an axis do not count
as error. Distances are in the dataset's world-coordinate units; the dataset
does not establish a physical unit such as meters here.
