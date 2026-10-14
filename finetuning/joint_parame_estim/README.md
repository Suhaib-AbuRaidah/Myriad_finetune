# Joint-parameter estimation from MYRIAD trajectories

Run from the repository root with the existing `myriad_finetune` environment:

```bash
MPLCONFIGDIR=/tmp/myriad_joint_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m finetuning.joint_parame_estim.run
```

The script reuses the **saved fine-tuned test1 predictions** in
`finetuning/runs/test1_presentation/{with_poke,without_poke}/results/`.
It does not reload or copy a checkpoint. It uses hypothesis 0, not an oracle
best-of-five selection. Both modes are fitted to states 2–4, which are
model-generated in both modes. State 0 is the RGB-D reference anchor. This
ensures the with-poke run does not gain an observed state-1 trajectory sample
inside the joint-parameter fit.

The numeric 2D fitting targets come only from the saved MYRIAD predictions.
As in the earlier run, point inclusion uses the saved `position_valid` mask,
which was generated from the dataset's future geometry. This is a benchmark
visibility assumption and should be replaced by predicted visibility for a
fully observation-only deployment.

The dataset supplies the identified link labels and parent-child hierarchy
(`joint_parent_link_indices` and `joint_child_link_indices`). This run assumes
that structure is known; it does not infer the segmentation or joint types.

The estimation procedure is:

1. Reproduce the selected surface points, then back-project their reference
   pixels using reference depth and the calibrated camera. Source-image pixel
   coordinates are reconstructed from the saved normalized predictions.
2. Hold the identified base exactly stationary. Traverse the known
   parent-child hierarchy from that base. Each child's motion is the product
   of its parent's **fitted joint-chain motion** and its own joint motion.
   Calibrated PnP on the child points only supplies an initial guess for that
   child's joint; independent PnP poses are never propagated to descendants.
3. Fit exactly three candidate types—fixed, prismatic, and revolute—to each
   child's 2D tracks by robust, damped reprojection-error minimization.
   Revolute motion follows `p_t = R(axis, angle_t)(p_0 - pivot) + pivot`;
   prismatic motion follows `p_t = p_0 + distance_t * axis`; fixed motion
   follows the parent chain. The worst residuals are trimmed once and the
   dynamic models are refitted.
4. Select the joint type by the lowest robust reprojection loss, without a
   model-complexity penalty. A small-motion guard keeps nearly stationary
   links fixed. A unit-axis constraint and the revolute pivot's
   axial gauge reduce unidentifiable degrees of freedom. The stationary
   base's predicted-track error is reported as a diagnostic, not used to
   loosen the type threshold. No ground-truth type enters this selection.

This is an adaptation of [Rest2Art section 3.3](https://arxiv.org/html/2607.27749),
not a reproduction of its mesh-penetration and part-boundary priors. Umeyama
requires corresponding points in the same dimension and is therefore
insufficient for recovering a 3D axis directly from 3D reference points and
2D target trajectories. The camera-aware reprojection fit solves that
dimensional mismatch.

Ground-truth joint arrays are read only for evaluation. An axis orientation
error is `acos(abs(dot(estimated_axis, GT_axis)))` in degrees, treating opposite
axis directions as equivalent. Pivot distance is the perpendicular distance
from the ground-truth origin to the estimated rotation-axis line. This is
invariant to where the pivot point is placed *along* that axis. Distances are
in the simulator's world-coordinate units. Axis and pivot errors are reported
for **correctly classified revolute joints**; joint-type accuracy and coverage
must be read alongside them. Fixed joints have no meaningful axis or pivot.

The test1 dataset contains no prismatic ground-truth joints, so prismatic
recall is undefined. The estimator can still output a prismatic type, shown
as a false positive in the confusion matrix. See `REPORT.md` for the full
results, `results/<mode>/per_joint.csv` for every estimate and ground-truth
comparison, `results/<mode>/per_scene.csv` for per-scene results, and
`figures/` for PNG and PDF charts and example axis overlays. Run
`python -m finetuning.joint_parame_estim.run --report-only` to redraw all figures from
saved per-joint estimates without another inference or fitting pass.

The earlier independent-link-pose results and figures are preserved under
`baseline_independent_pose/` for comparison.

For the small parent-child-only inference experiment and its matched
full-scene comparison, see [`PAIRWISE_EXPERIMENT.md`](PAIRWISE_EXPERIMENT.md).
