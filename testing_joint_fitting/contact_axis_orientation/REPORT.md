# Contact axis orientation experiment

For the full fitting procedure, equations, optimizer settings, limitations and
the exact meaning of `ambiguous_mixture_strength128`, see [ALGORITHM.md](ALGORITHM.md).

Evaluated 50 configurations on the original **80 test1 scenes / 204 revolute pairs**. The recommended development configuration is `ambiguous_mixture_strength128`.

These are the original saved uncropped hypothesis-zero Myriad predictions, without a poke. No neural inference or training was rerun. The cohort, predictions and production fitting are unchanged.

## Main results

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot distance | Mean rotation ° | Reprojection RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 46.600 | 46.075 | 9.3% | 0.67634 | 44.523 | 9.075 | 0 |
| rgbd_position | 46.603 | 46.981 | 13.7% | 0.07921 | 43.516 | 8.651 | 0 |
| mesh_position | 46.028 | 47.573 | 12.7% | 0.07753 | 43.950 | 8.675 | 0 |
| major_4 | 41.780 | 37.070 | 19.6% | 0.07616 | 42.990 | 8.689 | 0 |
| shape_16 | 29.326 | 18.992 | 42.6% | 0.08426 | 40.350 | 9.098 | 0 |
| quad_shape_16 | 25.803 | 13.402 | 52.9% | 0.09005 | 40.509 | 10.498 | 0 |
| surface_normal_field_64 | 5.376 | 0.573 | 94.1% | 0.08761 | 29.847 | 11.170 | 0 |
| paired64_normal_field_64_position4 | 3.315 | 0.463 | 97.1% | 0.04825 | 29.007 | 11.882 | 0 |
| ambiguous_mixture_strength128 | 2.390 | 0.124 | 97.5% | 0.04769 | 29.255 | 12.258 | 0 |

![Axis error distributions and per-joint results](results/comparison.png)

## What changed, and why

1. **Position-only controls.** Reproduced the previous visible RGB-D contact prior, then replaced only its contact points with initial mesh samples. The latter alone changes axis error very little; richer geometry is not silently credited to the orientation term.

2. **Principal directions of contact positions.** Compared longest, middle and shortest covariance directions, perpendicularity to the longest direction, and competing candidate axes. A stable line supports its longest direction; a stable plane supports its normal. The eigenvalue-gap shape rule helped more than imposing one direction on all contacts. Multiple candidates chosen by reprojection alone often favored a geometrically wrong direction.

3. **Prior treatment and optimization.** Treating the orientation residual as a pixel outlier weakens it at large disagreement. A quadratic, sign-invariant geometric prior plus a contact-axis initialization improved the result. Data residuals retain the previous soft-L1 pixel loss. Axis, pivot and all three angles are refitted together.

4. **Principal directions of contact surface normals.** Mesh points include normals. A planar mating interface has normals along its normal even when its point positions form a ring, irregular arc or partial disc. We therefore tested the uncentered normal second moment, in addition to point covariance. Local point-plane checks supported world-frame normals (median absolute dot product 0.991 over 1,027 planar neighborhoods). This is additional initial mesh geometry, not a ground-truth joint direction.

5. **Sampling balance and interface consistency.** Repeated nearest neighbors can overrepresent one surface. Compared deduplication, equal mass from both surfaces, and a parallel-normal weight `abs(n_parent · n_child)^8`. This favors locally compatible mating surfaces while allowing either normal sign. Orientation patches were varied separately from the fixed 32-pair position patch. Strengthening the position prior prevented noisy predicted tracks from pulling the axis line away from the connection.

6. **Ambiguity checks.** Tested point/normal fusion, a shape fallback, a projected fallback, multiscale agreement, and multiple normal-axis candidates when the top eigenvalue gap is below 0.2. All candidates are selected with the same reprojection-plus-geometric objective; ground-truth errors never select an example's output. Final patch-size and prior-strength checks are retained below, including regressions.

7. **Stopping and control guard.** `ambiguous_mixture_strength256` has the lowest predicted-track mean axis error (2.271°), but its stronger prior creates a 61.53° axis error in a GT-track control. The recommended strength retains 0.300° maximum control axis error while giving almost the same predicted-track result. The final doubling of strength gained only 0.119° on predictions and worsened that control. We stopped at this tradeoff rather than continuing to overpower contradictory trajectory evidence. The recommendation minimizes predicted-track mean axis error among tested configurations whose GT-track control has no axis error over 15°.

## Selected objective

The contact normal matrix is

```text
S = sum_i w_i (np_i np_i^T + nc_i nc_i^T) / (2 sum_i w_i)
w_i = abs(np_i dot nc_i)^8
M = I - S / lambda_max(S)
orientation residual = sigma * strength * sqrt(M) * unit_axis
```

This softly favors the best-supported normal direction, permits an ambiguous dominant subspace, and produces no directional force for an isotropic normal distribution. Candidate-axis alternatives use `sigma * strength * cross(unit_axis, candidate)`, whose squared norm is sign-invariant. Sigma is calibrated once from the original seed's median reprojection residual and shared across starts. The axis-to-contact position residual is a whitened perpendicular distance to the line. The fitted pivot uses the plane through the initial anchor centroid perpendicular to the fitted axis as its gauge.

```json
{
  "mode": "ambiguous_mixture",
  "strength": 128.0,
  "quadratic": true,
  "starts": true,
  "normal_method": "paired",
  "orientation_count": 64,
  "position_strength": 4.0
}
```

## Per-joint comparison

| Joint ordinal | Pairs | Old axis ° | Selected axis ° | Old pivot | Selected pivot | Selected rotation ° |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 39.246 | 4.641 | 0.09863 | 0.06569 | 23.869 |
| 2 | 67 | 48.372 | 0.205 | 0.05382 | 0.03802 | 25.454 |
| 3 | 46 | 53.972 | 0.225 | 0.08322 | 0.03543 | 39.049 |
| 4 | 11 | 58.525 | 8.376 | 0.07586 | 0.02693 | 50.624 |

Per-object distributions and all pair parameters are in `results/selected.json` and the variant JSONs.

## Ground-truth trajectory controls

These controls replace predicted tracks with GT tracks only in explicitly separate control runs. They test the optimizer and prior bias; they are not model prediction results.

| Configuration | Mean axis ° | Median axis ° | Mean pivot | Mean rotation ° |
|---|---:|---:|---:|---:|
| ambiguous_mixture_64 | 0.01753 | 0.00309 | 0.0005654 | 0.34361 |
| ambiguous_mixture_strength128 | 0.01423 | 0.00212 | 0.0005597 | 0.34181 |
| ambiguous_mixture_strength256 | 0.31730 | 0.00115 | 0.0006233 | 0.72501 |
| mesh_position | 0.74212 | 0.00024 | 0.0011433 | 1.54857 |
| original | 0.75612 | 0.00002 | 0.0054903 | 1.51772 |
| paired64_normal_field_64_position4 | 0.01756 | 0.00309 | 0.0005655 | 0.34361 |
| rgbd_position | 0.41166 | 0.00116 | 0.0008784 | 0.85344 |

## Uncertainty and remaining errors

Compared with the old RGB-D position prior, 199/204 pairs have lower axis error; 5 still exceed 15°. No failed fits were discarded.

- axis_error_deg: mean paired change -44.213754; descriptive 95% scene-bootstrap interval [-48.047144, -40.073950].
- pivot_distance: mean paired change -0.031523; descriptive 95% scene-bootstrap interval [-0.038905, -0.024348].
- rotation_error_deg: mean paired change -14.260348; descriptive 95% scene-bootstrap interval [-17.997888, -10.554062].

The bootstrap resamples scenes rather than treating their joints as independent. It does not account for selecting a configuration on this repeatedly used development set. The remaining 240 scenes were not used for tuning or scoring here. A frozen-configuration evaluation on them is the next generalization check.

Low axis error does not imply an equally accurate transformation: angle magnitude/direction and pivot still depend on noisy predicted motion. Strong geometry priors can increase reprojection error while improving joint parameters. Complete initial mesh geometry and mesh normals are required for the strongest results; RGB-D-only performance has not been established by these experiments.

One GT-track control still has approximately 64° rotation error with subpixel reprojection RMSE, despite an accurate axis direction. Rotation/angle ambiguity is therefore a separate remaining limitation; the method is not a complete solution to transformation recovery.

## Metric definitions and provenance

- Axis: `degrees(acos(abs(dot(normalized predicted axis, normalized GT axis))))`, range 0–90°. A failed fit is assigned 90° in the primary selection score.
- Pivot: perpendicular distance from the transform-consistent GT pivot reference to the predicted infinite axis line, in dataset world length units. Displacement along the same axis line is not penalized.
- Rotation: mean geodesic rotation error over states 2, 3 and 4, in degrees.
- Reprojection: per-pair RMSE of the fitted 3D motion against the saved predicted 2D tracks, averaged over pairs, in original 512-image pixels. It is not trajectory ADE against GT.
- Hypothesis zero, original validity masks, the fixed original scene subset and the same selected child pairs are used for every configuration. GT kind defines an assumed-revolute cohort; classification is not measured.
- `results/manifest.json` records source fingerprints and initial-mesh hashes. `cache_manifest.json` pins input file size/mtime and the locally generated case-cache SHA256. Each variant has its code snapshot hash.

## All tested configurations

| Configuration | Mean axis ° | Median axis ° | Mean pivot | Mean rotation ° | CPU fit seconds |
|---|---:|---:|---:|---:|---:|
| ambiguous_mixture_strength256 | 2.2708 | 0.0319 | 0.048136 | 29.6670 | 23.40 |
| ambiguous_mixture_strength128 | 2.3896 | 0.1239 | 0.047689 | 29.2553 | 22.62 |
| ambiguous_mixture_64 | 2.8577 | 0.4556 | 0.048104 | 29.0219 | 22.89 |
| paired64_normal_major_64 | 3.1671 | 0.3818 | 0.090913 | 30.5788 | 22.76 |
| projected_64 | 3.2015 | 0.4629 | 0.048086 | 29.0995 | 21.85 |
| paired64_normal_major_64_position4 | 3.2223 | 0.4343 | 0.048271 | 28.8771 | 23.46 |
| paired64_normal_field_64 | 3.2378 | 0.4257 | 0.091716 | 30.5323 | 21.27 |
| paired64_normal_field_64_position2 | 3.2641 | 0.4636 | 0.067023 | 29.9443 | 21.92 |
| multiscale_64 | 3.2665 | 0.4322 | 0.050114 | 29.0861 | 23.29 |
| paired64_normal_field_64_position4 | 3.3150 | 0.4629 | 0.048248 | 29.0072 | 22.07 |
| gated_64 | 3.3188 | 0.4629 | 0.049005 | 29.1004 | 21.93 |
| fused_64 | 3.5820 | 0.9209 | 0.048075 | 29.0905 | 21.91 |
| balanced64_normal_field_64 | 4.1186 | 0.6116 | 0.087746 | 30.7272 | 21.29 |
| balanced64_normal_major_64 | 4.1460 | 0.3939 | 0.089570 | 30.8493 | 22.90 |
| balanced64_normal_field_64_position2 | 4.1794 | 0.6262 | 0.063928 | 30.2148 | 21.56 |
| balanced32_normal_major_64 | 4.2119 | 0.4004 | 0.087672 | 29.9266 | 22.78 |
| ambiguous_mixture_count128_64 | 4.2371 | 0.5124 | 0.048754 | 28.0975 | 23.75 |
| balanced64_normal_field_64_position4 | 4.2531 | 0.6427 | 0.047063 | 28.9552 | 21.96 |
| balanced32_normal_field_64 | 4.2543 | 0.6214 | 0.086090 | 30.0512 | 21.40 |
| paired128_normal_field_64_position4 | 4.4202 | 0.5231 | 0.048890 | 28.1762 | 21.55 |
| ambiguous_mixture_strength32 | 4.5394 | 1.7125 | 0.046173 | 30.2863 | 22.29 |
| surface_normal_field_64 | 5.3759 | 0.5728 | 0.087608 | 29.8467 | 20.90 |
| surface_normal_major_64 | 5.5112 | 0.4008 | 0.090378 | 29.8403 | 22.41 |
| surface_normal_confident_64 | 5.5573 | 0.6277 | 0.087083 | 29.9406 | 22.58 |
| paired32_normal_field_64_position4 | 6.7149 | 0.4938 | 0.050456 | 28.7276 | 22.28 |
| surface_normal_major_16 | 11.2719 | 4.9098 | 0.086475 | 33.3135 | 23.63 |
| surface_normal_field_16 | 12.4163 | 6.5678 | 0.081306 | 33.0890 | 21.88 |
| surface_normal_confident_16 | 13.8341 | 7.2660 | 0.079817 | 33.7811 | 23.71 |
| surface_normal_shape_16 | 19.1016 | 7.5611 | 0.084904 | 35.2046 | 24.59 |
| quad_shape_16 | 25.8028 | 13.4019 | 0.090046 | 40.5088 | 24.70 |
| shape_16 | 29.3261 | 18.9915 | 0.084260 | 40.3496 | 18.66 |
| major_16 | 34.5697 | 21.6020 | 0.087203 | 41.2630 | 18.53 |
| surface_normal_major_4 | 34.9053 | 29.0018 | 0.074166 | 40.2516 | 24.50 |
| quad_mixture_16 | 35.4401 | 18.1294 | 0.083265 | 37.4036 | 40.17 |
| mixture_16 | 35.8430 | 20.0801 | 0.079944 | 39.5489 | 28.16 |
| quad_major_16 | 39.7819 | 23.8897 | 0.113328 | 45.5556 | 24.36 |
| quad_shape_4 | 40.4758 | 37.1913 | 0.074660 | 42.3910 | 24.82 |
| quad_major_4 | 41.3950 | 36.3339 | 0.076003 | 43.0274 | 24.18 |
| shape_4 | 41.6342 | 37.8961 | 0.076090 | 42.4421 | 18.70 |
| major_4 | 41.7800 | 37.0701 | 0.076161 | 42.9895 | 18.59 |
| quad_mixture_4 | 42.2991 | 38.9682 | 0.073914 | 42.1212 | 40.10 |
| mixture_4 | 43.1866 | 40.6421 | 0.074973 | 42.5607 | 27.37 |
| all_4 | 45.0587 | 43.6481 | 0.076177 | 42.7828 | 36.04 |
| mesh_solver_zero | 46.0277 | 47.5732 | 0.077526 | 43.9500 | 17.07 |
| mesh_position | 46.0279 | 47.5732 | 0.077526 | 43.9500 | 19.45 |
| original | 46.6002 | 46.0749 | 0.676344 | 44.5228 | 9.42 |
| rgbd_position | 46.6034 | 46.9807 | 0.079212 | 43.5156 | 19.06 |
| normal_4 | 46.6908 | 48.3156 | 0.077884 | 43.0207 | 18.19 |
| middle_4 | 48.4350 | 50.8775 | 0.078705 | 44.2726 | 18.20 |
| perp_major_4 | 49.0977 | 50.9369 | 0.077301 | 44.5362 | 17.85 |

These timings include geometry extraction, the original fit, refinement and scoring, but exclude the one-time data load. The selected configuration's measured wall time for 204 pairs with four CPU workers was 5.70 s; summed fit/scoring time was 22.62 CPU seconds. Workers contend for the host, so these are approximate run timings.

Inspired by [Rest2Art §3.3](https://arxiv.org/html/2607.27749v1#S3.SS3); this implementation was developed independently and is not a reproduction of unpublished code.
