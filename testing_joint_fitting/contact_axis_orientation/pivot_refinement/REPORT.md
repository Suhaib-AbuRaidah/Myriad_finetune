# Soft contact-center pivot refinement

Completed 25 October 2026.

Selected `symmetric_mean64_adaptive128` from 29 configurations on the original 80 development scenes / 204 revolute pairs. Axis orientation strength remains 128. All predictions are the existing uncropped hypothesis-zero trajectories, without a poke, states 2–4. No neural inference or training was rerun.

## Development: 80 scenes / 204 pairs

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous orientation fitter | 2.3896 | 0.1239 | 97.55% | 0.047689 | 0.034737 | 29.2553 | 12.2579 | 0 |
| symmetric_mean64_adaptive128 | 2.4823 | 0.1739 | 97.55% | 0.007573 | 0.004675 | 28.3801 | 16.6882 | 0 |

Mean pivot error falls **84.12%**. Pivot improves on 189/204 pairs; 15 worsen. All pairs remain in the comparison.

- pivot_distance: paired mean change -0.040116; descriptive 95% scene-cluster bootstrap interval [-0.045634, -0.034721].
- axis_error_deg: paired mean change 0.092704; descriptive 95% scene-cluster bootstrap interval [0.059759, 0.126650].
- rotation_error_deg: paired mean change -0.875166; descriptive 95% scene-cluster bootstrap interval [-2.731490, 0.934218].

GT-track controls (separate diagnostic fits, not predicted-track performance):

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous, GT tracks | 0.0142 | 0.0021 | 100.00% | 0.000560 | 0.000014 | 0.3418 | 0.0123 | 0 |
| Selected, GT tracks | 0.1194 | 0.0528 | 100.00% | 0.001203 | 0.000402 | 0.1482 | 0.0665 | 0 |

Maximum control axis error: 0.2999° → 1.0690°. Soft centering still introduces measurable geometry bias on clean tracks.

### Per-joint results

| Joint | Pairs | Previous pivot | Selected pivot | Previous axis ° | Selected axis ° | Selected rotation ° |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 0.065687 | 0.010185 | 4.6407 | 4.7442 | 23.4993 |
| 2 | 67 | 0.038021 | 0.007078 | 0.2048 | 0.3297 | 24.4222 |
| 3 | 46 | 0.035432 | 0.004129 | 0.2254 | 0.2605 | 38.5139 |
| 4 | 11 | 0.026931 | 0.005998 | 8.3758 | 8.4353 | 45.6064 |

### Per-object results

| Object | Pairs | Previous pivot | Selected pivot | Previous axis ° | Selected axis ° | Selected rotation ° |
|---|---:|---:|---:|---:|---:|---:|
| rec_cantilever_articulated_arm_0004 | 18 | 0.035031 | 0.015408 | 5.0812 | 5.1785 | 34.9561 |
| rec_cantilever_articulated_arm_53eb02010d80465abf4bd189075c6435 | 18 | 0.036171 | 0.004254 | 0.2838 | 0.4586 | 20.4916 |
| rec_cantilever_articulated_arm_90c74bf5a086403498a11d9a92c24869 | 17 | 0.041702 | 0.005272 | 0.1451 | 0.1549 | 29.9807 |
| rec_cantilever_articulated_arm_e5d81287dde44db1a9e400dc7b2ac9bc | 18 | 0.060443 | 0.004175 | 0.2521 | 0.3254 | 32.1852 |
| rec_cantilever_articulated_arm_e66020a994604556ae4625eb7cf5d2b2 | 15 | 0.035472 | 0.008803 | 0.0844 | 0.2692 | 18.4766 |
| rec_robotic_arms_0007 | 24 | 0.057661 | 0.014560 | 0.2999 | 0.2883 | 28.8212 |
| rec_robotic_arms_174c417651d64b078988f7688414b70a | 12 | 0.037283 | 0.006877 | 0.3208 | 0.5431 | 44.8317 |
| rec_robotic_arms_50fae505269842ff849990e1ec095c0f | 15 | 0.048478 | 0.003752 | 6.1661 | 6.2544 | 37.8751 |
| rec_robotic_arms_a0610297f2d749b3bd3ce8a85fd1bd3e | 15 | 0.033251 | 0.004767 | 0.1438 | 0.2079 | 33.7518 |
| rec_serial_elbow_arm_0c99ac188d8746fbb453e13c48b0b0be | 12 | 0.035071 | 0.004530 | 0.2264 | 0.3683 | 16.1610 |
| rec_serial_elbow_arm_3ae3f7254c8147138241aa87eef5272c | 10 | 0.085802 | 0.005931 | 0.1563 | 0.2506 | 10.9696 |
| rec_serial_elbow_arm_64e2a3789d314da5b31c753a30dcc067 | 10 | 0.080034 | 0.011050 | 26.9750 | 26.8590 | 56.7004 |
| rec_serial_elbow_arm_c0b80854bd2e43a8bc9a9afcecff9759 | 10 | 0.030923 | 0.006853 | 0.1527 | 0.3489 | 15.5321 |
| rec_serial_elbow_arm_c8111e5b8ef843138835f71ed1538cda | 10 | 0.067226 | 0.004170 | 0.1391 | 0.3025 | 9.5287 |

## Frozen transfer: 50 scenes / 134 pairs

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous orientation fitter | 6.2397 | 0.1080 | 93.28% | 0.044783 | 0.036818 | 31.5819 | 10.5054 | 0 |
| symmetric_mean64_adaptive128 | 6.9959 | 0.1784 | 92.54% | 0.012450 | 0.006220 | 30.0011 | 13.1509 | 0 |

Mean pivot error falls **72.20%**. Pivot improves on 124/134 pairs; 10 worsen. All pairs remain in the comparison.

- pivot_distance: paired mean change -0.032333; descriptive 95% object-cluster bootstrap interval [-0.044212, -0.023431].
- axis_error_deg: paired mean change 0.756235; descriptive 95% object-cluster bootstrap interval [0.070991, 2.042448].
- rotation_error_deg: paired mean change -1.580813; descriptive 95% object-cluster bootstrap interval [-7.052109, 2.583000].

GT-track controls (separate diagnostic fits, not predicted-track performance):

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous, GT tracks | 2.1503 | 0.0049 | 97.01% | 0.001172 | 0.000037 | 1.1538 | 0.1194 | 0 |
| Selected, GT tracks | 2.0342 | 0.1028 | 97.76% | 0.001816 | 0.000883 | 1.0473 | 0.3315 | 0 |

Maximum control axis error: 78.1393° → 87.6914°. Soft centering still introduces measurable geometry bias on clean tracks.

### Per-joint results

| Joint | Pairs | Previous pivot | Selected pivot | Previous axis ° | Selected axis ° | Selected rotation ° |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 0.049621 | 0.022565 | 2.1557 | 4.1220 | 23.3432 |
| 2 | 49 | 0.046138 | 0.008220 | 9.2435 | 9.2471 | 24.5748 |
| 3 | 31 | 0.036494 | 0.003856 | 8.8369 | 8.9256 | 50.9733 |
| 4 | 4 | 0.031954 | 0.004426 | 0.3651 | 0.3888 | 17.1610 |

### Per-object results

| Object | Pairs | Previous pivot | Selected pivot | Previous axis ° | Selected axis ° | Selected rotation ° |
|---|---:|---:|---:|---:|---:|---:|
| rec_cantilever_articulated_arm_03eac2c3c56c4eb8b696f08713ff0f23 | 11 | 0.036578 | 0.005424 | 0.1098 | 0.1798 | 15.2602 |
| rec_cantilever_articulated_arm_0c46128c7eda409b9d37704d9aea8000 | 13 | 0.035842 | 0.002801 | 0.1448 | 0.1795 | 37.4820 |
| rec_cantilever_articulated_arm_24c67e93e3fa4c3990061124f78ec3a3 | 12 | 0.040316 | 0.007326 | 0.2149 | 0.3942 | 7.2696 |
| rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 12 | 0.020795 | 0.005136 | 22.5629 | 22.5970 | 16.4481 |
| rec_robotic_arms_010b00e58fc346aaa15598c0cf170a6c | 12 | 0.033347 | 0.008616 | 0.6220 | 1.1094 | 40.0062 |
| rec_robotic_arms_047c9d09d0e54055947483c5c5d62db8 | 12 | 0.091723 | 0.007802 | 0.1769 | 0.2875 | 64.5543 |
| rec_robotic_arms_22664768f47842089f994d0b19305bd6 | 14 | 0.064467 | 0.053507 | 38.5128 | 44.6952 | 65.9055 |
| rec_robotic_arms_9ff7f2676a894797a6fdfa7cf09f98b8 | 16 | 0.042051 | 0.008223 | 0.2545 | 0.3484 | 26.9479 |
| rec_serial_elbow_arm_1a535fdff36e42e18f0fa70bd3fbe662 | 8 | 0.048159 | 0.012845 | 0.0847 | 0.1328 | 7.8632 |
| rec_serial_elbow_arm_4ae0d61630354d279fa240118d7617cd | 8 | 0.032062 | 0.007896 | 0.2680 | 0.3165 | 11.7437 |
| rec_serial_elbow_arm_b6f8d1f7bbc449cbaa3e50bc173c3be9 | 8 | 0.043950 | 0.012387 | 0.2306 | 0.3835 | 17.0428 |
| rec_serial_elbow_arm_dfee8ddd003d4de1b9f8f7eba24f5351 | 8 | 0.041216 | 0.009991 | 0.2736 | 0.3139 | 22.3295 |

## What changed

The original broad contact-position prior allowed noisy predicted motion to displace the axis line. The new estimator increases that prior's strength and estimates the center using the selected patch rule. The arithmetic centroid also avoids the coordinate dependence of a coordinate-wise median. Orientation evidence remains a separate fixed 64-pair patch; changing the position patch cannot change its normal matrix.

The existing whitened perpendicular residual is retained:

```text
sigma = max(0.25, seed_median_reprojection_px / sqrt(2 log 2))
effective_strength = position_strength * sigma / hypot(sigma, noise_reference_px)
d = W axis; b = W (pivot - contact_center)
r_position = effective_strength * sigma * (b - d (d.T b)/(d.T d))
```

Without adaptation, noise_reference_px is zero and effective strength equals nominal strength. At the 0.25-pixel sigma floor, a 1-pixel reference multiplies the nominal strength by about 0.243. For large sigma the multiplier approaches one. This scale is a heuristic based only on the initial fit, not a calibrated estimate of model error. Low-residual coherent but wrong tracks can still mislead it.

The same robust soft-L1 position and pixel losses, quadratic orientation loss, multistart optimizer, axis/pivot/angle joint fit and objective-based candidate selection are retained. No post-fit hard snapping or ground-truth candidate selection is used. The returned pivot gauge remains the anchor-centroid plane. Moving only along the axis does not affect pivot distance or the transforms; the improvement comes from moving the infinite line.

Selected configuration:

```json
{
  "position_strength": 128.0,
  "position_count": 64,
  "center_method": "mean",
  "symmetric": true,
  "noise_reference_px": 1.0
}
```

## Selection and all configurations

Minimum complete-cohort development mean pivot error among candidates passing all guards; no validation selection or per-pair truth choice.

Guards fixed before completion of the adaptive sweep:

```json
{
  "max_axis_mean_increase_deg": 0.25,
  "min_axis_success_change": 0.0,
  "max_rotation_mean_increase_deg": 1.0,
  "max_control_pivot_mean": 0.002,
  "max_control_axis_deg": 5.0,
  "max_control_rotation_mean_increase_deg": 1.0,
  "failures": 0
}
```

These are development acceptance limits, not guarantees for new data. The clean-track pivot guard deliberately allows some bias (0.002 world units); the full control change is reported above. Larger patches and stronger forces are retained here even when they regress.

| Configuration | Mean axis ° | Mean pivot | Rotation ° | GT-track mean pivot | GT-track max axis ° | Eligible / rejection |
|---|---:|---:|---:|---:|---:|---|
| orientation_baseline | 2.3896 | 0.047689 | 29.2553 | 0.000560 | 0.2999 | yes |
| median32_soft4 | 2.3896 | 0.047689 | 29.2553 | 0.000560 | 0.2999 | yes |
| median32_soft8 | 2.4089 | 0.034894 | 29.2834 | 0.000729 | 0.6110 | yes |
| median32_soft16 | 2.4408 | 0.023833 | 29.6966 | 0.001157 | 1.3916 | yes |
| median32_soft32 | 2.4579 | 0.016902 | 29.1101 | 0.002026 | 2.4231 | control_pivot |
| median32_soft64 | 2.4741 | 0.012911 | 28.6944 | 0.003390 | 3.5231 | control_pivot |
| median32_soft128 | 2.4861 | 0.011661 | 28.9604 | 0.005168 | 4.5773 | control_pivot, control_rotation |
| mean32_soft16 | 2.4397 | 0.021859 | 28.7507 | 0.000905 | 0.7056 | yes |
| mean32_soft64 | 2.4749 | 0.010069 | 28.9361 | 0.002427 | 2.1097 | control_pivot |
| mean64_soft16 | 2.4330 | 0.022055 | 29.2719 | 0.000979 | 0.7106 | yes |
| mean64_soft64 | 2.4726 | 0.010255 | 27.9548 | 0.002612 | 3.6227 | control_pivot |
| mean128_soft16 | 2.4229 | 0.027534 | 28.5393 | 0.001703 | 1.0343 | yes |
| mean128_soft64 | 2.4693 | 0.016009 | 28.4696 | 0.004578 | 3.7226 | control_pivot |
| mean256_soft16 | 2.4112 | 0.039146 | 28.7119 | 0.002926 | 1.0481 | control_pivot |
| mean256_soft64 | 2.4531 | 0.032919 | 27.8444 | 0.008997 | 4.3302 | control_pivot |
| symmetric_mean64_soft16 | 2.4356 | 0.021451 | 29.4048 | 0.000850 | 0.5719 | yes |
| symmetric_mean64_soft64 | 2.4755 | 0.009625 | 28.0590 | 0.002189 | 1.9980 | control_pivot |
| mean32_adaptive32 | 2.4569 | 0.014742 | 29.0082 | 0.000590 | 0.3149 | yes |
| mean32_adaptive64 | 2.4742 | 0.010122 | 28.9384 | 0.000874 | 0.6834 | yes |
| mean32_adaptive128 | 2.4823 | 0.008459 | 28.3859 | 0.001338 | 1.2995 | yes |
| mean32_adaptive256 | 2.4821 | 0.009798 | 28.7285 | 0.002374 | 2.0757 | control_pivot |
| mean64_adaptive32 | 2.4549 | 0.015029 | 29.2365 | 0.000614 | 0.3496 | yes |
| mean64_adaptive64 | 2.4717 | 0.010318 | 27.9871 | 0.000944 | 0.6948 | yes |
| mean64_adaptive128 | 2.4793 | 0.008724 | 28.5598 | 0.001429 | 1.5265 | yes |
| mean64_adaptive256 | 2.4815 | 0.009927 | 28.7092 | 0.002558 | 3.5242 | control_pivot |
| symmetric_mean64_adaptive32 | 2.4615 | 0.014575 | 29.1020 | 0.000560 | 0.2915 | yes |
| symmetric_mean64_adaptive64 | 2.4748 | 0.009693 | 28.0602 | 0.000820 | 0.5539 | yes |
| symmetric_mean64_adaptive128 | 2.4823 | 0.007573 | 28.3801 | 0.001203 | 1.0690 | yes |
| symmetric_mean64_adaptive256 | 2.4845 | 0.008669 | 27.8742 | 0.002140 | 1.9493 | control_pivot |

## Limits and verification

Pivot distances use dataset world units and the transform-consistent scoring reference. Axis errors are sign-invariant; reprojection is against predicted tracks in source-image pixels. The larger reprojection residual reflects a changed geometry/motion tradeoff; it is not improved trajectory prediction. Angle recovery remains limited, as the rotation errors show.

The repeatedly used development cohort is used for global selection. Scene bootstrap intervals do not account for this search. The selected configuration was frozen before fitting the existing 50-scene validation archive; its 12 object identities do not overlap development's 14 objects. Validation was evaluated previously for the old fitter, so this is a transfer check on an existing archive, not a newly untouched benchmark. Object bootstrap uses just 12 validation clusters. No validation scores change the selected configuration.

The full initial mesh points/normals and existing visibility masks are still required; RGB-D-only performance is untested. A contact centroid need not lie on a real hinge for every geometry. No GT joint parameters, future geometry or future transforms enter the estimator. GT tracks enter only separately labeled controls. Cohort definition uses GT revolute labels, as before.

`results/audit.json` independently checks every saved current-sweep fit's axis, pivot, polar-projected SO(3) rotation, reprojection and positive scored depth. Immutable code snapshots are hash-verified. Synthetic tests cover exact recovery, noisy-track displacement, pivot gauge, rigid-frame centroids, missing geometry, and the existing orientation invariants. Production code, the prior frozen prototype, training and original artifacts remain intact.

![Pivot and axis comparison](results/comparison.png)
