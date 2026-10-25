# Unchanged orientation contact patch for both priors

Completed 25 October 2026. This is a fixed comparison of three fitters, not a parameter search.

**Reusing the orientation patch unchanged did not improve pivot accuracy over the preceding pivot refinement.** Mean axis error changes very little. The new run is retained as an experimental ablation; the previous selected prototype is not replaced.

## Exact change

The existing call `mesh_contact(parent, child, 64, False, parent_normals, child_normals, 'paired')` runs once per normal-supported shared fit. Its returned object is used directly by both `direction_options` and the position term in `solve`. Nothing in that object is overwritten.

- Support: the existing 64 closest child-to-parent nearest-neighbor pairs, without reverse pairs.
- Center: the existing coordinate-wise median of pair midpoints.
- Covariance, uncertainty floor and whitening: exactly those returned by the existing extractor.
- Orientation: the existing paired-normal matrix with `abs(n_parent dot n_child)^8`, its original normal validity/fallback rules, and its original ambiguous-direction candidates.

Normal agreement continues to weight the orientation matrix only. No new normal filtering, normal-weighted centroid, arithmetic centroid or patch-finding rule was introduced. This distinction is necessary to obey the request to reuse the existing contact area without modification.

Compared with the preceding pivot refinement, the pivot now uses this one-way patch and its median/covariance instead of a separately extracted symmetric patch with an arithmetic mean. Position strength 128, 1-pixel noise reference, orientation strength 128, optimizer, initialization, multistarts and loss functions are fixed. The same orientation evidence can yield a slightly different optimized axis because axis, pivot and angles are fitted jointly.

The frozen configuration is:

```json
{
  "position_strength": 128.0,
  "noise_reference_px": 1.0,
  "contact_source": "unchanged orientation contact object"
}
```

## Development: 80 scenes / 204 revolute pairs

Primary fits use saved **Myriad predicted trajectories**, hypothesis zero, without a poke, states 2/3/4. No GT trajectories enter these fits.

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original orientation fitter | 2.3896 | 0.1239 | 97.55% | 0.047689 | 0.034737 | 29.2553 | 12.2579 | 0 |
| Previous pivot refinement | 2.4823 | 0.1739 | 97.55% | 0.007573 | 0.004675 | 28.3801 | 16.6882 | 0 |
| Shared unchanged orientation patch | 2.4793 | 0.1708 | 97.55% | 0.010799 | 0.007533 | 28.7530 | 16.7543 | 0 |

Relative to the preceding pivot refinement, mean pivot error increases **42.60%**. 43/204 pairs improve and 161 worsen. All pairs are retained, with no failed fits.

- Shared minus previous, pivot_distance: mean change 0.003226; descriptive 95% scene-cluster bootstrap interval [0.002365, 0.004116].
- Shared minus previous, axis_error_deg: mean change -0.003047; descriptive 95% scene-cluster bootstrap interval [-0.009714, 0.003607].
- Shared minus previous, rotation_error_deg: mean change 0.372906; descriptive 95% scene-cluster bootstrap interval [-0.653119, 1.391531].

### Separate GT-track controls

These diagnostic runs replace predicted trajectories with GT trajectories to check prior bias. They do not select or retune any configuration.

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original orientation fitter | 0.0142 | 0.0021 | 100.00% | 0.000560 | 0.000014 | 0.3418 | 0.0123 | 0 |
| Previous pivot refinement | 0.1194 | 0.0528 | 100.00% | 0.001203 | 0.000402 | 0.1482 | 0.0665 | 0 |
| Shared unchanged orientation patch | 0.1832 | 0.0879 | 100.00% | 0.001719 | 0.000652 | 0.2551 | 0.0942 | 0 |

| Fitter | Maximum control axis ° | Maximum control pivot |
|---|---:|---:|
| Original orientation fitter | 0.2999 | 0.069350 |
| Previous pivot refinement | 1.0690 | 0.022067 |
| Shared unchanged orientation patch | 1.5453 | 0.024501 |

The shared-patch fit also has greater mean clean-track pivot bias than the preceding pivot refinement. Geometry consistency alone does not guarantee that a patch center is a good hinge location.

### Per-joint predicted-track comparison

| Joint | Pairs | Previous pivot | Shared pivot | Previous axis ° | Shared axis ° |
|---|---:|---:|---:|---:|---:|
| 1 | 80 | 0.010185 | 0.013677 | 4.7442 | 4.7406 |
| 2 | 67 | 0.007078 | 0.009854 | 0.3297 | 0.3311 |
| 3 | 46 | 0.004129 | 0.006726 | 0.2605 | 0.2534 |
| 4 | 11 | 0.005998 | 0.012649 | 8.4353 | 8.4266 |

### Per-object predicted-track comparison

| Object | Pairs | Previous pivot | Shared pivot | Previous axis ° | Shared axis ° |
|---|---:|---:|---:|---:|---:|
| rec_cantilever_articulated_arm_0004 | 18 | 0.015408 | 0.019254 | 5.1785 | 5.1902 |
| rec_cantilever_articulated_arm_53eb02010d80465abf4bd189075c6435 | 18 | 0.004254 | 0.006099 | 0.4586 | 0.4645 |
| rec_cantilever_articulated_arm_90c74bf5a086403498a11d9a92c24869 | 17 | 0.005272 | 0.007676 | 0.1549 | 0.1580 |
| rec_cantilever_articulated_arm_e5d81287dde44db1a9e400dc7b2ac9bc | 18 | 0.004175 | 0.005358 | 0.3254 | 0.3272 |
| rec_cantilever_articulated_arm_e66020a994604556ae4625eb7cf5d2b2 | 15 | 0.008803 | 0.010243 | 0.2692 | 0.2447 |
| rec_robotic_arms_0007 | 24 | 0.014560 | 0.022106 | 0.2883 | 0.2667 |
| rec_robotic_arms_174c417651d64b078988f7688414b70a | 12 | 0.006877 | 0.009706 | 0.5431 | 0.5239 |
| rec_robotic_arms_50fae505269842ff849990e1ec095c0f | 15 | 0.003752 | 0.008180 | 6.2544 | 6.2778 |
| rec_robotic_arms_a0610297f2d749b3bd3ce8a85fd1bd3e | 15 | 0.004767 | 0.007224 | 0.2079 | 0.2130 |
| rec_serial_elbow_arm_0c99ac188d8746fbb453e13c48b0b0be | 12 | 0.004530 | 0.005730 | 0.3683 | 0.3661 |
| rec_serial_elbow_arm_3ae3f7254c8147138241aa87eef5272c | 10 | 0.005931 | 0.010515 | 0.2506 | 0.2444 |
| rec_serial_elbow_arm_64e2a3789d314da5b31c753a30dcc067 | 10 | 0.011050 | 0.013856 | 26.8590 | 26.8394 |
| rec_serial_elbow_arm_c0b80854bd2e43a8bc9a9afcecff9759 | 10 | 0.006853 | 0.007970 | 0.3489 | 0.3333 |
| rec_serial_elbow_arm_c8111e5b8ef843138835f71ed1538cda | 10 | 0.004170 | 0.009575 | 0.3025 | 0.3132 |

## Validation: 50 scenes / 134 revolute pairs

Primary fits use saved **Myriad predicted trajectories**, hypothesis zero, without a poke, states 2/3/4. No GT trajectories enter these fits.

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original orientation fitter | 6.2397 | 0.1080 | 93.28% | 0.044783 | 0.036818 | 31.5819 | 10.5054 | 0 |
| Previous pivot refinement | 6.9959 | 0.1784 | 92.54% | 0.012450 | 0.006220 | 30.0011 | 13.1509 | 0 |
| Shared unchanged orientation patch | 6.9945 | 0.1964 | 92.54% | 0.014616 | 0.007738 | 30.5436 | 13.1965 | 0 |

Relative to the preceding pivot refinement, mean pivot error increases **17.40%**. 34/134 pairs improve and 100 worsen. All pairs are retained, with no failed fits.

- Shared minus previous, pivot_distance: mean change 0.002166; descriptive 95% object-cluster bootstrap interval [0.001659, 0.002654].
- Shared minus previous, axis_error_deg: mean change -0.001431; descriptive 95% object-cluster bootstrap interval [-0.014810, 0.012833].
- Shared minus previous, rotation_error_deg: mean change 0.542498; descriptive 95% object-cluster bootstrap interval [0.098864, 1.002420].

### Separate GT-track controls

These diagnostic runs replace predicted trajectories with GT trajectories to check prior bias. They do not select or retune any configuration.

| Configuration | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Median pivot | Mean rotation ° | RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original orientation fitter | 2.1503 | 0.0049 | 97.01% | 0.001172 | 0.000037 | 1.1538 | 0.1194 | 0 |
| Previous pivot refinement | 2.0342 | 0.1028 | 97.76% | 0.001816 | 0.000883 | 1.0473 | 0.3315 | 0 |
| Shared unchanged orientation patch | 2.0272 | 0.1541 | 97.01% | 0.002631 | 0.001160 | 1.3062 | 0.3601 | 0 |

| Fitter | Maximum control axis ° | Maximum control pivot |
|---|---:|---:|
| Original orientation fitter | 78.1393 | 0.048704 |
| Previous pivot refinement | 87.6914 | 0.040926 |
| Shared unchanged orientation patch | 77.4994 | 0.061503 |

The shared-patch fit also has greater mean clean-track pivot bias than the preceding pivot refinement. Geometry consistency alone does not guarantee that a patch center is a good hinge location.

### Per-joint predicted-track comparison

| Joint | Pairs | Previous pivot | Shared pivot | Previous axis ° | Shared axis ° |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 0.022565 | 0.024573 | 4.1220 | 4.1184 |
| 2 | 49 | 0.008220 | 0.010105 | 9.2471 | 9.2642 |
| 3 | 31 | 0.003856 | 0.006615 | 8.9256 | 8.9004 |
| 4 | 4 | 0.004426 | 0.007424 | 0.3888 | 0.3714 |

### Per-object predicted-track comparison

| Object | Pairs | Previous pivot | Shared pivot | Previous axis ° | Shared axis ° |
|---|---:|---:|---:|---:|---:|
| rec_cantilever_articulated_arm_03eac2c3c56c4eb8b696f08713ff0f23 | 11 | 0.005424 | 0.006968 | 0.1798 | 0.2177 |
| rec_cantilever_articulated_arm_0c46128c7eda409b9d37704d9aea8000 | 13 | 0.002801 | 0.004830 | 0.1795 | 0.1667 |
| rec_cantilever_articulated_arm_24c67e93e3fa4c3990061124f78ec3a3 | 12 | 0.007326 | 0.010729 | 0.3942 | 0.3998 |
| rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 12 | 0.005136 | 0.007616 | 22.5970 | 22.6009 |
| rec_robotic_arms_010b00e58fc346aaa15598c0cf170a6c | 12 | 0.008616 | 0.011198 | 1.1094 | 1.0775 |
| rec_robotic_arms_047c9d09d0e54055947483c5c5d62db8 | 12 | 0.007802 | 0.011035 | 0.2875 | 0.2536 |
| rec_robotic_arms_22664768f47842089f994d0b19305bd6 | 14 | 0.053507 | 0.056362 | 44.6952 | 44.6611 |
| rec_robotic_arms_9ff7f2676a894797a6fdfa7cf09f98b8 | 16 | 0.008223 | 0.009768 | 0.3484 | 0.3534 |
| rec_serial_elbow_arm_1a535fdff36e42e18f0fa70bd3fbe662 | 8 | 0.012845 | 0.013681 | 0.1328 | 0.1658 |
| rec_serial_elbow_arm_4ae0d61630354d279fa240118d7617cd | 8 | 0.007896 | 0.008561 | 0.3165 | 0.3488 |
| rec_serial_elbow_arm_b6f8d1f7bbc449cbaa3e50bc173c3be9 | 8 | 0.012387 | 0.015543 | 0.3835 | 0.3823 |
| rec_serial_elbow_arm_dfee8ddd003d4de1b9f8f7eba24f5351 | 8 | 0.009991 | 0.010571 | 0.3139 | 0.3289 |

## Interpretation

The unchanged orientation patch remains strong direction evidence, but its existing midpoint median is less effective for pivot localization than the symmetric-patch mean used in the preceding run at these fixed fitting strengths. This result concerns the entire position-evidence substitution (support, center and covariance together); it does not isolate the effect of the center statistic alone. It also does not establish that sharing a better contact patch would fail.

The shared run still improves pivot distance relative to the original weak-position orientation fitter, but that comparison includes the stronger adaptive position prior and cannot attribute the gain to sharing. No further contact-area changes or strength tuning were made after observing these results.

Pivot errors measure perpendicular distance from the transform-consistent GT pivot to the predicted infinite axis line, in dataset world units. Moving a pivot along its axis does not improve the metric. Axis errors are sign-invariant and rotation errors average states 2/3/4. Reprojection RMSE is against predicted tracks in source-image pixels. The same masks and assumed-revolute cohort are retained.

Both cohorts were evaluated previously. Development is repeatedly used; validation's 12 object identities do not overlap development's 14, but its saved predictions are not a new untouched test. Bootstrap intervals are descriptive and use scene clusters on development and object clusters on validation. The estimator receives only state-zero mesh geometry, initial anchors, camera and trajectories; GT articulation parameters enter scoring only. Full-mesh inputs and prior axis/angle limitations remain.

## Verification

Seventeen tests passed, including a test that both priors receive the identical contact object and that every array/scalar in it remains unmodified, plus synthetic exact recovery and existing geometry/refinement tests. The three new tests add evidence-reuse and fallback coverage.

The independent audit passed for all **2,028 saved fits**: 1,224 development and 804 validation (three fitters × predicted/GT tracks). All **676** shared-patch fits actually used the shared object; no insufficient-geometry fallback occurred. Axis, pivot, rotation and reprojection metrics were recomputed from saved parameters and all scored depths were positive. All freshly rerun comparison baselines reproduced their historical parameters within 1e-12. The source snapshot was hash-verified. The original contact extractor and previous fitters are unchanged.

See `results/protocol.json`, `results/audit.json`, `results/comparison.json`, the per-cohort `variants/` and `controls/`, and `results/code/` for reproducible inputs, parameters and sources.

![Fixed contact-patch comparison](results/comparison.png)
