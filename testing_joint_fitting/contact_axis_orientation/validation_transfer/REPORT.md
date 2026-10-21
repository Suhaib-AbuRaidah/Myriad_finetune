# Frozen contact-axis fitting on validation objects

## Object split and scope

The 80-scene fitting development subset covers all 14 test1 objects. The remaining **240** test1 scenes also contain those same 14 objects. They provide additional poses/configurations, not additional object identities.

Validation contains 12 object IDs with **zero overlap** with test1/development and with the 57 training object IDs. The requested evaluation uses **50 validation scenes**, selected by seeded round-robin object coverage: four scenes per object, plus one additional scene for two objects. Every eligible joint in those scenes was enumerated; none was chosen from its fitting score. The remaining 270 validation scenes were not inferred.

The 50 scenes contain 142 source joints, 137 initially eligible parent–child pairs, and 134 eligible revolute pairs. The axis comparison scores 134 revolute pairs across 50 scenes. There are 0 revolute pairs with unscoreable future trajectories and 4 source revolute joints excluded by the original initial-image eligibility rule.

Fixed joints retain inference outputs but are outside this assumed-revolute fitting experiment, consistent with development. Source-joint exclusions are listed in the inference manifest; skipped examples remain explicit in `results/cohort.json`.

## Frozen protocol

The exact development checkpoint is used: step 108000, SHA256 `8ec2c744e165f5994492da735340318b8be77ec00d333b6218915edc24e8899e`. Original uncropped images, 384 model resolution, 16 diffusion steps, five hypotheses, hypothesis zero, without-poke mode, and fitting states 2/3/4 match the development evaluation. Missing original-image DINO features were computed once with the same frozen encoder and stored in the existing feature cache. The model ran in FP32. No weights were retrained.

The recommended strength-128 fitting configuration and source hashes were recorded in `protocol.json` before validation fitting. No thresholds, prior strengths, point selection rules or candidates were tuned on validation. GT tracks are used only in the explicitly separate diagnostic controls, and GT joint parameters only for scoring.

## Main paired results

| Fitter | Pairs | Mean axis ° | Median axis ° | Axis ≤15° | Mean pivot distance | Mean rotation ° | Mean reprojection RMSE px | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 134 | 49.291 | 49.932 | 11.9% | 0.40400 | 41.703 | 8.156 | 0 |
| rgbd_position | 134 | 49.579 | 49.244 | 9.0% | 0.08240 | 42.231 | 7.778 | 0 |
| mesh_position | 134 | 50.010 | 51.032 | 9.7% | 0.08331 | 41.325 | 7.766 | 0 |
| frozen_candidate | 134 | 6.240 | 0.108 | 93.3% | 0.04478 | 31.582 | 10.505 | 0 |

Pivot distance is measured in dataset world units. Reprojection is against the predicted tracks in original 512-image pixels, not ADE against GT. All fitters use exactly the same pairs, anchors, masks and predictions. `mesh_position` isolates richer mesh inputs from adding the normal-based orientation prior.

![Validation comparison](results/generalization.png)

## Comparison with fitting development

| Dataset | Scenes | Distinct objects | Pairs | Candidate mean axis ° | Median axis ° | Axis ≤15° | Mean pivot | Mean rotation ° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test1 development | 80 | 14 | 204 | 2.390 | 0.124 | 97.5% | 0.04769 | 29.255 |
| Validation transfer | 50 | 12 | 134 | 6.240 | 0.108 | 93.3% | 0.04478 | 31.582 |

## Per-object comparison

Mean axis error improves for **12/12 objects** and **125/134 pairs** relative to the old visible-RGB-D contact-position prior.

| Object ID | Pairs | Old mean axis ° | Frozen mean axis ° | Frozen median axis ° | Old pivot | Frozen pivot | Frozen rotation ° |
|---|---:|---:|---:|---:|---:|---:|---:|
| rec_cantilever_articulated_arm_03eac2c3c56c4eb8b696f08713ff0f23 | 11 | 51.962 | 0.110 | 0.058 | 0.07356 | 0.03658 | 13.248 |
| rec_cantilever_articulated_arm_0c46128c7eda409b9d37704d9aea8000 | 13 | 54.677 | 0.145 | 0.067 | 0.07277 | 0.03584 | 36.882 |
| rec_cantilever_articulated_arm_24c67e93e3fa4c3990061124f78ec3a3 | 12 | 31.556 | 0.215 | 0.093 | 0.05461 | 0.04032 | 6.011 |
| rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 12 | 58.582 | 22.563 | 0.129 | 0.05482 | 0.02080 | 16.434 |
| rec_robotic_arms_010b00e58fc346aaa15598c0cf170a6c | 12 | 35.216 | 0.622 | 0.559 | 0.08169 | 0.03335 | 40.962 |
| rec_robotic_arms_047c9d09d0e54055947483c5c5d62db8 | 12 | 68.474 | 0.177 | 0.061 | 0.10482 | 0.09172 | 67.104 |
| rec_robotic_arms_22664768f47842089f994d0b19305bd6 | 14 | 56.444 | 38.513 | 1.419 | 0.16269 | 0.06447 | 57.131 |
| rec_robotic_arms_9ff7f2676a894797a6fdfa7cf09f98b8 | 16 | 54.993 | 0.254 | 0.095 | 0.06982 | 0.04205 | 47.963 |
| rec_serial_elbow_arm_1a535fdff36e42e18f0fa70bd3fbe662 | 8 | 50.735 | 0.085 | 0.111 | 0.08237 | 0.04816 | 7.195 |
| rec_serial_elbow_arm_4ae0d61630354d279fa240118d7617cd | 8 | 50.636 | 0.268 | 0.202 | 0.09526 | 0.03206 | 12.041 |
| rec_serial_elbow_arm_b6f8d1f7bbc449cbaa3e50bc173c3be9 | 8 | 26.941 | 0.231 | 0.218 | 0.04938 | 0.04395 | 17.327 |
| rec_serial_elbow_arm_dfee8ddd003d4de1b9f8f7eba24f5351 | 8 | 42.341 | 0.274 | 0.152 | 0.06550 | 0.04122 | 22.612 |

Equal-object macro means (each object counts once):

| Fitter | Axis ° | Pivot distance | Rotation ° |
|---|---:|---:|---:|
| rgbd_position | 48.546 | 0.08061 | 40.248 |
| mesh_position | 48.893 | 0.08134 | 39.467 |
| frozen_candidate | 5.288 | 0.04421 | 28.743 |

## Per-joint comparison

| Joint ordinal | Pairs | Old mean axis ° | Frozen mean axis ° | Axis ≤15° | Old pivot | Frozen pivot | Frozen rotation ° |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 47.655 | 2.156 | 98.0% | 0.10411 | 0.04962 | 27.618 |
| 2 | 49 | 48.725 | 9.243 | 89.8% | 0.06405 | 0.04614 | 23.888 |
| 3 | 31 | 52.710 | 8.837 | 90.3% | 0.08188 | 0.03649 | 51.941 |
| 4 | 4 | 59.839 | 0.365 | 100.0% | 0.03975 | 0.03195 | 17.600 |

## Object-level uncertainty

Paired bootstrap intervals below resample **objects**, preserving each object's scenes and joints. There are only 12 held-out object identities; intervals are descriptive, not a broad category-generalization guarantee.

- axis_error_with_failure_penalty_deg: mean candidate-minus-baseline change -43.339794; 95% interval [-51.709184, -34.746692].
- pivot_distance: mean candidate-minus-baseline change -0.037615; 95% interval [-0.053818, -0.024651].
- rotation_error_deg: mean candidate-minus-baseline change -10.649345; 95% interval [-14.346291, -7.516247].

## GT-trajectory controls

These runs test geometry-prior bias on the same cohort with GT trajectories. They never select a new configuration.

| Fitter | Mean axis ° | Maximum axis ° | Mean pivot | Mean rotation ° |
|---|---:|---:|---:|---:|
| original | 0.71316 | 72.50689 | 0.0011809 | 1.30622 |
| rgbd_position | 0.78172 | 71.17145 | 0.0013621 | 1.32755 |
| mesh_position | 1.09727 | 73.12899 | 0.0029046 | 1.40643 |
| frozen_candidate | 2.15026 | 78.13934 | 0.0011725 | 1.15384 |

**The controls expose prior bias on new objects.** The frozen prior introduces 4 axis errors above 15° on pairs for which the previous position prior has an accurate axis. These failures persist with GT trajectories, so they cannot be attributed only to Myriad prediction noise. A dominant contact-normal direction is not always a reliable hinge direction, and the fixed prior can overpower good trajectory evidence. This is a reason to keep the candidate experimental.

| Pair | Object ID | Joint ordinal | Previous GT-track axis ° | Frozen GT-track axis ° | Frozen RMSE px |
|---|---|---:|---:|---:|---:|
| 7 | rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 3 | 0.39876 | 78.139 | 0.665 |
| 19 | rec_robotic_arms_22664768f47842089f994d0b19305bd6 | 2 | 0.01012 | 65.206 | 5.593 |
| 66 | rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 3 | 0.00210 | 53.065 | 2.969 |
| 118 | rec_cantilever_articulated_arm_2fda465c8fa24bd79052a17771e312a5 | 3 | 0.00595 | 75.385 | 3.436 |

## Model trajectory quality on this cohort

The fitting algorithm does not change the Myriad tracks. These diagnostics help separate model motion errors from geometric fitting errors; they are identical across fitters.

- ADE_1_pixels: mean 11.85614, median 8.90382, n=134.
- ADE_percent_full_path: mean 70.06391, median 57.71494, n=134.
- minADE_1: mean 0.00914, median 0.00353, n=134.
- gt_path_pixels: mean 28.58429, median 20.94826, n=134.

## Interpretation and limits

This split checks transfer of the frozen fitting rules to different object identities. The objects remain robotic arms and can share procedural shape families. It does not establish transfer to unrelated object categories.

The neural checkpoint was selected using validation metrics during its original training. This is an independent object split for fitting development; it is not an untouched end-to-end model-selection test set.

The strongest fitter requires full initial mesh samples and their normals. It never receives future mesh states or true articulation parameters. Any eventual RGB-D-only deployment needs its own check.

Axis direction alone does not determine motion: angle magnitude, angle sign and pivot still affect the recovered transformation. Large errors and fit failures are retained in the per-pair JSON results. There are 9 candidate pairs above 15° axis error.

All 9 large predicted-track axis errors are concentrated in 2 object identities. The other 125 pairs have 0.247° mean axis error. The higher validation mean versus development is driven by these specific contact-geometry failures. No validation failures were used to retune or replace the frozen configuration.

Frozen inputs, object membership, exclusions, per-pair fit parameters, controls and source hashes are saved beside this report. The aborted full-320 setup is recorded separately; only the selected 50 scenes received completed inference in this evaluation.
