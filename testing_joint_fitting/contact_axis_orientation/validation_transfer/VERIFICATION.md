# Verification — 2026-10-21

- Confirmed object IDs from dataset metadata: train 57 objects / 4,000 scenes;
  test1 14 objects / 320 scenes; validation 12 objects / 320 scenes. Validation
  overlaps neither test1 nor training. Both the development 80 test1 scenes and
  remaining 240 scenes contain all the same 14 test1 objects.
- Followed the user's reduced scope: completed inference for **50 scenes only**,
  selected by seeded round-robin object coverage. Every selected scene's
  eligible joint was enumerated: 142 source joints, 137 eligible pairs,
  including 134 revolute pairs and three fixed pairs. Four revolute joints and
  one fixed joint lack the existing required initial visibility.
- The earlier full-320 inference attempt failed at the first missing DINO
  feature, before producing predictions. Its caching process was stopped when
  the user limited scope to 50. The superseded protocol is retained separately.
- Selected-subset caching: 137 pair images checked, two hits and 135 newly
  computed features; 30.85 seconds after initialization. Encoder was frozen
  FP32, stored features FP16, matching the original cache namespace.
- Inference: original step-108000 checkpoint, five hypotheses, 16 steps,
  without-poke mode, FP32, both available GPUs. Both workers completed in
  approximately 85 seconds after initialization, with 137 cache hits and zero
  cache misses. Checkpoint SHA256 matches the development evaluation.
- Inference audit passed for all 137 persisted prediction arrays. No skipped
  or unscoreable future trajectories in the selected cohort.
- Fitting: four fixed configurations on the exact same 134 revolute pairs;
  four separate GT-track controls. No fitting source, threshold, strength or
  model weight was altered from the recorded protocol.
- Independent audit passed for **1,072 result rows**. Maximum disagreement:
  axis 3.09e-7 degrees, pivot 1.78e-15 world units, rotation 0.000453 degrees,
  reprojection RMSE 2.31e-14 pixels. No failed fits or nonpositive scored depths.
- Plot `results/generalization.png` was opened and inspected. Its three panels
  show distributions, per-object means and per-joint means correctly.
- Production and training code remain unchanged; tracked-file git diff is
  empty. All new evaluation utilities are isolated in this directory.

The frozen candidate reduces mean axis error from 49.5795 to 6.23967 degrees
versus the old RGB-D contact-position prior, improving the mean for all 12
validation objects. Nine large prediction errors occur on two objects. Four
GT-track cases expose new large axis errors induced by the orientation prior;
the worst is 78.1393 degrees versus 0.398758 degrees with the old position prior.
These limitations are reported without retuning on validation.
