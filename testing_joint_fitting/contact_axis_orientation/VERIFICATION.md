# Verification — 2026-10-21

All experiments ran with the local `myriad_finetune` conda Python on CPU.
BLAS/OpenMP thread counts were one; batch comparisons used four CPU workers.

- **Frozen data:** original 80-scene subset; 204 scoreable GT-revolute pairs;
  saved uncropped hypothesis zero; without-poke mode; fitted states 2, 3, 4.
  Mesh inputs use state zero only. The remaining 240 scenes were not evaluated
  by this experiment.
- **Comparisons:** 50 predicted-trajectory configurations and seven explicitly
  separate GT-trajectory control configurations, each with all 204 pairs.
- **Synthetic tests:** seven passed: exact motion recovery, normal sign
  symmetry, rigid-frame/scale behavior of principal directions, isotropic
  degeneracy, planar normal constraint, sample-order invariance, invalid-normal
  handling.
- **Independent audit:** 11,628 fitted parameter sets checked. Maximum metric
  disagreement: axis 7.98e-7 degrees, pivot 1.78e-15 world units, rotation
  0.000510 degrees (the independent scorer polar-projects float32-derived GT
  rotations). All axes unit length within 4.45e-16; no nonpositive scored depths.
- **Public prototype replay:** `candidate.estimate_motion` reproduced all 204
  saved strength-128 parameter vectors with maximum difference **0.0**. Its
  output rotation matrices passed orthogonality checks. Sequential replay took
  22.34 seconds, excluding initial loading.
- **Visual check:** opened `results/comparison.png`; all three panels and their
  axes, legends and per-joint bars render correctly.
- **Isolation:** tracked-file `git diff --stat` is empty. Only the new
  `testing_joint_fitting/contact_axis_orientation/` directory was added by this
  task. The pre-existing untracked `training_from_scratch_cropped/` directory was
  left untouched. No model checkpoint, training script or production fitter
  was changed.

The recommended strength-128 variant has 2.38964-degree mean axis error,
0.0476888 mean pivot distance and 29.2553-degree mean rotation error. Five pairs
still exceed 15 degrees of axis error. Strength 256 gives a slightly lower
development mean but fails a GT-track control with a 61.53-degree axis error.
The report retains this regression rather than recommending the strongest prior.

Final Python-source snapshot SHA256:
`43d7421c5d1312708a3ad9ed95c0bb8e736b65b4e7caa8d6b7cdf25a9749eb89`.
Machine-readable evidence is in `results/audit.json`, `candidate_replay.json`,
`manifest.json`, `selected.json` and the per-configuration JSON files.
