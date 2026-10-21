# Implementation verification

The current reviewed manifest selects **100 unique revolute pairs from 100
scenes and all 12 object identities**, out of 854 scoreable revolute validation
pairs. All 100 enter the axis-scoring cohort. It excludes 27 fixed pairs and
four revolute pairs with insufficient correspondences; this split has no
eligible prismatic pairs. Only one object category is available.

Selection follows proportional object quotas and population marginal counts.
The measured total variation distances between selected and eligible
population distributions are:

| Dimension | Total variation distance |
|---|---:|
| Object identity | 1.611% |
| Joint ordinal / chain depth | 0.529% |
| Motion quartile | 0.117% |
| Apparent child-size quartile | 0.176% |
| Crop-magnification quartile | 1.293% |
| Viewpoint | 3.281% |

The 13-test CPU suite passed in 3.889 seconds, including exclusion of otherwise
scoreable fixed/prismatic pairs and an imbalanced 80% / 15% / 5% synthetic
population whose selected object counts match 80 / 15 / 5 exactly. The real
manifest was built without a model, GPU training, or checkpoint writes. Each
selected joint type was independently checked against its source archive, and
training configuration was verified to reuse the reviewed manifest exactly.
Temporary audit files were removed.

The historical GPU measurements below used the previous mixed-type cohort
and are not measurements of the new selection.

Measured in `myriad_finetune` on two RTX 2080 SUPER 8 GiB GPUs.

The bounded real run performed two full-FP32 updates with microbatch 1,
accumulation 4, eight child slots, four transitions, and activation checkpointing.
Its one training scene contained two eligible pairs; all pairs were visited
exactly once per epoch across four completed epochs.

| Measurement | Result |
|---|---:|
| First update, including warmup | 23.257 s |
| Second update, four examples | 1.395 s |
| Second-update optimizer transfers/update | 0.207 s |
| Second-update throughput | 2.868 examples/s |
| Peak allocated motion GPU (`cuda:1`) | 2916.8 MiB |
| Peak allocated AdamW GPU (`cuda:0`) | 5516.4 MiB |
| Training losses | 1.086640, 1.007312 |
| Gradient norms before clipping | 25.995108, 23.408804 |
| Valid cropped feature entries | 102 / 102 |
| Full validation elapsed | 164.623 s |
| Validation pairs / GT-revolute cohort | 100 / 91 |
| Mean sign-invariant axis error | 57.168037 degrees |
| Fit failures | 0 |
| Originally valid future targets outside crop | 100 / 3174 (3.151%) |

Validation covered 100 scenes and 12 objects. Its score included six GT-revolute
pairs classified as prismatic. Eligible validation examples include revolute
and fixed joints; no eligible prismatic examples and only one object category
are available. These are smoke-test scores from random initialization, not
evidence of a trained model.

The earlier 12-test CPU regression suite passed in 49.499 seconds. Checkpoint
serialization and restoration run entirely in memory, without creating checkpoint
files. Checks cover exact resumed model/AdamW/scheduler/RNG/pair cursor state,
FPS uniqueness and padding, crop reprojection and coordinate round trips,
persistent cache hits/corruption/coverage, no encoder construction after cache
completion, immutable 100-pair selection, soft contact priors and occlusion
fallback, axis-only scoring/failure penalties/ties, and smoke-mode defaults and
automatic temporary-output cleanup.

The initial attempt exposed movable-joint versus fixed-edge indexing during
stratification; this was corrected and covered by a regression test.

All generated smoke checkpoints and incomplete checkpoint writes were deleted
on request, along with smoke run directories and redundant target/image caches.
Further smoke runs disable checkpoint writes and use temporary output folders.

A real GPU checkpoint-resume test was not completed; CPU serialization/resume
was verified. The full training dataset has not been trained or precomputed.
Long-run convergence, arbitrary RGB-D uploads, and prismatic validation
performance remain unverified. No long training run was started.
