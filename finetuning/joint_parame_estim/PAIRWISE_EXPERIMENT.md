# Parent-child-only trajectory experiment

`pairwise_subset.py` tests whether showing MYRIAD only a joint's parent and child points improves trajectory prediction and joint estimates. It visits joints from the fixed base toward the leaves. For every joint it sends the **same reference image** but only the selected parent and child points to the fine-tuned checkpoint. With poke, it supplies both observed states; without poke, only state 0. The model predicts both links' later points. The fitter holds the base stationary, carries each fitted parent transform down the chain, and separately fits the child's fixed, prismatic, and revolute motions. It uses the same no-complexity-penalty selection rule as `run.py`.

The fair baseline takes those **same parent and child points** from the saved full-scene MYRIAD rollout and runs the **same base-to-leaf fitter**. Both methods use hypothesis 0, states 2–4, the same point-validity mask, and the same 5-hypothesis/16-step sampler settings. Joints are processed in hierarchy order, but each MYRIAD call is independent: the sampler API does not accept a previously predicted parent trajectory as a fixed condition for a later call. The previously fitted parent transform is used in joint fitting, not as an input constraint to MYRIAD.

By default, the script selects eight deterministic test1 scenes: four containing a fixed joint and four without one. With seed 42, these are scenes `10, 33, 67, 113, 171, 191, 220, 239` (26 ground-truth joints). This intentionally oversamples fixed joints; subset percentages should not be presented as whole-test1 estimates.

Run the saved-prediction baseline without loading the model:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m finetuning.joint_parame_estim.pairwise_subset --baseline-only
```

On a machine with CUDA and the existing `myriad_finetune` environment, run the pairwise inference and comparison:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m finetuning.joint_parame_estim.pairwise_subset --device cuda:0 --amp-dtype bf16
```

The output goes to `finetuning/runs/pairwise_joint_test1/`. `summary.json` includes both all-evaluable and matched-joint comparisons; `per_joint.csv` includes type, axis, pivot, fit, and trajectory errors. Per-pair `predictions/*.npz` files are cached so an interrupted run can resume without repeating completed model calls. The checkpoint is loaded from its existing location and is never copied into the output folder. Use `--scene-indices 10,33,...` to choose a different small subset.

The pairwise CUDA run completed on the eight selected scenes. It saved 48 per-joint prediction archives and evaluated the same 24 of 26 joints in each mode as the full-scene baseline. Every baseline type decision and axis error matches the existing full-scene forward-chain evaluation on these joints.

| Metric ↓ unless marked ↑ | Without poke: full → pair | With poke: full → pair |
|:--|--:|--:|
| Mean child trajectory ADE, source-image px | 53.9 → **46.1** | 28.7 → **21.3** |
| Joint-type accuracy ↑ | 79.2% → 79.2% | 83.3% → 75.0% |
| Fixed-joint recall ↑ | 0/4 → 0/4 | 0/4 → 0/4 |
| Mean axis error, correctly typed revolute joints | 48.0° (n=19) → 59.5° (n=19) | 28.6° (n=20) → 26.8° (n=18) |
| Mean axis error on **the same correctly typed joints** | 48.4° → 58.1° (n=18) | 26.6° → 26.8° (n=18) |
| Mean pivot distance on those same joints | 0.350 → 0.626 | 0.101 → 0.136 |

Pairwise input reduced mean child-point trajectory error by 14.5% without poke and 25.5% with poke. Seventeen of 24 joints improved in each mode; at the scene level, four of eight improved without poke and six of eight with poke. **This did not improve the inferred joint parameters.** Without poke, matched-joint mean axis error rose by 9.7°; with poke it was essentially unchanged, and two revolute joints switched to prismatic. All four fixed joints were still called revolute in both modes. Because this is a small subset chosen to include fixed joints and only hypothesis 0 was assessed, these numbers are an exploratory result, not a whole-test1 estimate.

## Ground-truth revolute joints only

Filtering the saved per-joint records by `gt_type == revolute` removes all fixed joints **from scoring**, without changing the earlier model rollouts or the fitted parent chains. There are 22 ground-truth revolute joints in these scenes; both methods estimated the same 20 in each poke mode. The filtered records are in `finetuning/runs/pairwise_joint_test1/per_joint_revolute_only.csv`, and the machine-readable metrics are in `summary_revolute_only.json` beside it.

| Metric ↓ unless marked ↑ | Without poke: full → pair | With poke: full → pair |
|:--|--:|--:|
| Revolute joints estimated / all | 20 / 22 → 20 / 22 | 20 / 22 → 20 / 22 |
| Correct type among estimated joints ↑ | 95% (19/20) → 95% (19/20) | 100% (20/20) → 90% (18/20) |
| Mean child trajectory ADE, source-image px | 54.8 → **48.0** | 28.6 → **20.2** |
| Mean axis error, correctly typed joints | 48.0° (n=19) → 59.5° (n=19) | 28.6° (n=20) → 26.8° (n=18) |
| Mean axis error on the **same correctly typed joints** | 48.4° → 58.1° (n=18) | 26.6° → 26.8° (n=18) |
| Mean pivot distance on those same joints | 0.350 → 0.626 | 0.101 → 0.136 |

Removing fixed joints raises the reported type accuracy, but does not change the axis-error numbers from the main table: those metrics already used correctly classified revolute joints only. Pairwise input improves 2D point error on this subset, while axis estimation does not improve on a matched-joint comparison.
