# Contact orientation experiment

This is an isolated development experiment. Production fitting, model weights,
training scripts, saved predictions, and the original subset are left untouched.

The fixed cohort is the original 80 test1 scenes, with their 204 scoreable
revolute pairs. Predictions are saved hypothesis zero, without a poke; fitted
states are 2, 3 and 4. Joint kind is assumed revolute throughout, so classification
does not hide difficult examples. Ground truth defines this cohort and scores
results; it never selects an axis candidate or fitting start.

Inputs add the dataset's sampled world-space mesh points, normals and link
labels at **state zero only**. Future point clouds, canonical joint axes/origins,
and future transforms are excluded from the estimator. Mesh geometry is richer
than the previous visible RGB-D contact input. Separate position-only controls
make that change explicit. Existing correspondence validity masks are retained
from the original evaluation; this is a fitting comparison, not a new deployed
visibility estimator.

Raw contact midpoint covariance defines principal directions. The covariance
floor used by the position prior is excluded when computing directional
confidence. Axis alignment is sign-invariant; perpendicularity constrains an
entire plane rather than an arbitrary perpendicular vector. The optimizer
jointly refits axis direction, pivot and per-frame rotation angles.

Inspiration: [Rest2Art, section 3.3](https://arxiv.org/html/2607.27749v1#S3.SS3).
This is an independently tested interpretation, not its unpublished code.

Run from the repository root, in the myriad_finetune environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -u -m testing_joint_fitting.contact_axis_orientation.run \
  --variants original rgbd_position mesh_position major_4 normal_4 --workers 4
```

Outputs persist per variant, including exact configurations, code snapshots,
source fingerprints, all pair scores and fitted parameters. The diagnosis
script's closest-principal-axis oracle is explicitly a ground-truth diagnostic,
never a usable estimator. Experiments use this development cohort repeatedly;
results are not claims of performance on untouched data.

The completed experiment and per-joint comparisons are in [REPORT.md](REPORT.md).
The subsequent frozen-object transfer check on 50 validation scenes is in
[validation_transfer/REPORT.md](validation_transfer/REPORT.md).
The frozen prototype is `candidate.estimate_motion`, with an explicit initial
mesh point/normal interface. Its recommended orientation strength is 128; the
slightly better development score at strength 256 was rejected after a large
axis error on a GT-track control. No production replacement has been made.

```bash
python -B -m unittest testing_joint_fitting.contact_axis_orientation.test_geometry -v
python -B -m testing_joint_fitting.contact_axis_orientation.audit
MPLCONFIGDIR=/tmp/myriad_contact_matplotlib python -B -m testing_joint_fitting.contact_axis_orientation.report
```

The local `results/cases.pkl` cache avoids repeatedly decompressing the original
evaluation inputs. Its manifest checks input file size/mtime and cache SHA256.
It contains no new network inference or DINO features and belongs only to this
experiment.
