# Exact algorithm behind `ambiguous_mixture_strength128`

This describes the frozen configuration in `candidate.py`, with implementation
in `geometry.py` and `solver.py`. It documents the algorithm without changing it
or running an experiment.

```python
mode = "ambiguous_mixture"
strength = 128.0
quadratic = True
starts = True
normal_method = "paired"
orientation_count = 64
position_strength = 4.0
```

## Inputs, motion model and initialization

Inputs are the initial child keypoints in world coordinates, predicted 2D tracks
in original-image pixels, correspondence validity masks, camera intrinsics and
world-to-camera extrinsics, and initial parent/child mesh point clouds and
normals. The current five-state pipeline fits predicted states 2, 3 and 4.
The predicted motion is relative to a parent frozen at its initial pose.
No GT articulation axis, pivot or future mesh geometry enters the estimator.
Evaluation cohort/type labels and visibility masks are separate protocol choices.

The estimator assumes a revolute joint. It estimates a shared unit axis `a`, a
point `p` on its infinite line, and three rotation angles `theta_t`:

\[
 R_t=\operatorname{Rodrigues}(a\theta_t),\qquad
 X_{ti}=R_t(X_i-p)+p,\qquad
 T_t=\begin{bmatrix}R_t&(I-R_t)p\\0&1\end{bmatrix}.
\]

Camera projection of `X_ti` is compared with the model's predicted pixel `u_ti`.
This is one shared hinge with separate angles, rather than independent rigid
transforms for every frame.

The initial estimate comes from the original `fit_joint`: calibrated PnP for
each predicted state, an initial axis from those rotations, a pivot from their
rotation/translation constraints, and a trajectory-only revolute refinement.
Although that helper also evaluates fixed/prismatic models, the candidate always
takes its `fits["revolute"]` result. It does not inherit the helper's type choice.

## Estimate the contact region from initial mesh samples

For every child mesh sample, find the nearest parent sample with a KD-tree.
Sort these child-to-parent pairs by gap. This configuration does not enable the
optional symmetric reverse search.

- The closest **32 pairs** define the position prior.
- The closest **64 pairs** define the orientation evidence.
- If a part has fewer samples, all available pairs up to that count are used.

Pair midpoints approximate a possible contact region. Its center `c` is their
coordinate-wise median. A covariance `C` represents its spatial extent, with
an isotropic uncertainty floor added. The floor is the maximum of half the
smaller part's RMS radius, the median selected gap, and `1e-5` world units.
Thus the prior allows a joint axis to lie inside a part rather than exactly on
one sampled surface. This is a geometric contact proxy, not a known physical
connection or a GT joint-location annotation.

The position residual is the distance from that region's center to the fitted
**infinite axis line**, measured after covariance whitening. With `W C W^T=I`,
`d=W a`, and `b=W(p-c)`, it is

\[
 r_{\rm pos}=4\sigma\left(b-d\frac{d^T b}{d^T d}\right).
\]

It encourages the axis line to pass near the contact region. It does not pin
the particular pivot representative `p` to the center `c`.

## Obtain preferred directions from contact normals

The orientation evidence uses the surface normals at the selected nearest
parent/child pairs. After unit normalization, pair `j` receives weight

\[
 w_j=|n_{pj}^T n_{cj}|^8.
\]

Locally parallel or opposing surface normals get high weight, while mismatched
orientations get low weight. Reversing a normal's sign has no effect.
The two surfaces contribute equal total mass to the second-moment matrix:

\[
 M=\frac{\sum_j w_j(n_{pj}n_{pj}^T+n_{cj}n_{cj}^T)}{2\sum_j w_j}.
\]

The code first checks for enough finite, nonzero distinct selected normals.
If paired weights are negligible it retains its fallback moment from the usable
deduplicated normals. If usable normal evidence is insufficient, fitting falls
back to the mesh position prior only. If mesh contact itself cannot be formed,
it returns the original trajectory-only seed.

Diagonalize `M` to obtain eigenvalues `lambda_0 <= lambda_1 <= lambda_2` and
orthonormal eigenvectors `e_0, e_1, e_2`. This is an **uncentered normal second
moment**, not PCA of contact point positions. Contact-position covariance is
used for the position prior; its longest direction is not the orientation target
of this frozen candidate configuration.

## What `ambiguous_mixture` means

Define the leading-direction separation

\[
 c_n=\frac{\lambda_2-\lambda_1}{\max(\lambda_2,10^{-15})}.
\]

If `c_n >= 0.2`, use the `normal_field` prior. Its dimensionless residual vector
is

\[
 g(a)_j=\sqrt{\max(0,1-\lambda_j/\lambda_2)}\ e_j^T a.
\]

It favors the dominant normal direction while penalizing the less-supported
components according to the eigenvalue gaps. It remains a soft constraint.

If `c_n < 0.2`, the two strongest normal directions are insufficiently separated.
Run **three alternative fits**, respectively favoring `e_2`, `e_1`, and `e_0`:

\[
 g_j(a)=a\times e_j.
\]

Choose the solution with the smallest **full fitting objective**. Do not average
the axes and do not select by reprojection error alone.

For example, a planar mating face can supply one strong normal direction, while
normals around a cylindrical interface can spread over a plane. In the latter
case the physical hinge axis can be the smallest-eigenvalue direction, so always
choosing the largest normal eigenvector would miss a plausible hypothesis.

`ambiguous` names this eigenvalue-gap decision; `mixture` names the alternative
direction hypotheses. It is a discrete optimization choice, not a probabilistic
mixture with learned probabilities. All three ambiguous alternatives receive
confidence 1 and the full strength. Low confidence does not turn the prior off.
If `M` is nearly isotropic, its eigenvectors can be unstable, yet this branch
still uses them: this is a real limitation of the implementation.

## What `128` means

It is the orientation **residual multiplier**, `strength=128`, in

\[
 r_{\rm ori}=128\sigma\,g(a),\qquad
 \sigma=\max\left(0.25,\frac{m}{\sqrt{2\log 2}}\right),
\]

where `m` is the original seed's median pixel reprojection error. The same seed
and scale are shared by all alternatives and starts. `sigma` is a residual-scale
heuristic, not a calibrated uncertainty or a GT-derived error estimate.

The orientation loss is quadratic, so its objective contribution is

\[
 J_{\rm ori}=\tfrac12(128\sigma)^2\|g(a)\|^2.
\]

For alignment to one unit direction `v`, `||a x v||^2 = sin^2(angle(a,v))`.
Holding `sigma` fixed, changing strength from 128 to 256 quadruples the
orientation penalty for the same axis disagreement. The number is neither an
angle, an iteration count nor a sample count. `orientation_count=64` and
`position_strength=4` are separate settings.

Strength 128 was selected on the original 80-scene development experiment as a
compromise: 256 slightly improved mean predicted-track axis error but produced
a large error in a separate GT-trajectory control. This selection does not
establish a universally optimal or universally safe strength for new objects.

## Full objective and numerical solve

The optimized residual vector concatenates:

1. Pixel reprojection residuals against predicted tracks.
2. The contact-position residual above.
3. The contact-orientation residual above.
4. A penalty for transformed points behind the camera.

Pixel, position and depth components use soft-L1 loss with scale 8. Orientation
components use quadratic loss (`quadratic=True`), so large disagreement with
the geometry prior is not downweighted as an outlier. This deliberately gives
the contact orientation substantial influence.

Axis orientation uses two free parameters; the pivot has two after fixing its
unobservable coordinate along the line; and the three states have three angles.
There are **seven free variables**, although the output parameter vector has
nine entries: axis(3), pivot(3), angles(3).

Each orientation hypothesis is tried from three starts (`starts=True`):

1. The original trajectory-only axis and pivot line.
2. That axis with its pivot line passing through the estimated contact center.
3. A geometry-aligned axis with its pivot line passing through the contact center.

`scipy.optimize.least_squares` jointly adjusts axis, pivot and angles, with up to
100 function evaluations per start, Jacobian-based variable scaling, and
`ftol=xtol=gtol=1e-6`. Finite solutions with positive depths at scored observations
are retained; convergence status is recorded but is not itself a rejection rule.
First choose the best start, then the best orientation alternative, by full cost.

The output contains the unit axis, pivot representative, three angles and
corresponding 4x4 transforms. Axis sign is arbitrary provided angle signs change
with it. Moving the pivot along the axis leaves the motion unchanged.

## Interpretation of the observed errors

Geometry can make axis direction accurate even when the predicted trajectory
has substantial error. Rotation angle and pivot still depend on predicted
motion, so a good orientation score alone does not certify a good transform.
Conversely, a strong incorrect geometry prior can make an axis nearly 90 degrees
wrong while most other examples fit very accurately. The current results do
not by themselves prove which residual term caused an individual failure.

Source entry points: `candidate.py:estimate_motion`, `geometry.py:mesh_contact`,
`geometry.py:direction_options`, and `solver.py:fit`/`solve`.
