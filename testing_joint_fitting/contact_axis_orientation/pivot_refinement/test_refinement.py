"""Synthetic checks of soft centering, pivot gauge and missing-geometry fallback."""
import unittest
from dataclasses import replace

import cv2
import numpy as np

from finetuning.joint_parame_estim.run import project, axis_distance, axis_error
from ..data import Inputs
from .solver import fit, position_contact


class PivotRefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6302)
        self.xyz = rng.uniform(-.1, .1, (40, 3))+[.05, 0., 2.]
        self.axis = np.array([0., 0., 1.])
        self.pivot = np.array([0., 0., 2.])
        self.angles = np.array([.2, .4, .6])
        k = np.array([[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]])
        uv = np.zeros((5, len(self.xyz), 2))
        for state, angle in zip((2, 3, 4), self.angles):
            world = (self.xyz-self.pivot)@cv2.Rodrigues(self.axis*angle)[0].T+self.pivot
            uv[state] = project(world, k, np.eye(4))
        circle = np.linspace(0, 2*np.pi, 64, endpoint=False)
        parent = np.c_[.03*np.cos(circle), .03*np.sin(circle), np.full(64, 1.995)]
        child = parent+[0., 0., .01]
        normals = np.tile(self.axis, (64, 1))
        self.data = Inputs(self.xyz, uv, np.ones(uv.shape[:2], bool), k,
            np.eye(4), parent, child, normals, normals, None)
        self.seed = dict(params=np.r_[self.axis, [.01, .01, 2.], self.angles], median_px=4.)
        self.config = dict(position_strength=64., position_count=64, center_method="mean")

    def test_exact_tracks_and_center_recover_the_axis_line(self):
        fitted = fit(self.data, self.config, self.seed)
        self.assertLess(axis_error(fitted["params"][:3], self.axis), .01)
        self.assertLess(axis_distance(fitted["params"][3:6], fitted["params"][:3], self.pivot), 1e-6)
        self.assertLess(fitted["rmse_px"], 1e-5)

    def test_stronger_soft_constraint_reduces_noisy_track_displacement(self):
        uv = self.data.uv.copy()
        uv[2:5] += [8., -5.]
        data = replace(self.data, uv=uv)
        weak = fit(data, dict(self.config, position_strength=4.), self.seed)
        strong = fit(data, self.config, self.seed)
        distance = lambda r: axis_distance(r["params"][3:6], r["params"][:3], self.pivot)
        self.assertLess(distance(strong), distance(weak)*.5)
        self.assertLess(axis_error(strong["params"][:3], self.axis), 1.)

    def test_pivot_gauge_does_not_change_the_fitted_line(self):
        a = fit(self.data, self.config, self.seed)
        seed = dict(self.seed, params=self.seed["params"].copy())
        seed["params"][3:6] += 100.*self.axis
        b = fit(self.data, self.config, seed)
        np.testing.assert_allclose(a["params"], b["params"], atol=1e-7)

    def test_mean_center_is_rigid_frame_equivariant(self):
        rotation = cv2.Rodrigues(np.array([.4, -.2, .8]))[0]
        scale, translation = 7., np.array([1., 2., -3.])
        a = position_contact(self.data, self.config)
        data = replace(self.data,
            parent=scale*self.data.parent@rotation.T+translation,
            child=scale*self.data.child@rotation.T+translation)
        b = position_contact(data, self.config)
        np.testing.assert_allclose(b["center"], scale*a["center"]@rotation.T+translation, atol=1e-12)

    def test_missing_geometry_retains_original_seed(self):
        data = replace(self.data, parent=np.empty((0, 3)), child=np.empty((0, 3)),
            parent_normals=np.empty((0, 3)), child_normals=np.empty((0, 3)))
        self.assertIs(fit(data, self.config, self.seed), self.seed)

    def test_adaptation_preserves_more_track_influence_for_clean_data(self):
        config = dict(self.config, noise_reference_px=1.)
        clean = fit(self.data, config, dict(self.seed, median_px=0.))
        noisy = fit(self.data, config, dict(self.seed, median_px=8.))
        self.assertLess(clean["diagnostics"]["position_strength"], 16.)
        self.assertGreater(noisy["diagnostics"]["position_strength"], 63.)

    def test_recommended_public_interface_reconstructs_transforms(self):
        from .candidate import estimate_motion
        d = self.data
        result = estimate_motion(d.xyz, d.uv, d.valid, d.k, d.e,
            parent_points=d.parent, child_points=d.child,
            parent_normals=d.parent_normals, child_normals=d.child_normals)
        self.assertIsNotNone(result)
        self.assertEqual(result["states"], [2, 3, 4])
        self.assertLess(axis_distance(result["pivot"], result["axis"], self.pivot), 1e-5)
        for angle, transform in zip(result["angles"], result["transforms"]):
            rotation = cv2.Rodrigues(result["axis"]*angle)[0]
            np.testing.assert_allclose(transform[:3, :3], rotation)
            np.testing.assert_allclose(transform[:3, 3], result["pivot"]-rotation@result["pivot"])


if __name__ == "__main__":
    unittest.main()
