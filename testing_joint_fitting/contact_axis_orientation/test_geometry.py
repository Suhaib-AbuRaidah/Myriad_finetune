"""Geometric invariants and exact-reprojection recovery, independent of test1 truth."""
import unittest
import cv2
import numpy as np

from .data import Inputs
from .geometry import mesh_contact, direction_options
from .solver import solve
from finetuning.joint_parame_estim.run import axis_error, axis_distance, project


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(6302)
        self.parent = self.rng.normal(size=(200, 3))*[.08, .025, .005]
        self.child = self.parent + self.rng.normal(scale=.002, size=(200, 3)) + [0, 0, .01]
        self.normals = np.tile([0., 0., 1.], (200, 1))

    def contact(self, parent=None, child=None, normals=None):
        return mesh_contact(self.parent if parent is None else parent,
            self.child if child is None else child,
            parent_normals=self.normals if normals is None else normals,
            child_normals=self.normals if normals is None else normals)

    def test_normal_sign_symmetry(self):
        a = self.contact()
        b = mesh_contact(self.parent, self.child, parent_normals=-self.normals, child_normals=self.normals)
        np.testing.assert_allclose(a["normal_spectrum"], b["normal_spectrum"])
        self.assertLess(axis_error(a["normal_axes"][:, 2], b["normal_axes"][:, 2]), 1e-5)

    def test_rigid_frame_and_scale_equivariance(self):
        rotation = cv2.Rodrigues(np.array([.4, -.2, .8]))[0]
        translation = np.array([1., 2., -3.]); scale = 7.
        a = self.contact()
        b = self.contact(scale*self.parent@rotation.T+translation,
            scale*self.child@rotation.T+translation, self.normals@rotation.T)
        np.testing.assert_allclose(a["eigenvalues"]*scale**2, b["eigenvalues"], rtol=1e-10)
        for i in range(3):
            self.assertLess(axis_error(rotation@a["axes"][:, i], b["axes"][:, i]), 1e-5)
        self.assertLess(axis_error(rotation@a["normal_axes"][:, 2], b["normal_axes"][:, 2]), 1e-5)

    def test_isotropic_normals_do_not_force_arbitrary_axis(self):
        contact = dict(axes=np.eye(3), linearity=0., planarity=0., normal_axes=np.eye(3),
            normal_spectrum=np.ones(3)/3)
        matrix, relation, _ = direction_options(contact, "normal_field")[0]
        self.assertEqual(relation, "subspace")
        np.testing.assert_allclose(matrix, 0.)

    def test_planar_normal_field_and_sign(self):
        contact = self.contact()
        matrix, _, _ = direction_options(contact, "normal_field")[0]
        self.assertLess(np.linalg.norm(matrix@np.array([0, 0, 1.])), 1e-10)
        self.assertGreater(np.linalg.norm(matrix@np.array([1., 0, 0])), .99)
        axis = np.array([.2, .3, .9]); axis /= np.linalg.norm(axis)
        self.assertAlmostEqual(float(np.sum((matrix@axis)**2)), float(np.sum((matrix@-axis)**2)))

    def test_unusable_normals_produce_position_evidence_only(self):
        contact = mesh_contact(self.parent, self.child,
            parent_normals=np.zeros_like(self.parent), child_normals=np.zeros_like(self.child))
        self.assertIsNotNone(contact)
        self.assertNotIn("normal_spectrum", contact)
        self.assertTrue(np.isfinite(contact["whitening"]).all())

    def test_point_input_order_cannot_change_full_patch_spectrum(self):
        parent_order = self.rng.permutation(len(self.parent))
        child_order = self.rng.permutation(len(self.child))
        a = self.contact()
        b = mesh_contact(self.parent[parent_order], self.child[child_order],
            parent_normals=self.normals[parent_order], child_normals=self.normals[child_order])
        np.testing.assert_allclose(a["eigenvalues"], b["eigenvalues"], rtol=1e-12)
        np.testing.assert_allclose(a["normal_spectrum"], b["normal_spectrum"], atol=1e-12)

    def test_exact_revolute_recovery(self):
        xyz = self.rng.uniform(-.1, .1, size=(40, 3))+[.05, 0., 2.]
        axis = np.array([0., 0., 1.]); pivot = np.array([0., 0., 2.])
        angles = np.array([.2, .4, .6])
        k = np.array([[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]])
        e = np.eye(4)
        uv = np.zeros((5, len(xyz), 2))
        for state, angle in zip((2, 3, 4), angles):
            transformed = (xyz-pivot)@cv2.Rodrigues(axis*angle)[0].T+pivot
            uv[state] = project(transformed, k, e)
        data = Inputs(xyz, uv, np.ones(uv.shape[:2], bool), k, e,
            self.parent, self.child, self.normals, self.normals, None)
        contact = self.contact(); contact["center"] = pivot
        seed = dict(params=np.r_[np.array([.05, 0., .99875]), [.01, .01, 2.], angles], median_px=2.)
        fitted = solve(data, seed, contact, (axis, "align", 1.), 16., True, True)
        self.assertIsNotNone(fitted)
        self.assertLess(axis_error(fitted["params"][:3], axis), .01)
        self.assertLess(axis_distance(fitted["params"][3:6], fitted["params"][:3], pivot), 1e-5)
        self.assertLess(fitted["rmse_px"], 1e-5)


if __name__ == "__main__": unittest.main()
