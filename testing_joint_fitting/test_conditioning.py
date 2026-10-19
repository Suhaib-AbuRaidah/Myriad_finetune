"""Checks for inference-only conditioning and identifiable geometry recovery."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from finetuning.joint_parame_estim.run import project, axis_error, axis_distance
from .conditioning import fit_conditioned, rescore
from .geometric import refine_revolute, tangent_frame, surface_points
from .selection import select_shape_fit, global_joint_seed
from .candidate import estimate_motion


class ConditioningTests(unittest.TestCase):
    def scene(self):
        xyz=np.array([[.8,-.1,3.8],[1.,.1,4.],[.9,.2,4.1],[1.1,-.2,3.9],
                      [.7,.15,4.2],[1.2,.1,3.7],[.85,-.15,4.15],[1.05,.25,3.85]])
        pivot=np.array([0.,0.,4.])
        k=np.array([[500.,0,256],[0,500.,256],[0,0,1.]])
        e=np.eye(4)
        uv=np.array([project((xyz-pivot)@cv2.Rodrigues(np.array([0.,0.,a]))[0].T+pivot,k,e)
                     for a in (0.,.1,.2,.35,.5)])
        return xyz,pivot,k,e,uv,np.ones(uv.shape[:2],bool)

    def test_tangent_frame_is_orthonormal_and_perpendicular(self):
        u=np.array([.2,.3,.4]);u/=np.linalg.norm(u)
        b=tangent_frame(u)
        np.testing.assert_allclose(b.T@b,np.eye(2),atol=1e-14)
        np.testing.assert_allclose(b.T@u,0,atol=1e-14)

    def test_adding_early_state_keeps_three_scoring_angles_and_correct_axis(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        result=fit_conditioned(xyz,uv,valid,k,e,variant="all_states")
        fit=result["fits"]["revolute"]
        self.assertEqual(len(fit["params"]),9)
        self.assertLess(axis_error(fit["params"][:3],[0,0,1]),.01)
        self.assertLess(axis_distance(fit["params"][3:6],fit["params"][:3],pivot),1e-4)

    def test_refinement_recovers_known_axis_without_gt_inputs(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        seed=dict(params=np.r_[.02,.01,1.,pivot+[.03,.01,.0],.18,.33,.48])
        fit=refine_revolute(seed,xyz,uv[2:],valid[2:],k,e)
        self.assertLess(axis_error(fit["params"][:3],[0,0,1]),.01)
        self.assertLess(axis_distance(fit["params"][3:6],fit["params"][:3],pivot),1e-4)
        self.assertLess(fit["rmse_px"],1e-4)

    def test_filtered_candidate_is_scored_on_every_original_observation(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        result=fit_conditioned(xyz,uv,valid,k,e)
        altered=uv[2:].copy();altered[:,0]+=100
        result=rescore(result,xyz,altered,valid[2:],k,e)
        self.assertEqual(result["fits"]["revolute"]["n"],24)
        self.assertGreater(result["fits"]["revolute"]["rmse_px"],40)

    def test_depth_surface_backprojection_uses_world_camera_transform(self):
        depth=np.ones((4,4))*3
        labels=np.ones((4,4),int)
        k=np.array([[5.,0.,1.5],[0.,5.,1.5],[0.,0.,1.]])
        e=np.eye(4);e[:3,3]=[.2,.3,.4]
        cloud=surface_points(depth,labels,1,k,e)
        expected=np.array([[x,y] for y in range(4) for x in range(4)])
        np.testing.assert_allclose(project(cloud,k,e),expected,atol=1e-12)

    def test_incorrect_shape_directions_do_not_override_clean_trajectory(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        seed=dict(params=np.r_[0.,0.,1.,pivot,.2,.35,.5],median_px=0.)
        free=refine_revolute(seed,xyz,uv[2:],valid[2:],k,e)
        with patch("testing_joint_fitting.selection.shape_axes",return_value=[("wrong",np.array([1.,0.,0.]))]):
            result,info=select_shape_fit(free,free,xyz,uv[2:],valid[2:],k,e,None,{})
        self.assertEqual(info["selected"],"free")
        self.assertLess(result["rmse_px"],1e-4)

    def test_repeated_single_point_cannot_claim_well_observed_axis(self):
        _,pivot,k,e,_,_=self.scene()
        xyz=np.tile([1.,0.,4.],(8,1))
        angles=np.array([.2,.35,.5])
        uv=np.array([project((xyz-pivot)@cv2.Rodrigues(np.array([0.,0.,a]))[0].T+pivot,k,e) for a in angles])
        seed=dict(params=np.r_[0.,0.,1.,pivot,angles])
        fit=refine_revolute(seed,xyz,uv,np.ones(uv.shape[:2],bool),k,e)
        self.assertGreater(fit["refinement"]["axis_local_uncertainty_deg"],89.)

    def test_candidate_transforms_reproject_known_motion_in_world_frame(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        result=estimate_motion(xyz,uv,valid,k,e,initial_depth=np.ones((4,4)),
            initial_segmentation=np.ones((4,4),int),pair_labels=[0,1])
        self.assertEqual(result["kind"],"revolute")
        for transform,target in zip(result["transforms_world"],uv[2:]):
            rotation=transform[:3,:3]
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
            np.testing.assert_allclose(project(xyz@rotation.T+transform[:3,3],k,e),target,atol=1e-3)

    def test_global_pose_initialization_converts_nonidentity_camera_to_world(self):
        xyz,pivot,k,e,uv,valid=self.scene()
        e[:3,:3]=cv2.Rodrigues(np.array([.2,-.1,.3]))[0]
        e[:3,3]=[.2,-.1,.5]
        uv=np.array([project((xyz-pivot)@cv2.Rodrigues(np.array([0.,0.,a]))[0].T+pivot,k,e)
                     for a in (.2,.35,.5)])
        fit=global_joint_seed(xyz,uv,valid[2:],k,e)
        self.assertIsNotNone(fit)
        self.assertLess(axis_error(fit["params"][:3],[0,0,1]),.01)
        self.assertLess(axis_distance(fit["params"][3:6],fit["params"][:3],pivot),1e-4)


if __name__ == "__main__":
    unittest.main()
