"""Analytic controls for units, axis physics and shift-handling objectives."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from finetuning.joint_parame_estim.run import project,axis_error
from .common import FitInput,perturb
from .algorithms import quality_weights
from .flow import rotation_jacobian
from .refinement import profile_offset,fit_variant
from .candidate import estimate_revolute_motion


def synthetic():
    xyz=np.array([[.8,-.1,3.8],[1.,.1,4.],[.9,.2,4.1],[1.1,-.2,3.9],
                  [.7,.15,4.2],[1.2,.1,3.7],[.85,-.15,4.15],[1.05,.25,3.85]])
    pivot=np.array([0.,0.,4.]);axis=np.array([.2,-.3,1.]);axis/=np.linalg.norm(axis)
    k=np.array([[500.,0.,256.],[0.,500.,256.],[0.,0.,1.]])
    e=np.eye(4);e[:3,:3]=cv2.Rodrigues(np.array([.1,-.1,.2]))[0];e[:3,3]=[.1,.2,.3]
    uv=np.array([project((xyz-pivot)@cv2.Rodrigues(axis*a)[0].T+pivot,k,e) for a in (0.,.1,.2,.3,.4)])
    contact=dict(center=pivot,covariance=.01*np.eye(3),whitening=10*np.eye(3))
    return FitInput(xyz,uv,np.ones(uv.shape[:2],bool),k,e,contact,4/3),axis,pivot


class AxisMotionTests(unittest.TestCase):
    def test_rotational_flow_jacobian_matches_world_camera_projection(self):
        data,_,pivot=synthetic();analytic=rotation_jacobian(data.xyz,pivot,data.k,data.e)
        numerical=np.empty_like(analytic);h=1e-6
        for i in range(3):
            r=cv2.Rodrigues(np.eye(3)[i]*h)[0]
            positive=project((data.xyz-pivot)@r.T+pivot,data.k,data.e)
            r=cv2.Rodrigues(-np.eye(3)[i]*h)[0]
            negative=project((data.xyz-pivot)@r.T+pivot,data.k,data.e)
            numerical[:,:,i]=(positive-negative)/(2*h)
        np.testing.assert_allclose(analytic,numerical,rtol=1e-6,atol=1e-6)

    def test_profile_offset_has_exact_regularized_quadratic_solution(self):
        error=np.tile([3.,-2.],(4,8,1));valid=np.ones((4,8),bool);weights=np.ones((4,8))
        residual,prior,offset=profile_offset(error,valid,weights,2.)
        np.testing.assert_allclose(offset,np.tile([2.,-4/3],(8,1)))
        expected=((error-offset)**2).sum()+2*(offset**2).sum()
        self.assertAlmostEqual(float((residual**2).sum()+(prior**2).sum()),expected)

    def test_shared_offset_and_jitter_are_not_interchangeable(self):
        error=np.tile(np.array([1.,-1.,1.,-1.])[:,None,None],(1,8,2))
        residual,prior,offset=profile_offset(error,np.ones((4,8),bool),np.ones((4,8)),1.)
        np.testing.assert_allclose(offset,0.)
        np.testing.assert_allclose(residual,error)

    def test_perturbation_uses_model_pixels_and_preserves_observed_state(self):
        data,_,_=synthetic();altered=perturb(data,"common_x",1.,42)
        np.testing.assert_array_equal(altered.uv[0],data.uv[0])
        np.testing.assert_allclose(altered.uv[1:]-data.uv[1:],np.tile([4/3,0.],(4,8,1)))
        np.testing.assert_allclose(data.uv,synthetic()[0].uv)

    def test_initial_reanchoring_alone_is_identical_to_absolute_residual(self):
        data,_,_=synthetic();observed=perturb(data,"common_x",1.,42).uv
        absolute=data.uv[1:]-observed[1:]
        relative=(data.uv[1:]-data.uv[0])-(observed[1:]-observed[0])
        np.testing.assert_allclose(relative,absolute,atol=1e-12)

    def test_bounded_length_weights_preserve_total_weight(self):
        data,_,_=synthetic()
        for variant in ("length_soft","length_square","extent_square"):
            weights=quality_weights(data,variant)
            self.assertTrue(np.all(weights>0))
            self.assertAlmostEqual(float(weights.sum()),8.)

    def test_clean_motion_axis_survives_shift_aware_objective(self):
        data,axis,_=synthetic()
        for variant in ("full_absolute","bias_track_2px","bias_shared","frame_bias"):
            fit=fit_variant(data,variant)
            self.assertIsNotNone(fit)
            self.assertLess(axis_error(fit["params"][:3],axis),.05)

    def test_axis_sign_does_not_measure_rotation_direction(self):
        _,axis,_=synthetic()
        self.assertEqual(axis_error(axis,-axis),0.)
        positive=cv2.Rodrigues(axis*.2)[0];negative=cv2.Rodrigues(axis*(-.2))[0]
        error=float(np.linalg.norm(cv2.Rodrigues(positive@negative.T)[0]))
        self.assertAlmostEqual(error,.4,places=8)

    def test_ensemble_covariance_does_not_change_under_shared_perturbation(self):
        data,_,_=synthetic()
        ensemble=np.stack([data.uv+np.arange(5)[:,None,None]*value for value in (-.2,-.1,0.,.1,.2)])
        data=replace(data,uv_hypotheses=ensemble)
        altered=perturb(data,"per_track",2.,42)
        np.testing.assert_allclose(np.var(altered.uv_hypotheses,axis=0),np.var(ensemble,axis=0),atol=1e-10)
        np.testing.assert_array_equal(altered.uv_hypotheses[:,0],ensemble[:,0])

    def test_covariance_weighting_and_geometry_weighting_preserve_clean_axis(self):
        data,axis,_=synthetic();data=replace(data,uv_hypotheses=data.uv[None])
        for variant in ("frame_covariance","frame_covariance_consistent","frame_length_geometry"):
            fit=fit_variant(data,variant)
            self.assertLess(axis_error(fit["params"][:3],axis),.05)

    def test_frozen_candidate_returns_correct_world_transforms(self):
        data,axis,_=synthetic()
        result=estimate_revolute_motion(data.xyz,data.uv,data.valid,data.k,data.e,
            initial_depth=np.ones((512,512))*3,initial_segmentation=np.ones((512,512),int),pair_labels=[0,1])
        self.assertLess(axis_error(result["axis_world"],axis),.05)
        self.assertEqual(result["states"],[1,2,3,4])
        for transform,target in zip(result["transforms_world"],data.uv[1:]):
            rotation=transform[:3,:3]
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
            pixels=project(data.xyz@rotation.T+transform[:3,3],data.k,data.e)
            np.testing.assert_allclose(pixels,target,atol=1e-3)


if __name__=="__main__":unittest.main()
