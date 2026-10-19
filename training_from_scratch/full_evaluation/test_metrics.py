"""Analytic checks for pixel/percentage metrics and calibrated joint fitting."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from .metrics import trajectory_metrics, summarize, kinematic_reference


class MetricTests(unittest.TestCase):
    def tracks(self):
        truth = np.zeros((5, 1, 2), dtype=np.float64)
        truth[:, 0, 0] = np.linspace(.2, .6, 5)
        truth[:, 0, 1] = .3
        pred = truth[None].copy()
        pred[:, 1:, :, 1] += .02
        return pred, truth, np.ones((5, 1), bool)

    def test_pixel_ade_and_path_percentage_have_explicit_units(self):
        pred, truth, valid = self.tracks()
        row, tracks = trajectory_metrics(pred, truth, valid, known_states=1, image_size=101,
                                         gt_type='revolute', source_side=202)
        self.assertAlmostEqual(row['ADE_1_pixels'], 2.)
        self.assertAlmostEqual(row['ADE_source_pixels'], 4.)
        self.assertAlmostEqual(row['gt_path_pixels'], 40.)
        self.assertAlmostEqual(row['ADE_percent_full_path'], 5.)
        self.assertAlmostEqual(row['minADE_1'], .0016)
        self.assertEqual(row['normalized_tracks'], 1)
        self.assertAlmostEqual(tracks[0]['ADE_percent_path'], 5.)

    def test_with_poke_keeps_full_path_and_separates_forecast_denominator(self):
        pred, truth, valid = self.tracks()
        row, _ = trajectory_metrics(pred, truth, valid, known_states=2, image_size=101, gt_type='revolute')
        self.assertAlmostEqual(row['ADE_percent_full_path'], 5.)
        self.assertAlmostEqual(row['ADE_percent_forecast_path'], 100*2/30)

    def test_fixed_short_and_incomplete_tracks_are_not_divided_by_epsilon(self):
        pred, truth, valid = self.tracks()
        for kind in ('fixed', 'revolute'):
            still = np.broadcast_to(truth[:1], truth.shape).copy()
            row, _ = trajectory_metrics(pred, still, valid, known_states=1, image_size=101, gt_type=kind)
            self.assertIsNone(row['ADE_percent_full_path'])
            self.assertEqual(row['normalized_tracks'], 0)
            self.assertGreater(row['ADE_1_pixels'], 0)
        valid[-1] = False
        row, _ = trajectory_metrics(pred, truth, valid, known_states=1, image_size=101, gt_type='revolute')
        self.assertIsNone(row['ADE_percent_full_path'])
        self.assertEqual(row['incomplete_tracks'], 1)
        self.assertAlmostEqual(row['ADE_1_pixels'], 2.)

    def test_absolute_scores_keep_all_points_and_oracle_does_not_replace_first(self):
        pred, truth, valid = self.tracks()
        row, _ = trajectory_metrics(np.concatenate([pred, truth[None]], 0), truth, valid,
                                     known_states=1, image_size=101, gt_type='revolute')
        self.assertAlmostEqual(row['ADE_1_pixels'], 2.)
        self.assertEqual(row['minADE_5'], 0.)
        self.assertEqual(row['oracle_5_ADE_pixels'], 0.)

    def test_summary_includes_failed_type_fits_and_weights_pairs_equally(self):
        rows = [dict(scene_index=0, gt_type='revolute', fit_available=True, predicted_type='revolute',
                     type_correct=True, improves_stationary=True, ADE_1_pixels=2.),
                dict(scene_index=1, gt_type='revolute', fit_available=False,
                     improves_stationary=False, ADE_1_pixels=10.)]
        result = summarize(rows, source_count=3)
        self.assertEqual(result['type_accuracy'], .5)
        self.assertEqual(result['type_accuracy_given_fit'], 1.)
        self.assertEqual(result['metrics']['ADE_1_pixels']['mean'], 6.)
        self.assertEqual(result['recall']['revolute'], .5)
        self.assertEqual(result['source_coverage'], 2/3)

    def test_transform_reference_recovers_axis_line_and_exposes_bad_stored_pivot(self):
        pivot = np.array([.4, .2, 0.])
        matrices = []
        for angle in np.linspace(0, .6, 5):
            r = cv2.Rodrigues(np.array([0., 0., angle]))[0]
            child = np.eye(4); child[:3, :3] = r; child[:3, 3] = pivot-r@pivot
            matrices.append([np.eye(4), child])
        source = dict(link_transforms_world=np.array(matrices),
                      joint_origins_world=np.tile(pivot+[.2,0,0], (5,1,1)),
                      joint_axes_world=np.tile([0.,0.,1.], (5,1,1)))
        ref = kinematic_reference(source, 0, 0, 1)
        self.assertTrue(ref['observable'])
        self.assertAlmostEqual(abs(np.dot(ref['axis'], [0,0,1])), 1.)
        self.assertLess(ref['transform_residual'], 1e-9)
        self.assertAlmostEqual(ref['stored_pivot_axis_offset'], .2)

    def test_ground_truth_revolute_fit_recovers_geometry(self):
        from finetuning_pair.analyze_joints import fit_joint
        from finetuning.joint_parame_estim.run import axis_error, axis_distance, project
        xyz = np.array([[.8,-.1,3.8],[1.,.1,4.],[.9,.2,4.1],[1.1,-.2,3.9],
                        [.7,.15,4.2],[1.2,.1,3.7],[.85,-.15,4.15],[1.05,.25,3.85]])
        pivot = np.array([0.,0.,4.])
        k = np.array([[500.,0,256],[0,500.,256],[0,0,1.]])
        tracks=[]
        for angle in (.2,.35,.5):
            r=cv2.Rodrigues(np.array([0.,0.,angle]))[0]
            tracks.append(project((xyz-pivot)@r.T+pivot,k,np.eye(4)))
        fitted=fit_joint(xyz,np.array(tracks),np.ones((3,8),bool),k,np.eye(4))
        self.assertEqual(fitted['kind'],'revolute')
        params=fitted['fits']['revolute']['params']
        self.assertLess(axis_error(params[:3],[0,0,1]),1.)
        self.assertLess(axis_distance(params[3:6],params[:3],pivot),.02)

    def test_matched_modes_compare_identical_point_slots(self):
        from .report import matched_comparisons
        pred,truth,valid=self.tracks()
        truth=np.repeat(truth,2,axis=1)
        a=truth[None].copy(); a[:,1:,0,1]+=.02; a[:,1:,1,1]+=.1
        b=truth[:,1:][None].copy(); b[:,2:,:,1]+=.02
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            root=Path(directory); (root/'predictions').mkdir()
            np.savez(root/'predictions/0000_without_poke.npz',predictions=a,ground_truth=truth,
                     valid=np.ones((5,2),bool),point_indices=np.array([0,1]))
            np.savez(root/'predictions/0000_with_poke.npz',predictions=b,ground_truth=truth[:,1:],
                     valid=np.ones((5,1),bool),point_indices=np.array([1]))
            rows=[dict(pair_index=0,scene_index=0,mode=mode,gt_type='revolute') for mode in ('without_poke','with_poke')]
            manifest=dict(modes=['without_poke','with_poke'],train_config=dict(image_size=101),min_path_pixels=1.)
            result=matched_comparisons(manifest,rows,root)
            self.assertAlmostEqual(result['common_ADE_pixels']['without']['mean'],10.)
            self.assertAlmostEqual(result['common_ADE_pixels']['with_poke']['mean'],2.)

    def test_normalized_stationary_baseline_uses_same_track_denominator(self):
        from .report import normalized_stationary_comparison
        pred,truth,valid=self.tracks()
        metrics,tracks=trajectory_metrics(pred,truth,valid,known_states=1,image_size=101,gt_type='revolute')
        row=dict(pair_index=0,joint_index=0,mode='without_poke',gt_type='revolute',track_metrics=tracks)
        manifest=dict(train_config=dict(image_size=101),min_path_pixels=1.)
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            root=Path(directory);(root/'predictions').mkdir()
            np.savez(root/'predictions/0000_without_poke.npz',ground_truth=truth,valid=valid,known_states=np.array(1))
            result=normalized_stationary_comparison(root,manifest,[row],[dict(mode='without_poke',min_path_pixels=1.)])
            self.assertAlmostEqual(result[0]['model']['mean'],5.)
            self.assertAlmostEqual(result[0]['stationary']['mean'],62.5)

    def test_cached_checkpoint_loader_preserves_fp32_weights_without_dino(self):
        from training_from_scratch.fp32_dual_gpu.test_training import tiny_cached
        from training_from_scratch.fp32_dual_gpu.model import CachedFP32Model
        from training_from_scratch.checkpoint_io import make_encoder_reference, save_training_checkpoint
        from finetuning_pair.geometry import TARGET_DEFINITION
        from .run import load_cached_weights
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory)
            source=root/'encoder.pt'; torch.save({'model':{'unused':torch.ones(1)}},source)
            reference=make_encoder_reference(source)
            model=CachedFP32Model(tiny_cached()).place(torch.device('cpu'))
            model.frozen_encoder_reference=reference
            config=dict(target_definition=TARGET_DEFINITION,precision='fp32',freeze_dino=True,
                        image_size=32,dino_identity=reference['identity'])
            checkpoint=root/'best.pt'
            save_training_checkpoint(checkpoint,model,None,None,step=8,train_config=config,weights_only=True)
            with patch('training_from_scratch.full_evaluation.run.build_cached_myriad',side_effect=tiny_cached):
                restored=load_cached_weights(checkpoint,torch.device('cpu'),root/'features')
            self.assertEqual(len(list(restored.image_embedder.parameters())),0)
            self.assertFalse(restored.training)
            for a,b in zip(model.parameters(),restored.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertEqual(b.dtype,torch.float32)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
