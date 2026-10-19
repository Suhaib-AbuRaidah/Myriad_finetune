import unittest
import numpy as np
import torch

from finetuning_pair.conditioning import normalized_pixels
from testing_crop_inference.geometry import SquareCrop, mask_crop, crop_sample, restore_predictions


class CropGeometryTests(unittest.TestCase):
    def test_roundtrip_including_outside_points(self):
        points = np.array([[-20., 40], [0, 0], [130, 220], [600, 720]])
        original, crop = SquareCrop(0, 0, 512), SquareCrop(101, 32, 257)
        np.testing.assert_allclose(crop.unnormalize(crop.normalize(points)), points, atol=1e-12)
        np.testing.assert_allclose(restore_predictions(crop.normalize(points), crop, original),
                                   original.normalize(points), atol=1e-12)
        np.testing.assert_allclose(original.normalize(points), normalized_pixels(points, 512, 512, 384))

    def test_camera_intrinsics_project_to_same_cropped_pixels(self):
        crop = SquareCrop(87, 12, 251)
        k = np.array([[410., 0, 255.5], [0, 410, 255.5], [0, 0, 1.]])
        xyz = np.array([[.1, -.2, 2], [-.2, .1, 1.]])
        projected = xyz @ k.T
        projected = projected[:, :2]/projected[:, 2:]
        new = xyz @ (crop.matrix @ k).T
        np.testing.assert_allclose(new[:, :2]/new[:, 2:], crop.normalize(projected)*383, atol=1e-12)

    def test_union_contains_both_masks_at_image_edge(self):
        a, b = np.zeros((512, 512), bool), np.zeros((512, 512), bool)
        a[0:30, 0:25], b[60:90, 100:170] = True, True
        crop = mask_crop(a, b)
        y, x = np.nonzero(a | b)
        uv = crop.normalize(np.column_stack([x, y]))
        self.assertTrue(((uv >= 0) & (uv <= 1)).all())
        self.assertLess(crop.side, 512)
        self.assertEqual(crop.left, 0)

    def test_full_frame_crop_and_trajectory_scale(self):
        a = np.ones((512, 512), bool)
        self.assertEqual(mask_crop(a, ~a), SquareCrop(0, 0, 512))
        full, crop = SquareCrop(0, 0, 512), SquareCrop(50, 50, 256)
        points = np.array([[110., 110.], [140., 125.]])
        self.assertAlmostEqual(np.linalg.norm(np.diff(crop.normalize(points), axis=0)),
                               2*np.linalg.norm(np.diff(full.normalize(points), axis=0)))

    def test_sample_keeps_future_outside_crop_for_scoring(self):
        seg = np.full((512, 512), -2)
        seg[80:100, 80:100], seg[100:120, 100:120] = 0, 1
        source = dict(initial_depth=np.ones((512, 512)), initial_segmentation=seg,
                      initial_rgb=np.full((512, 512, 3), 125, np.uint8), camera_intrinsic=np.eye(3))
        full, crop = SquareCrop(0, 0, 512), SquareCrop(70, 70, 70)
        p = torch.tensor(full.normalize(np.array([[[110., 110.]], [[300., 300.]]])), dtype=torch.float32)
        sample = dict(trajectories=p, actual_trajectories=p, pair_labels=torch.tensor([0, 1]),
                      overlay_alpha=.85, position_valid=torch.ones((2, 1), dtype=torch.bool),
                      visibility=torch.ones((2, 1), dtype=torch.bool), supervise_occluded=True,
                      state_times=torch.tensor([0., 1.]), reference_pixels=torch.tensor([[110., 110.]]))
        result = crop_sample(sample, source, crop)
        self.assertTrue(result["position_valid"].all())
        self.assertFalse(result["crop_position_valid"][1, 0])
        self.assertGreater(result["trajectories"][1, 0, 0], 1)
        self.assertTrue(result["flow_loss_mask"].all())
        np.testing.assert_allclose(restore_predictions(result["trajectories"].numpy(), crop, full), p, atol=1e-7)
        self.assertEqual(result["x"].shape, (3, 384, 384))
        self.assertTrue(result["child_mask"].any())


if __name__ == "__main__":
    unittest.main()
