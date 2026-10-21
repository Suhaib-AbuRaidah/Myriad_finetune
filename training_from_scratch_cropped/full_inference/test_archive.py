"""Verify metric/coordinate conventions before producing the full archive."""
import unittest
import numpy as np
from training_from_scratch_cropped.geometry import SquareCrop
from .run import trajectory_scores


class ArchiveTests(unittest.TestCase):
    def test_source_pixel_metrics_do_not_depend_on_crop_magnification(self):
        truth = np.array([[[100.+10*t, 100.], [150.+5*t, 120.]] for t in range(5)])
        prediction = truth.copy(); prediction[1:] += [3., 4.]
        valid = np.ones((5, 2), bool)
        values = []
        for crop in (SquareCrop(0, 0, 512), SquareCrop(80, 80, 128)):
            cropped = crop.normalize(truth)
            crop_valid = valid & ((cropped>=0)&(cropped<=1)).all(-1)
            row = trajectory_scores(prediction[None], truth, valid, crop_valid, crop, [512,512], "revolute")
            self.assertAlmostEqual(row["primary"]["pixel_ADE"], 5.)
            self.assertAlmostEqual(row["primary"]["source_RMSE_pixels"], 5.)
            values.append(row["primary"]["crop_ADE_pixels"])
        self.assertAlmostEqual(values[1]/values[0], 4.)

    def test_outside_crop_targets_remain_scored(self):
        truth = np.array([[[100.+100*t, 100.]] for t in range(5)])
        prediction = truth.copy(); prediction[1:] += [0., 5.]
        valid = np.ones((5,1), bool); crop = SquareCrop(50,50,100)
        cropped = crop.normalize(truth)
        crop_valid = valid & ((cropped>=0)&(cropped<=1)).all(-1)
        row = trajectory_scores(prediction[None], truth, valid, crop_valid, crop, [512,512], "revolute")
        self.assertAlmostEqual(row["primary"]["pixel_ADE"], 5.)
        self.assertEqual(row["primary"]["future_outside_crop"], 4)

    def test_unscoreable_forecast_and_fixed_path_percentage(self):
        truth = np.zeros((5,1,2)); valid = np.zeros((5,1), bool); valid[0] = True
        crop = SquareCrop(0,0,512)
        row = trajectory_scores(truth[None], truth, valid, valid, crop, [512,512], "revolute")
        self.assertFalse(row["scoreable"])
        valid[:] = True
        row = trajectory_scores(truth[None], truth, valid, valid, crop, [512,512], "fixed")
        self.assertIsNone(row["primary"]["GT_path_normalized_error_percent"])


if __name__=="__main__": unittest.main()
