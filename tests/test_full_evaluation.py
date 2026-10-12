import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from finetuning.evaluate import aggregate_records, common_horizon_comparison, pixel_errors, scene_metrics, select_cases
from finetuning.validation import ValidationConfig, evaluate_validation


class TinyDataset(Dataset):
    training = False

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"x": torch.zeros(3, 8, 8), "trajectories": torch.arange(3)[:, None, None].float().expand(3, 2, 2) * .1,
                "position_valid": torch.ones(3, 2, dtype=torch.bool),
                "visibility": torch.ones(3, 2, dtype=torch.bool), "point_valid": torch.ones(2, dtype=torch.bool),
                "state_times": torch.arange(3).float(), "camera_static": torch.tensor(True)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def embed_image(self, image):
        return {"L": 1, "feat": torch.zeros(1, 1)}

    def predict_simulate(self, *, n_traj, ts, given_pos, **kwargs):
        result = ts[:, :, None, None].expand(-1, -1, n_traj, 2) * .1
        result = result + torch.randn_like(result) * .01
        known = given_pos.reshape(ts.shape[0], -1, n_traj, 2)
        result[:, :known.shape[1]] = known
        return result


class FullEvaluationTests(unittest.TestCase):
    def test_common_horizon_matches_point_ids_and_intersects_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = {}
            for mode, indices, error, slot, known in (
                ("with_poke", [1, 2], .2, 0, 2),
                ("without_poke", [0, 1], .4, 1, 1),
            ):
                truth = np.zeros((4, 2, 2))
                predicted = np.full((5, 4, 2, 2), 100.)
                predicted[:, 2, slot] = [error, 0]
                valid = np.ones((4, 2), dtype=bool)
                if mode == "without_poke":
                    valid[3, slot] = False
                path = root / f"{mode}.npz"
                np.savez(path, point_indices=np.array(indices), known_states=known,
                         ground_truth=truth, predictions=predicted, valid=valid)
                records[mode] = [{"index": 0, "sample_id": "example", "object_id": "object",
                                  "skipped": False, "results_file": path.name}]
            rows = common_horizon_comparison(root, records)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["common_points"], 1)
            self.assertEqual(rows[0]["scored_point_states"], 1)
            self.assertAlmostEqual(rows[0]["with_poke_minADE_1"], .16)
            self.assertAlmostEqual(rows[0]["without_poke_minADE_1"], .64)

    def test_callback_preserves_aggregate_rng_and_resumed_scene_seeds(self):
        for poke in (True, False):
            config = ValidationConfig(num_scenes=3, ensemble_size=5, diffusion_steps=2, seed=19,
                                      initial_poke=poke, amp_dtype="none")
            model, dataset = TinyModel(), TinyDataset()
            before = torch.random.get_rng_state().clone()
            plain = evaluate_validation(model, dataset, config)
            results = {}
            def save(index, sample, predicted, errors):
                results[index] = errors.clone()
            recorded = evaluate_validation(model, dataset, config, scene_callback=save)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertTrue(model.training)
            for key in ("minADE_1", "minADE_5", "scenes_evaluated", "points_conditioned"):
                self.assertEqual(plain[key], recorded[key])
            subset = {}
            evaluate_validation(model, dataset, config, scene_indices=[1],
                                scene_callback=lambda i, s, p, e: subset.update({i: e.clone()}))
            self.assertTrue(torch.equal(results[1], subset[1]))

    def test_ranking_uses_first_prediction_not_best_of_five(self):
        rows = [{"index": 0, "skipped": False, "minADE_1": .3, "minADE_5": .001},
                {"index": 1, "skipped": False, "minADE_1": .1, "minADE_5": .09},
                {"index": 2, "skipped": False, "minADE_1": .2, "minADE_5": .01},
                {"index": 3, "skipped": True}]
        selected = select_cases(rows, 1)
        self.assertEqual(selected["top_10"][0]["index"], 1)
        self.assertEqual(selected["failures"][0]["index"], 0)

    def test_pixel_metrics_and_validity_mask(self):
        truth = np.zeros((2, 2, 2), dtype=np.float32)
        prediction = truth.copy()
        prediction[..., 0] = 1 / 7
        prediction[0, 1] = 100  # Invalid target must not affect the score.
        mask = np.array([[True, False], [True, True]])
        metrics = pixel_errors(prediction, truth, mask, 8)
        self.assertAlmostEqual(metrics["ADE_pixels"], 1, places=6)
        self.assertAlmostEqual(metrics["FDE_pixels"], 1, places=6)
        self.assertAlmostEqual(metrics["RMSE_pixels"], 1, places=6)
        mask[-1] = False
        self.assertIsNone(pixel_errors(prediction, truth, mask, 8)["FDE_pixels"])

    def test_scene_metrics_and_equal_scene_aggregation(self):
        truth = np.zeros((3, 2, 2), dtype=np.float32)
        truth[1:, :, 0] = .1
        predictions = np.broadcast_to(truth, (5, *truth.shape)).copy()
        predictions[0, 1:, :, 0] += .2
        arrays = {"known_states": np.array(1), "image": np.zeros((3, 11, 11)),
                  "ground_truth": truth, "valid": np.ones((3, 2), dtype=bool),
                  "predictions": predictions, "link_nodes": np.array([0, 1])}
        metrics = scene_metrics(arrays, np.array([.16, 0, 0, 0, 0]))
        self.assertAlmostEqual(metrics["ADE_1_pixels"], 2, places=6)
        self.assertAlmostEqual(metrics["stationary_minADE"], .04, places=6)
        self.assertFalse(metrics["improves_stationary"])
        row = {**metrics, "skipped": False, "evaluation_seconds": 1}
        summary = aggregate_records([row, {**row, "minADE_1": .32}])
        self.assertAlmostEqual(summary["minADE_1"], .24)
        self.assertEqual(summary["scenes_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
