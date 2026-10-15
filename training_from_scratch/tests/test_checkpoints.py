"""Storage, compatibility and exact-restoration checks for compact checkpoints."""
import errno
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from finetuning_pair.geometry import TARGET_DEFINITION
from training_from_scratch import train
from training_from_scratch.checkpoint_io import (
    inspect_checkpoint, load_model_weights, make_encoder_reference,
    resume_training_checkpoint, save_training_checkpoint,
)
from training_from_scratch.features import load_frozen_encoder
from training_from_scratch.model import ScratchModel
from training_from_scratch.precision import MasterAdamW
from training_from_scratch.tests.test_training import tiny_myriad

ROOT = Path(__file__).resolve().parents[1]


class CompactCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.source = self.root / "dino.pt"
        encoder = tiny_myriad().image_embedder
        torch.save({"model": {"image_embedder." + key: value
                              for key, value in encoder.state_dict().items()}}, self.source)
        self.reference = make_encoder_reference(self.source)
        self.config = {"target_definition": TARGET_DEFINITION, "freeze_dino": True,
                       "dino_identity": self.reference["identity"], "dino_precision": "fp32"}
        self.model = self.new_model()
        load_frozen_encoder(self.model.image_embedder, self.source)
        self.model.frozen_encoder_reference = self.reference
        self.optimizer = MasterAdamW(self.model.trainable_parameters(), lr=.001)
        self.scheduler = train.make_scheduler(self.optimizer, 4, 0)
        for parameter in self.model.trainable_parameters():
            parameter.grad = torch.full_like(parameter, .01)
        self.optimizer.capture_model_gradients()
        self.optimizer.step()
        self.scheduler.step()

    def tearDown(self):
        self.temporary.cleanup()

    def new_model(self):
        return ScratchModel(tiny_myriad(), freeze_dino=True).place(torch.device("cpu"), precision="fp16")

    def save(self, name="latest.pt", **kwargs):
        path = self.root / name
        save_training_checkpoint(path, self.model, self.optimizer, self.scheduler, step=1,
            train_config=self.config, metrics={"minADE_1": .1}, data_state={"epoch": 1, "offset": 2}, **kwargs)
        return path

    def assert_models_equal(self, other):
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)

    def test_compact_training_and_inference_restore_all_model_tensors(self):
        latest = self.save()
        expected_random = torch.rand(3)
        best = self.save("best.pt", weights_only=True)
        training = torch.load(latest, weights_only=True)
        inference = torch.load(best, weights_only=True)
        self.assertEqual(training["kind"], "training")
        self.assertTrue(training["frozen_encoder"])
        self.assertFalse(any(key.startswith("model.image_embedder.") for key in training["model"]))
        self.assertLess(best.stat().st_size, latest.stat().st_size)
        for key in ("optimizer", "scheduler", "rng", "scaler", "data_state"):
            self.assertIsNone(inference[key])
        restored = self.new_model()
        optimizer = MasterAdamW(restored.trainable_parameters(), lr=.001)
        scheduler = train.make_scheduler(optimizer, 4, 0)
        self.assertEqual(resume_training_checkpoint(latest, restored, optimizer, scheduler), 1)
        self.assert_models_equal(restored)
        torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), self.scheduler.state_dict())
        for a, b in zip(self.optimizer.optimizer_parameters(), optimizer.optimizer_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(self.optimizer.state[a][key], optimizer.state[b][key], rtol=0, atol=0)
        predicted = self.new_model()
        self.assertEqual(load_model_weights(best, predicted), 1)
        self.assert_models_equal(predicted)
        with self.assertRaisesRegex(ValueError, "inference weights only"):
            resume_training_checkpoint(best, restored, optimizer, scheduler)

    def test_relocated_encoder_override_and_missing_or_changed_archive(self):
        latest = self.save()
        moved = self.root / "moved.pt"
        shutil.copyfile(self.source, moved)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            load_model_weights(latest, self.new_model())
        restored = self.new_model()
        load_model_weights(latest, restored, dino_checkpoint=moved)
        self.assert_models_equal(restored)
        changed = self.root / "changed.pt"
        torch.save({"model": {"different": torch.zeros(1)}}, changed)
        with self.assertRaisesRegex(ValueError, "differs from"):
            load_model_weights(latest, self.new_model(), dino_checkpoint=changed)

    def test_legacy_full_checkpoint_remains_independent_of_encoder_file(self):
        path = self.save(checkpoint_storage="full")
        payload = torch.load(path, weights_only=True)
        payload["version"] = 1
        del payload["kind"], payload["frozen_encoder"]
        torch.save(payload, path)
        self.source.unlink()
        self.assertEqual(inspect_checkpoint(path)["kind"], "training")
        restored = self.new_model()
        optimizer = MasterAdamW(restored.trainable_parameters(), lr=.001)
        scheduler = train.make_scheduler(optimizer, 4, 0)
        resume_training_checkpoint(path, restored, optimizer, scheduler)
        self.assert_models_equal(restored)

    def test_atomic_save_failure_preserves_previous_checkpoint(self):
        latest = self.save()
        before = latest.read_bytes()

        def out_of_space(payload, handle):
            handle.write(b"incomplete checkpoint")
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch("finetuning_pair.checkpoint_io.torch.save", side_effect=out_of_space):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual(before, latest.read_bytes())
        self.assertFalse(list(self.root.glob(".latest.pt.*.tmp")))


if __name__ == "__main__":
    unittest.main()
