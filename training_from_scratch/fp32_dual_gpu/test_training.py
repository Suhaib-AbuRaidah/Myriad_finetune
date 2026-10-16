"""Numerical, migration, and resume checks; CUDA checks use both real GPUs."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from finetuning_pair.tests.test_pair_pipeline import make_scene
from training_from_scratch.tests.test_training import tiny_myriad
from training_from_scratch.dataset import ArticulatedDataset, collate_articulated
from training_from_scratch.features import FeatureCache
from training_from_scratch.model import ScratchModel
from training_from_scratch.precision import MasterAdamW
from training_from_scratch.train import make_scheduler
from training_from_scratch.checkpoint_io import make_encoder_reference, save_training_checkpoint, inspect_checkpoint
from .model import CachedEncoderSpec, CachedFP32Model, training_loss
from .optimizer import DeviceAdamW
from .checkpoint_io import resume_checkpoint, check_resume_config
from . import train

ROOT = Path(__file__).resolve().parents[1]


def tiny_cached(*, image_size=32):
    bare = tiny_myriad()
    bare.image_embedder = CachedEncoderSpec(image_size=image_size, embed_dim=24)
    return bare


class OptimizerTests(unittest.TestCase):
    def compare_adam(self, model_device, optimizer_device):
        torch.manual_seed(7)
        a = nn.Sequential(nn.Linear(3, 7), nn.SiLU(), nn.Linear(7, 2)).to(model_device)
        b = copy.deepcopy(a)
        unused_a = nn.Parameter(torch.ones(5, device=model_device))
        unused_b = nn.Parameter(unused_a.detach().clone())
        pa, pb = list(a.parameters()) + [unused_a], list(b.parameters()) + [unused_b]
        ordinary = torch.optim.AdamW(pa, lr=.003, betas=(.9, .99), weight_decay=.01, foreach=False)
        remote = DeviceAdamW(pb, device=optimizer_device, lr=.003, betas=(.9, .99), weight_decay=.01)
        for step in range(6):
            ordinary.zero_grad(set_to_none=True)
            remote.zero_grad(set_to_none=True)
            for micro in range(3):
                x = torch.arange(12, device=model_device).float().reshape(4, 3) / (step + micro + 1)
                a(x).square().mean().backward()
                b(x).square().mean().backward()
            remote.capture_model_gradients()
            self.assertTrue(all(p.grad is None for p in pb))
            norm_a = nn.utils.clip_grad_norm_(pa, .2, foreach=False)
            norm_b = nn.utils.clip_grad_norm_(remote.optimizer_parameters(), .2, foreach=False)
            torch.testing.assert_close(norm_a.cpu(), norm_b.cpu(), rtol=1e-6, atol=1e-7)
            ordinary.step()
            remote.step()
            for left, right in zip(pa, pb):
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            for state in remote.state.values():
                self.assertEqual(state["exp_avg"].device, torch.device(optimizer_device))
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertTrue(torch.equal(unused_b, torch.ones_like(unused_b)))
        self.assertNotIn("master_parameters", remote.state_dict())

    def test_cpu_matches_ordinary_adam(self):
        self.compare_adam("cpu", "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_cuda1_model_cuda0_optimizer_matches_ordinary_adam(self):
        self.compare_adam("cuda:1", "cuda:0")


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        make_scene(self.root, "train")
        make_scene(self.root, "val")
        bare = tiny_myriad().eval()
        self.source = self.root / "dino.pt"
        torch.save({"model": bare.state_dict()}, self.source)
        self.reference = make_encoder_reference(self.source)
        self.data = ArticulatedDataset(self.root, split="val", image_size=32)
        self.cache = self.root / "features"
        cache = FeatureCache(self.cache, identity=self.reference["identity"], image_size=32,
                             dtype=torch.float32, feature_dtype=torch.float16)
        with torch.no_grad():
            for sample in self.data:
                image = sample["x"].unsqueeze(0)
                embedding = bare.embed_image(image)
                embedding["feat"] = embedding["feat"].half()
                cache.write(image, embedding)

    def tearDown(self):
        self.temporary.cleanup()

    def model(self, device="cpu", checkpointing=False):
        model = CachedFP32Model(tiny_cached(), gradient_checkpointing=checkpointing).place(torch.device(device))
        model.frozen_encoder_reference = self.reference
        model.configure_cache(self.cache, identity=self.reference["identity"])
        return model

    def test_full_loss_is_fp32_and_cache_is_read_only(self):
        model = self.model()
        batch = collate_articulated([self.data[0]])
        original_paths = set(self.cache.iterdir())
        dtypes = []
        hooks = [m.register_forward_hook(lambda m, inputs, out: dtypes.append(out.dtype))
                 for m in model.modules() if isinstance(m, nn.Linear)]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = training_loss(model, batch)
        loss.backward()
        for hook in hooks:
            hook.remove()
        self.assertEqual(set(dtypes), {torch.float32})
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertEqual(len(list(model.image_embedder.parameters())), 0)
        self.assertEqual(set(self.cache.iterdir()), original_paths)
        with self.assertRaises(FileNotFoundError):
            model.embed_image(torch.ones_like(batch["x"]))
        self.assertEqual(set(self.cache.iterdir()), original_paths)

    def test_activation_checkpointing_preserves_loss_and_gradients(self):
        ordinary, checkpointed = self.model(), self.model(checkpointing=True)
        checkpointed.load_state_dict(ordinary.state_dict())
        batch = collate_articulated([self.data[0]])
        for model in (ordinary, checkpointed):
            torch.manual_seed(123)
            training_loss(model, batch).backward()
        for a, b in zip(ordinary.parameters(), checkpointed.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_real_myriad_two_gpu_updates_and_checkpoint_resume(self):
        ordinary, remote = self.model("cuda:1"), self.model("cuda:1")
        remote.load_state_dict(ordinary.state_dict())
        reference_opt = torch.optim.AdamW(ordinary.trainable_parameters(), lr=.001,
                                         weight_decay=.01, betas=(.9, .99), foreach=False)
        optimizer = DeviceAdamW(remote.trainable_parameters(), device="cuda:0", lr=.001,
                                weight_decay=.01, betas=(.9, .99))
        reference_schedule = make_scheduler(reference_opt, 10, 0)
        schedule = make_scheduler(optimizer, 10, 0)
        batches = [collate_articulated([self.data[i]]) for i in (0, 1)]
        denominator = torch.tensor(sum(int(b["flow_loss_mask"].sum()) for b in batches), device="cuda:1")
        for step in range(3):
            for model, opt, sched in ((ordinary, reference_opt, reference_schedule), (remote, optimizer, schedule)):
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(100 + step)
                for batch in batches:
                    batch = {k: v.to("cuda:1") if torch.is_tensor(v) else v for k, v in batch.items()}
                    training_loss(model, batch, loss_denominator=denominator).backward()
                if isinstance(opt, DeviceAdamW):
                    opt.capture_model_gradients()
                    parameters = opt.optimizer_parameters()
                else:
                    parameters = list(model.trainable_parameters())
                nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
                opt.step()
                sched.step()
            for a, b in zip(ordinary.parameters(), remote.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        from finetuning_pair.geometry import TARGET_DEFINITION
        config = dict(target_definition=TARGET_DEFINITION, precision="fp32", freeze_dino=True,
                      dino_identity=self.reference["identity"])
        path = self.root / "cuda_latest.pt"
        save_training_checkpoint(path, remote, optimizer, schedule, step=3, train_config=config,
                                 data_state=dict(epoch=6, offset=0, dataset_size=1))
        expected_random = torch.rand(7, device="cuda:1")
        restored = self.model("cuda:1")
        restored_opt = DeviceAdamW(restored.trainable_parameters(), device="cuda:0", lr=.001,
                                   weight_decay=.01, betas=(.9, .99))
        restored_schedule = make_scheduler(restored_opt, 10, 0)
        self.assertEqual(resume_checkpoint(path, restored, restored_opt, restored_schedule), 3)
        torch.testing.assert_close(torch.rand(7, device="cuda:1"), expected_random, rtol=0, atol=0)
        for a, b in zip(remote.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(optimizer.state.values(), restored_opt.state.values()):
            self.assertEqual(b["exp_avg"].device, torch.device("cuda:0"))
            torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
        self.assertEqual(schedule.state_dict(), restored_schedule.state_dict())

    def test_legacy_fp16_migration_preserves_master_weights_and_adam(self):
        old = ScratchModel(tiny_myriad(), freeze_dino=True).place(torch.device("cpu"), precision="fp16")
        old.frozen_encoder_reference = self.reference
        opt = MasterAdamW(old.trainable_parameters(), lr=.001, betas=(.9, .99))
        scheduler = make_scheduler(opt, 10, 0)
        for _ in range(2):
            for parameter in old.trainable_parameters():
                parameter.grad = torch.full_like(parameter, .123)
            opt.capture_model_gradients()
            opt.step()
            scheduler.step()
        path = self.root / "old.pt"
        from finetuning_pair.geometry import TARGET_DEFINITION
        config = dict(target_definition=TARGET_DEFINITION, freeze_dino=True, precision="fp16",
                      dino_identity=self.reference["identity"])
        save_training_checkpoint(path, old, opt, scheduler, step=2, train_config=config,
                                 data_state=dict(epoch=1, offset=0, dataset_size=1))
        source_before = path.read_bytes()
        model = self.model()
        remote = DeviceAdamW(model.trainable_parameters(), device="cpu", lr=.001, betas=(.9, .99))
        schedule = make_scheduler(remote, 10, 0)
        self.assertEqual(resume_checkpoint(path, model, remote, schedule), 2)
        differs_from_half = False
        for actual, exact, half in zip(model.trainable_parameters(), opt.optimizer_parameters(), old.trainable_parameters()):
            torch.testing.assert_close(actual, exact, rtol=0, atol=0)
            differs_from_half |= not torch.equal(actual, half.float())
        self.assertTrue(differs_from_half)
        for left, right in zip(opt.state.values(), remote.state.values()):
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), schedule.state_dict())
        self.assertEqual(source_before, path.read_bytes())
        converted = self.root / "converted.pt"
        config["precision"] = "fp32"
        save_training_checkpoint(converted, model, remote, schedule, step=2, train_config=config)
        expected_random = torch.rand(5)
        new_model = self.model()
        new_opt = DeviceAdamW(new_model.trainable_parameters(), device="cpu", lr=.001, betas=(.9, .99))
        new_sched = make_scheduler(new_opt, 10, 0)
        resume_checkpoint(converted, new_model, new_opt, new_sched)
        torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
        for a, b in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(remote.state.values(), new_opt.state.values()):
            torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
        self.assertNotIn("master_parameters", torch.load(converted, weights_only=True)["optimizer"])

    def argv(self, output):
        return ["--device", "cpu", "--optimizer-device", "cpu", "--data-root", str(self.root),
                "--output-dir", str(output), "--dino-checkpoint", str(self.source),
                "--dino-feature-cache", str(self.cache), "--cache-dir", str(self.root / "targets"),
                "--image-size", "32", "--max-steps", "4", "--warmup-steps", "0",
                "--gradient-accumulation-steps", "2", "--val-num-scenes", "0", "--test-num-scenes", "0"]

    def test_cpu_end_to_end_resume_matches_uninterrupted_training(self):
        full, split = self.root / "full", self.root / "split"
        with patch.object(train, "build_cached_myriad", side_effect=tiny_cached), contextlib.redirect_stdout(io.StringIO()):
            train.main(self.argv(full))
            train.main(self.argv(split) + ["--run-steps", "2"])
            train.main(self.argv(split) + ["--resume", str(split / "latest.pt")])
        a = torch.load(full / "latest.pt", weights_only=True)
        b = torch.load(split / "latest.pt", weights_only=True)
        self.assertEqual(a["step"], 4)
        self.assertEqual(a["data_state"], b["data_state"])
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertTrue(torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"]))
        for left, right in zip(a["rng"]["torch_cuda"], b["rng"]["torch_cuda"]):
            self.assertTrue(torch.equal(left, right))
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        for key, value in a["optimizer"]["state"].items():
            for name in value:
                torch.testing.assert_close(value[name], b["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_rejects_objective_changes_and_original_output_directory(self):
        with self.assertRaisesRegex(ValueError, "max_steps"):
            check_resume_config(dict(max_steps=200), dict(max_steps=300))
        args, _ = train.parse_args(self.argv(self.root))
        args.resume = self.root / "latest.pt"
        with self.assertRaisesRegex(ValueError, "separate output directory"):
            train.run(args, dict(train_config={}))


if __name__ == "__main__":
    unittest.main()
