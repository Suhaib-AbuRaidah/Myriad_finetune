"""CPU regression tests using the real MYRIAD blocks and a tiny DINO encoder."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from finetuning_pair.dataset import PairDataset as OriginalDataset
from finetuning_pair.tests.test_pair_pipeline import make_scene, SpyModel
from finetuning_pair.parallel import ParallelContext
from training_from_scratch import train
from training_from_scratch.checkpoint_io import inspect_checkpoint, load_model_weights
from training_from_scratch.dataset import PairDataset, collate_pairs
from training_from_scratch.model import ScratchModel
from training_from_scratch.precision import MasterAdamW
from training_from_scratch.features import FeatureCache
from training_from_scratch.testing import run_test_subset, TestConfig as RolloutConfig

ROOT = Path(__file__).resolve().parents[1]


def tiny_myriad():
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
    from myriad.dinov3 import DinoFeatureExtractor
    from myriad.model import MyriadStepByStep

    config = DINOv3ViTConfig(image_size=32, patch_size=16, hidden_size=24,
                           intermediate_size=48, num_hidden_layers=1,
                           num_attention_heads=2, num_register_tokens=4)
    with patch.object(AutoModel, "from_pretrained", side_effect=lambda *a, **kw: DINOv3ViTModel(config)):
        encoder = DinoFeatureExtractor(image_size=32, model_version="dinov3_vits16", requires_grad=True)
    encoder.embed_dim = 24
    with patch("myriad.model.DinoFeatureExtractor", return_value=encoder):
        return MyriadStepByStep(
            width=48, depth=2,
            transformer_params=dict(d_head=24, out_mlp_depth=1, emb_depth=1,
                                    ff_expand=2, ada_norm_size=48, track_id_embedding=True),
            distribution_head_params=dict(depth=1, d_cond=48, value_scale_cascade=True,
                                          value_scale_cascade_steps=8, steps=2),
        )


class ScratchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        make_scene(self.root, "train")
        make_scene(self.root, "val")

    def tearDown(self):
        self.temporary.cleanup()

    def batch(self):
        data = PairDataset(self.root, split="val", image_size=32, boundary_margin=0)
        return collate_pairs([data[0], data[1]])

    def test_dataset_matches_pair_pipeline_and_isolates_cache(self):
        original = OriginalDataset(self.root, split="val", image_size=32)
        copied = PairDataset(self.root, split="val", image_size=32, cache_dir=self.root / "cache")
        self.assertEqual(len(original), len(copied))
        for i in range(len(original)):
            a, b = original[i], copied[i]
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if torch.is_tensor(a[key]):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertTrue(list((self.root / "cache").glob("*.pt")))

    def test_all_components_update_and_checkpointing_preserves_gradients(self):
        torch.manual_seed(123)
        regular = ScratchModel(tiny_myriad())
        checkpointed = ScratchModel(tiny_myriad(), gradient_checkpointing=True)
        checkpointed.load_state_dict(regular.state_dict())
        batch = self.batch()
        first = {name: p.detach().clone() for name, p in regular.named_parameters()}
        optimizers = [torch.optim.AdamW(m.trainable_parameters(), lr=.001, weight_decay=0)
                      for m in (regular, checkpointed)]
        for step in range(4):
            for model, optimizer in zip((regular, checkpointed), optimizers):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(100 + step)
                loss = train.training_loss(model, batch, amp_dtype="none")
                loss.backward()
                optimizer.step()
            for a, b in zip(regular.parameters(), checkpointed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for prefix in ("model.image_embedder.", "model.transformer.mid_level.", "model.distribution_head."):
            changed = [name for name, p in regular.named_parameters()
                       if name.startswith(prefix) and not torch.equal(first[name], p)]
            self.assertTrue(changed, prefix)
        for name, parameter in regular.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(first[name], parameter, rtol=0, atol=0)
        regular.eval()
        regular.train()
        self.assertTrue(all(module.training for module in regular.image_embedder.modules()))

    def test_accumulated_loss_uses_valid_tokens_across_microbatches(self):
        data = PairDataset(self.root, split="val", image_size=32, boundary_margin=0)
        samples = [data[0], data[1]]  # Different numbers of valid point slots.
        model, combined = SpyModel(), SpyModel()
        total = sum(int(s["flow_loss_mask"].sum()) for s in samples)
        for sample in samples:
            train.training_loss(model, collate_pairs([sample]), amp_dtype="none",
                                loss_denominator=torch.tensor(total)).backward()
        train.training_loss(combined, collate_pairs(samples), amp_dtype="none").backward()
        torch.testing.assert_close(model.weight.grad, combined.weight.grad)

    def test_masked_targets_never_enter_flow_head(self):
        batch = self.batch()
        valid = batch["flow_loss_mask"]
        self.assertTrue(bool((~valid).any()))
        batch["flow_query"][~valid] = float("nan")
        model = SpyModel()
        loss = train.training_loss(model, batch, amp_dtype="none")
        expected = (batch["flow_query"][valid] - model.weight).square().mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.weight.grad)))

    def test_masked_motion_inputs_cannot_contaminate_valid_loss_or_gradients(self):
        batch = self.batch()
        invalid = ~batch["motion_token_mask"]
        fields = ("flow_poke", "flow_query", "pos_poke", "pos_query",
                  "pos_orig_poke", "pos_orig_query", "t_poke", "t_query")
        clean = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        poisoned = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        for key in fields:
            clean[key][invalid] = 0
            poisoned[key][invalid] = float("nan")
        model = ScratchModel(tiny_myriad())
        torch.nn.init.normal_(model.distribution_head.head.emb_mapping.weight, std=.001)
        torch.nn.init.normal_(model.distribution_head.head.out_proj.weight, std=.02)
        torch.manual_seed(91)
        expected = train.training_loss(model, clean, amp_dtype="none")
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        torch.manual_seed(91)
        actual = train.training_loss(model, poisoned, amp_dtype="none")
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if name in gradients:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    def test_empty_local_shard_has_zero_differentiable_loss(self):
        batch = self.batch()
        batch["flow_loss_mask"].zero_()
        batch["motion_token_mask"].zero_()
        model = ScratchModel(tiny_myriad())
        loss = train.training_loss(model, batch, amp_dtype="none", loss_denominator=torch.tensor(5))
        self.assertEqual(float(loss), 0)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) and not bool(g.any()) for g in gradients))

    def test_nonfinite_valid_target_is_identified_as_bad_input(self):
        batch = self.batch()
        batch["flow_query"][batch["flow_loss_mask"]] = float("nan")
        with self.assertRaises(train.NonfiniteFlowLoss) as caught:
            train.training_loss(SpyModel(), batch, amp_dtype="none")
        self.assertEqual(caught.exception.details["source"], "inputs")
        self.assertIn("flow_query", caught.exception.details["nonfinite_inputs"])
        self.assertEqual(caught.exception.details["sample_ids"], batch["sample_id"])

    def run_training(self, folder, extra=()):
        args = train.parse_args([
            "--data-root", str(self.root), "--output-dir", str(folder),
            "--cache-dir", str(self.root / "cache"), "--device", "cpu",
            "--max-steps", "4", "--warmup-steps", "0", "--lr", ".001",
            "--gradient-accumulation-steps", "2", "--gradient-checkpointing",
            "--image-size", "32", "--val-num-scenes", "0", "--test-num-scenes", "0",
            "--save-every-steps", "2", *extra,
        ])
        with patch.object(train, "build_myriad", side_effect=tiny_myriad), contextlib.redirect_stdout(io.StringIO()):
            train._run_training(args, train._resolve_device(args.device), ParallelContext())

    def test_full_checkpoint_exact_resume_with_accumulation(self):
        full, staged = self.root / "full", self.root / "staged"
        self.run_training(full)
        self.run_training(staged, ["--run-steps", "2"])
        checkpoint = staged / "latest.pt"
        info = inspect_checkpoint(checkpoint)
        self.assertEqual(info["step"], 2)
        self.assertNotIn("base_checkpoint", info)
        self.run_training(staged, ["--resume", str(checkpoint)])
        a = torch.load(full / "latest.pt", weights_only=True)
        b = torch.load(checkpoint, weights_only=True)
        self.assertEqual(a["step"], b["step"])
        self.assertEqual(a["data_state"], b["data_state"])
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
        restored = ScratchModel(tiny_myriad())
        self.assertEqual(load_model_weights(checkpoint, restored), 4)
        self.assertEqual(set(b["model"]), set(restored.state_dict()))
        self.assertTrue(any(n.startswith("model.image_embedder.") for n in b["model"]))
        self.assertFalse(any("lora_" in n for n in b["model"]))
        with self.assertRaises(FileExistsError):
            self.run_training(full)

    def test_periodic_exports_stay_in_new_folder(self):
        data = PairDataset(self.root, split="val", image_size=32, boundary_margin=0)
        result = run_test_subset(SpyModel(), data, RolloutConfig(num_scenes=1, diffusion_steps=2,
                                poke_mode="without", amp_dtype="none"), self.root / "exports", step=2)
        self.assertEqual(result["step"], 2)
        self.assertTrue(list((self.root / "exports").rglob("*.gif")))

    def test_best_weights_do_not_replace_resumable_latest(self):
        folder = self.root / "best_selection"
        validation = ["--val-num-scenes", "1", "--val-every-steps", "2"]
        with patch.object(train, "evaluate_validation", side_effect=[{"minADE_1": .2}, {"minADE_1": .1}]):
            self.run_training(folder, validation + ["--run-steps", "2"])
            self.run_training(folder, validation + ["--resume", str(folder / "latest.pt")])
        latest = torch.load(folder / "latest.pt", weights_only=True)
        best = torch.load(folder / "best.pt", weights_only=True)
        self.assertEqual(latest["step"], 4)
        self.assertEqual(best["step"], 4)
        self.assertEqual(latest["kind"], "training")
        self.assertIsNotNone(latest["optimizer"])
        self.assertEqual(best["kind"], "weights")
        self.assertIsNone(best["optimizer"])
        self.assertEqual(best["metrics"]["best_val_minADE_1"], .1)
        with self.assertRaisesRegex(ValueError, "inference weights only"):
            self.run_training(folder, validation + ["--resume", str(folder / "best.pt")])

    def test_master_adam_retains_small_updates_and_full_precision_on_resume(self):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
        optimizer = MasterAdamW([parameter], lr=1e-4, weight_decay=0)
        for _ in range(10):
            parameter.grad = torch.ones_like(parameter)
            optimizer.capture_model_gradients()
            optimizer.step()
        self.assertLess(float(parameter[0]), 1.0)
        self.assertTrue(all(p.dtype == torch.float32 for p in optimizer.optimizer_parameters()))
        saved = optimizer.state_dict()
        replacement = torch.nn.Parameter(torch.zeros_like(parameter))
        resumed = MasterAdamW([replacement], lr=1e-4, weight_decay=0)
        resumed.load_state_dict(saved)
        torch.testing.assert_close(parameter, replacement, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.optimizer_parameters()[0], resumed.optimizer_parameters()[0], rtol=0, atol=0)

    def test_frozen_encoder_cache_hits_and_pair_image_identity(self):
        model = ScratchModel(tiny_myriad(), freeze_dino=True)
        model.place(torch.device("cpu"))
        model.feature_cache = FeatureCache(self.root / "features", identity="encoder-A", image_size=32, dtype=torch.float32)
        image = self.batch()["x"]
        a = model.embed_image(image)
        with patch.object(model.model, "embed_image", side_effect=AssertionError("cache miss")):
            b = model.embed_image(image)
        torch.testing.assert_close(a["feat"], b["feat"], rtol=0, atol=0)
        self.assertFalse(a["feat"].requires_grad)
        self.assertEqual(model.feature_cache.hits, 2)
        self.assertNotEqual(model.feature_cache.path(image[:1]), model.feature_cache.path(image[1:]))
        other = FeatureCache(self.root / "features", identity="encoder-B", image_size=32, dtype=torch.float32)
        self.assertNotEqual(model.feature_cache.path(image[:1]), other.path(image[:1]))
        model.train()
        self.assertFalse(any(m.training for m in model.image_embedder.modules()))
        self.assertFalse(any(p.requires_grad for p in model.image_embedder.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_residual_overflow_is_prevented_without_promoting_weights(self):
        model = ScratchModel(tiny_myriad(), freeze_dino=True).place(torch.device("cuda:0"), precision="fp16")
        batch = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                 for key, value in self.batch().items()}
        # Each finite block contribution fits FP16; their accumulated sum does not.
        handles = [block.out_proj.register_forward_hook(lambda module, inputs, output: output + 40000)
                   for block in model.transformer.mid_level]
        residuals = []
        handles.append(model.transformer.out_proj.in_norm.register_forward_pre_hook(
            lambda module, inputs: residuals.append((inputs[0].dtype, float(inputs[0].abs().max())))))
        try:
            loss = train.training_loss(model, batch, amp_dtype="fp16")
            loss.backward()
            self.assertEqual(residuals[-1][0], torch.float32)
            self.assertGreater(residuals[-1][1], torch.finfo(torch.float16).max)
            self.assertTrue(all(p.dtype == torch.float16 for p in model.trainable_parameters()))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            model.zero_grad(set_to_none=True)
            model._residual_precision_hook.remove()
            with self.assertRaises(train.NonfiniteFlowLoss) as caught:
                train.training_loss(model, batch, amp_dtype="fp16")
            self.assertEqual(caught.exception.details["source"], "transformer")
        finally:
            for handle in handles:
                handle.remove()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_gpu_updates_scaler_and_exact_resume(self):
        source = self.root / "tiny_dino.pt"
        encoder = tiny_myriad().image_embedder
        torch.save({"model": {"image_embedder." + key: value for key, value in encoder.state_dict().items()}}, source)
        extra = ["--device", "cuda:0", "--precision", "fp16", "--freeze-dino",
                 "--dino-feature-cache", str(self.root / "gpu_features"),
                 "--dino-checkpoint", str(source)]
        full, staged = self.root / "gpu_full", self.root / "gpu_staged"
        self.run_training(full, extra)
        self.run_training(staged, extra + ["--run-steps", "2"])
        self.run_training(staged, extra + ["--resume", str(staged / "latest.pt")])
        a = torch.load(full / "latest.pt", weights_only=True)
        b = torch.load(staged / "latest.pt", weights_only=True)
        self.assertEqual(a["scaler"], b["scaler"])
        self.assertTrue(a["frozen_encoder"])
        self.assertFalse(any(n.startswith("model.image_embedder.") for n in a["model"]))
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            if name.startswith("model.transformer."):
                self.assertEqual(a["model"][name].dtype, torch.float16)
        for x, y in zip(a["optimizer"]["master_parameters"], b["optimizer"]["master_parameters"]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_cpu_optimizer_offload_and_overflow_skip(self):
        parameter = torch.nn.Parameter(torch.ones(2, device="cuda:0", dtype=torch.float16))
        optimizer = MasterAdamW([parameter], device="cpu", lr=.001, weight_decay=0)
        scaler = torch.amp.GradScaler("cuda", init_scale=256)
        scaler.scale(parameter.float().square().mean()).backward()
        optimizer.capture_model_gradients()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        self.assertLess(float(parameter[0]), 1)
        self.assertTrue(all(p.device.type == "cpu" for p in optimizer.optimizer_parameters()))
        previous = parameter.detach().clone()
        optimizer.zero_grad()
        scaler.scale(parameter.float().sum() * float("inf")).backward()
        optimizer.capture_model_gradients()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
        self.assertEqual(scaler.get_scale(), 128)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_forward_overflow_discards_accumulated_gradients(self):
        source = self.root / "tiny_dino.pt"
        encoder = tiny_myriad().image_embedder
        torch.save({"model": {"image_embedder." + key: value for key, value in encoder.state_dict().items()}}, source)
        original = train.training_loss
        calls = 0
        initial = {}

        def overflow_once(model, batch, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                initial.update({name: p.detach().clone() for name, p in model.named_parameters()})
            if calls == 2:
                self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in model.parameters()))
                raise train.NonfiniteFlowLoss({"source": "flow_head", "sample_ids": batch["sample_id"]})
            if calls == 3:
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
            return original(model, batch, **kwargs)

        output = self.root / "forward_recovery"
        with patch.object(train, "training_loss", new=overflow_once):
            self.run_training(output, ["--device", "cuda:0", "--precision", "fp16", "--freeze-dino",
                                       "--dino-checkpoint", str(source)])
        saved = torch.load(output / "latest.pt", weights_only=True)
        self.assertEqual(calls, 10)  # Four completed two-microbatch updates, plus the discarded two.
        self.assertEqual(saved["step"], 4)
        self.assertEqual(saved["scaler"]["scale"], 256)
        self.assertEqual(saved["scaler"]["_growth_tracker"], 4)
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        skipped = [record for record in records if record["event"] == "nonfinite_forward"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["step"], 0)
        self.assertEqual(skipped[0]["errors"][0]["source"], "flow_head")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_repeated_fp16_forward_failure_stops_without_saving_bad_weights(self):
        source = self.root / "tiny_dino.pt"
        encoder = tiny_myriad().image_embedder
        torch.save({"model": {"image_embedder." + key: value for key, value in encoder.state_dict().items()}}, source)
        output = self.root / "persistent_forward_failure"

        def fail_forward(*args, **kwargs):
            raise train.NonfiniteFlowLoss({"source": "flow_head", "sample_ids": []})

        with patch.object(train, "training_loss", new=fail_forward):
            with self.assertRaisesRegex(FloatingPointError, "Repeated nonfinite FP16 forward"):
                self.run_training(output, ["--device", "cuda:0", "--precision", "fp16", "--freeze-dino",
                                           "--dino-checkpoint", str(source)])
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertFalse((output / "latest.pt").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_bad_inputs_features_and_parameters_stop_immediately(self):
        source = self.root / "tiny_dino.pt"
        encoder = tiny_myriad().image_embedder
        torch.save({"model": {"image_embedder." + key: value for key, value in encoder.state_dict().items()}}, source)
        for failure in ("inputs", "image_features", "parameters"):
            with self.subTest(failure=failure):
                output = self.root / failure

                def fail_forward(model, batch, **kwargs):
                    if failure == "parameters":
                        with torch.no_grad():
                            next(model.trainable_parameters()).fill_(float("nan"))
                    raise train.NonfiniteFlowLoss({
                        "source": "flow_head" if failure == "parameters" else failure,
                        "sample_ids": batch["sample_id"],
                    })

                with patch.object(train, "training_loss", new=fail_forward):
                    with self.assertRaisesRegex(FloatingPointError, "Nonfinite training inputs"):
                        self.run_training(output, ["--device", "cuda:0", "--precision", "fp16", "--freeze-dino",
                                                   "--dino-checkpoint", str(source)])
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertFalse((output / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
