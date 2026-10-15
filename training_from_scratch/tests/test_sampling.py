"""Real autoregressive rollouts must not require compiled FlexAttention."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from training_from_scratch.model import ScratchModel
from training_from_scratch.sampling import dense_sampler_mask, sampler_attention_context
from training_from_scratch.testing import sample_rollout
from training_from_scratch.tests.test_training import tiny_myriad


class SamplerAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dense_mask_preserves_causal_rules_and_empty_prefix_rows(self):
        from myriad.model import inference_query_causal_mask_mod
        mask = dense_sampler_mask(inference_query_causal_mask_mod(2, 2), 1, 1, 7, 7, device="cpu")
        expected = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 1],
        ], dtype=torch.bool)[None, None]
        torch.testing.assert_close(mask, expected)
        q, k, v = (torch.randn(1, 2, 7, 8) for _ in range(3))
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        torch.testing.assert_close(attention[:, :, :2], torch.zeros_like(attention[:, :, :2]), rtol=0, atol=0)
        for row in range(2, 7):
            allowed = expected[0, 0, row]
            logits = q[:, :, row:row + 1] @ k[:, :, allowed].transpose(-1, -2)
            reference = logits.softmax(-1) @ v[:, :, allowed]
            torch.testing.assert_close(attention[:, :, row:row + 1], reference)

    def test_nested_context_restores_mask_constructor_after_error(self):
        import myriad.model as original
        before = original.create_block_mask
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with sampler_attention_context():
                self.assertIs(original.create_block_mask, dense_sampler_mask)
                with sampler_attention_context():
                    self.assertIs(original.create_block_mask, dense_sampler_mask)
                self.assertIs(original.create_block_mask, dense_sampler_mask)
                raise RuntimeError("sentinel")
        self.assertIs(original.create_block_mask, before)

    def exercise_rollouts(self, device, precision):
        model = ScratchModel(tiny_myriad(), freeze_dino=True).place(device, precision=precision)
        image = torch.zeros(3, 32, 32)
        known = torch.tensor([[[.3, .4], [.6, .7]], [[.32, .4], [.62, .7]]])
        times = torch.tensor([0., 1., 2.])
        for observed in (1, 2):
            before = torch.random.get_rng_state().clone()
            with patch("myriad.model.flex_attention_compiled", side_effect=AssertionError("FlexAttention invoked")):
                predicted = sample_rollout(model, image, known[:observed], times,
                    ensemble_size=2, diffusion_steps=2, seed=19, amp_dtype="none")
                repeated = sample_rollout(model, image, known[:observed], times,
                    ensemble_size=2, diffusion_steps=2, seed=19, amp_dtype="none")
            self.assertEqual(predicted.shape, (2, 3, 2, 2))
            self.assertTrue(torch.isfinite(predicted).all())
            torch.testing.assert_close(predicted[:, :observed], known[:observed].to(device).expand(2, -1, -1, -1))
            torch.testing.assert_close(predicted, repeated, rtol=0, atol=0)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_cpu_real_sampler_without_compiled_attention(self):
        self.exercise_rollouts(torch.device("cpu"), "fp32")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_fp16_real_sampler_without_compiled_attention(self):
        self.exercise_rollouts(torch.device("cuda:0"), "fp16")


if __name__ == "__main__":
    unittest.main()
