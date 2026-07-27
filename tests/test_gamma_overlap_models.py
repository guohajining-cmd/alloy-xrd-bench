from __future__ import annotations

import unittest

import torch


class GammaOverlapModelTests(unittest.TestCase):
    def test_training_seed_disables_nondeterministic_attention_kernels(self):
        from alloy_xrd_bench.gamma_overlap_training import _seed_everything

        _seed_everything(2027)

        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_resnet_and_transformer_set_models_have_task_shapes(self):
        from alloy_xrd_bench.gamma_overlap_models import GammaSetPredictor

        for encoder in ("resnet1d", "spectrum_transformer"):
            model = GammaSetPredictor(
                encoder_name=encoder,
                class_count=7,
                query_count=3,
                point_count=151,
                overlap_mask=True,
                embedding_dim=32,
                input_channels=2,
            )
            output = model(torch.rand(2, 2, 151))
            self.assertEqual((2, 3), tuple(output["existence_logits"].shape))
            self.assertEqual((2, 3, 7), tuple(output["class_logits"].shape))
            self.assertEqual((2, 3, 5), tuple(output["parameters"].shape))
            self.assertEqual((2, 151), tuple(output["overlap_mask_logits"].shape))

    def test_hungarian_set_loss_is_finite_and_backpropagates(self):
        from alloy_xrd_bench.gamma_overlap_models import GammaSetPredictor, hungarian_set_loss

        torch.manual_seed(2027)
        model = GammaSetPredictor(
            encoder_name="resnet1d",
            class_count=3,
            query_count=3,
            point_count=151,
            overlap_mask=False,
            embedding_dim=32,
        )
        output = model(torch.rand(2, 1, 151))
        targets = [
            {"classes": torch.tensor([0, 2]), "parameters": torch.rand(2, 5)},
            {"classes": torch.tensor([1]), "parameters": torch.rand(1, 5)},
        ]
        losses = hungarian_set_loss(output, targets)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
