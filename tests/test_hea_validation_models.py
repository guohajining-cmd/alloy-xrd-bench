import unittest
import os

import numpy as np
import torch

from alloy_xrd_bench.hea_validation_models import (
    PerTargetExtraTrees,
    PerTargetRidge,
    ResNet1D,
    PhysicsConstrainedTransformer,
    SiameseRankNet,
    SpectrumTransformer,
    masked_mse,
    normalize_reconstruction_target,
    peak_features,
    seed_everything,
)


class ClassicalModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2027)
        self.x = rng.normal(size=(12, 8))
        self.y = rng.normal(size=(12, 128))
        self.mask = rng.random(size=(12, 128)) > 0.2
        self.mask[:, -1] = False
        self.two_theta = np.linspace(10.0, 90.0, 4001)
        self.pattern = (
            np.exp(-0.5 * ((self.two_theta - 30.0) / 0.08) ** 2)
            + 0.5 * np.exp(-0.5 * ((self.two_theta - 55.0) / 0.12) ** 2)
        )

    def test_masked_ridge_emits_all_targets(self):
        model = PerTargetRidge(alpha=1.0).fit(self.x, self.y, self.mask)

        prediction = model.predict(self.x[:2])

        self.assertEqual((2, 128), prediction.shape)
        self.assertTrue(np.isfinite(prediction).all())

    def test_masked_extra_trees_emits_all_targets_deterministically(self):
        first = PerTargetExtraTrees(n_estimators=4, random_state=2027, n_jobs=1).fit(
            self.x, self.y, self.mask
        )
        second = PerTargetExtraTrees(n_estimators=4, random_state=2027, n_jobs=1).fit(
            self.x, self.y, self.mask
        )
        np.testing.assert_array_equal(first.predict(self.x[:2]), second.predict(self.x[:2]))

    def test_peak_features_have_fixed_shape(self):
        features = peak_features(self.pattern, self.two_theta, max_peaks=32)

        self.assertEqual((98,), features.shape)
        self.assertTrue(np.isfinite(features).all())

    def test_peak_features_reject_mismatched_grid(self):
        with self.assertRaisesRegex(ValueError, "same one-dimensional shape"):
            peak_features(self.pattern[:-1], self.two_theta, max_peaks=32)


class NeuralModelTests(unittest.TestCase):
    def test_resnet_and_transformer_emit_128_targets(self):
        batch = torch.zeros(2, 1, 4001)

        with torch.no_grad():
            resnet = ResNet1D(128)(batch)
            transformer = SpectrumTransformer(128)(batch)

        self.assertEqual((2, 128), tuple(resnet.shape))
        self.assertEqual((2, 128), tuple(transformer.shape))

    def test_physics_transformer_emits_sro_and_finite_forward_pattern(self):
        batch = torch.zeros(2, 1, 4001)

        with torch.no_grad():
            result = PhysicsConstrainedTransformer(128, peak_count=8)(batch)

        self.assertEqual((2, 128), tuple(result["sro"].shape))
        self.assertEqual((2, 4001), tuple(result["reconstruction"].shape))
        self.assertTrue(torch.isfinite(result["reconstruction"]).all())
        self.assertTrue((result["reconstruction"] >= 0).all())
        self.assertTrue((result["reconstruction"] <= 1).all())

    def test_reconstruction_target_uses_unit_scale_and_handles_zero_patterns(self):
        spectra = torch.tensor([[0.0, 50.0, 100.0], [0.0, 0.0, 0.0]])

        normalized = normalize_reconstruction_target(spectra)

        torch.testing.assert_close(
            normalized,
            torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]),
        )

    def test_ranknet_emits_one_score_per_spectrum(self):
        batch = torch.zeros(2, 1, 4001)

        with torch.no_grad():
            score = SiameseRankNet()(batch)

        self.assertEqual((2,), tuple(score.shape))

    def test_masked_mse_ignores_missing_targets(self):
        prediction = torch.tensor([[1.0, 999.0]])
        truth = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[True, False]])

        self.assertEqual(1.0, masked_mse(prediction, truth, mask).item())

    def test_seed_everything_makes_initialization_repeatable(self):
        seed_everything(2027)
        first = ResNet1D(2).head.weight.detach().clone()
        seed_everything(2027)
        second = ResNet1D(2).head.weight.detach().clone()

        self.assertTrue(torch.equal(first, second))

    def test_seed_everything_configures_deterministic_cublas(self):
        previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            seed_everything(2027)
            self.assertEqual(":4096:8", os.environ["CUBLAS_WORKSPACE_CONFIG"])
        finally:
            if previous is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous

    def test_seed_everything_disables_nondeterministic_attention_kernels(self):
        seed_everything(2027)

        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


if __name__ == "__main__":
    unittest.main()
