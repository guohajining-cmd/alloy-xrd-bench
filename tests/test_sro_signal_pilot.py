import numpy as np
import pytest

from scripts.run_sro_signal_pilot import masked_parent_mae, paired_bootstrap, fit_grouped_ridge


def test_parent_macro_does_not_overweight_more_labels():
    truth = np.zeros((2, 3))
    pred = np.array([[1, 100, 100], [3, 3, 3]])
    mask = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    np.testing.assert_allclose(masked_parent_mae(truth, pred, mask), [1, 3])


def test_paired_bootstrap_uses_same_parents():
    report = paired_bootstrap(np.array([1., 2., 3.]), np.array([.5, 1.5, 2.5]), 42)
    assert report["delta_mae"] == pytest.approx(-.5)
    assert report["ci95"] == pytest.approx([-.5, -.5])


def test_missing_training_labels_do_not_enter_ridge():
    x = np.arange(8., dtype=float)[:, None]
    y = np.concatenate([2 * x, 3 * x], axis=1)
    mask = np.ones_like(y, dtype=bool)
    mask[4:, 1] = False
    corrupted = y.copy()
    corrupted[~mask] = 1e10
    a = fit_grouped_ridge(x, y, mask, x, 1.)
    b = fit_grouped_ridge(x, corrupted, mask, x, 1.)
    np.testing.assert_allclose(a, b)
