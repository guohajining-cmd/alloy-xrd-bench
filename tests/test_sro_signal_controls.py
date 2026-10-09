import numpy as np
import pytest

from scripts.run_sro_signal_controls import feature_sets, fixed_parameters


def test_matched_capacity_changes_only_information():
    c, x = np.array([[.2, .8], [.3, .7]]), np.array([[1., 2., 3.], [4., 5., 6.]])
    features = feature_sets(c, x)
    np.testing.assert_array_equal(features["CX"][:, :2], features["C"])
    np.testing.assert_array_equal(features["CX"][:, 2:], features["X"])
    budget = dict(n_estimators=300, max_depth=6, learning_rate=.05)
    assert fixed_parameters(budget, 2027, 4) == dict(**budget, tree_method="hist", device="cpu", random_state=2027, n_jobs=4)


@pytest.mark.parametrize("x", [np.ones((3, 2)), np.array([[np.nan], [0.]])])
def test_invalid_feature_alignment_or_values_rejected(x):
    with pytest.raises(ValueError):
        feature_sets(np.ones((2, 2)), x)


def test_invalid_threads_rejected():
    with pytest.raises(ValueError):
        fixed_parameters({}, 2027, 0)
