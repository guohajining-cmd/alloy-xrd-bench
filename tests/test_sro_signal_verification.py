import numpy as np
import pandas as pd
import pytest

from scripts.verify_sro_signal_increment import score_prediction, bootstrap_contrast, matched_arrays, reproduce_ridge, require_spectral_matrix


def test_score_averages_views_within_parent_before_global_mean():
    expected = pd.DataFrame(dict(xrd_sample_id=["a", "b", "c"], parent_group_id=["p", "p", "q"]))
    prediction = expected.assign(sro_predictions=[[0., 8.], [4., 0.], [8., 100.]])
    y, mask = np.zeros((3, 2)), np.array([[1, 1], [1, 1], [1, 0]], bool)
    errors, score = score_prediction(expected, prediction.iloc[::-1], y, mask)
    assert errors.to_dict() == {"p": 3., "q": 8.}
    assert score["mae"] == 5.5 and score["valid_target_count"] == 5
    with pytest.raises(ValueError, match="membership"):
        score_prediction(expected, prediction.iloc[:2], y, mask)


def test_crossed_interval_detects_shared_training_seed_variability():
    reference = pd.DataFrame({2027: np.ones(100)})
    candidate = pd.DataFrame({2027: np.full(100, .2), 2028: np.full(100, 1.8)})
    result = bootstrap_contrast(candidate, reference, repeats=1000)
    np.testing.assert_allclose(result["delta_ci95_parent"], [0., 0.], atol=1e-15)
    np.testing.assert_allclose(result["delta_ci95_seed_parent"], [-.8, .8], atol=1e-15)


def test_constant_paired_improvement_and_parent_order():
    reference = pd.DataFrame({2027: [1., 2., 3.]}, index=["c", "a", "b"])
    candidate = (reference-.2).iloc[::-1]
    result = bootstrap_contrast(candidate, reference, repeats=50)
    np.testing.assert_allclose(result["delta_ci95_parent"], [-.2, -.2])
    assert result == bootstrap_contrast(candidate, reference, repeats=50)


def test_unpaired_or_nonfinite_errors_rejected():
    c = pd.DataFrame({2027: [.2, .3]}, index=["a", "b"])
    with pytest.raises(ValueError, match="membership"):
        matched_arrays(c, c.rename(index={"a": "x"}))
    with pytest.raises(ValueError, match="identities"):
        matched_arrays(c, c.rename(columns={2027: 2028}))
    with pytest.raises(ValueError, match="Invalid"):
        matched_arrays(c, c*np.nan)


def test_ridge_refit_obeys_target_masks_and_zero_fallback():
    from sklearn.linear_model import Ridge
    x = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([[1., 99.], [2., 88.], [100., 77.], [4., 66.]])
    mask = np.array([[1, 0], [1, 0], [0, 0], [1, 0]], bool)
    result = reproduce_ridge(x, y, mask, x, 1.)
    expected = Ridge(alpha=1.).fit(x[[0, 1, 3]], y[[0, 1, 3], 0]).predict(x)
    np.testing.assert_allclose(result[:, 0], expected)
    np.testing.assert_array_equal(result[:, 1], np.zeros(4))


def test_full_matrix_gate_cannot_mark_partial_predictions_complete():
    config=dict(split_columns=["id", "composition", "chemistry", "sro"], seeds=list(range(2027,2032)))
    cases={(split,model,seed) for split in config["split_columns"]
           for model in ["peak-xgboost","resnet1d","transformer","physics-transformer"] for seed in config["seeds"]}
    assert len(cases)==80
    require_spectral_matrix(cases,config)
    with pytest.raises(ValueError,match="Incomplete"):
        require_spectral_matrix(list(cases)[:-1],config)
