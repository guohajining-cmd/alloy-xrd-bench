import pandas as pd
import pytest

from scripts.summarize_sro_matrix import compare_to_ridge


def test_paired_comparison_and_normalization():
    baseline = pd.Series([1., 2., 3.], index=["a", "b", "c"])
    result = compare_to_ridge(baseline * .5, baseline, iterations=100)
    assert result["relative_improvement_percent"] == 50.
    assert result["S_SRO"] == 50.
    assert result["bootstrap_improvement_fraction"] == 1.
    assert result["relative_improvement_ci95"] == [50., 50.]


def test_worse_model_has_negative_relative_improvement_but_zero_score():
    baseline = pd.Series([1., 2.], index=["a", "b"])
    result = compare_to_ridge(baseline * 2, baseline, iterations=100)
    assert result["relative_improvement_percent"] == -100.
    assert result["S_SRO"] == 0.


def test_parent_mismatch_is_not_silently_dropped():
    with pytest.raises(ValueError):
        compare_to_ridge(pd.Series([1.], index=["a"]), pd.Series([1.], index=["b"]))
