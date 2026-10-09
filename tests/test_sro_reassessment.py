import numpy as np
import pandas as pd
import pytest

from alloy_xrd_bench.sro_reassessment import align_predictions, assess_revision, paired_interval


def test_predictions_are_aligned_by_id_not_row_position():
    expected = pd.DataFrame(dict(xrd_sample_id=["a", "b"], parent_group_id=["p", "q"]))
    predicted = pd.DataFrame(dict(xrd_sample_id=["b", "a"], parent_group_id=["q", "p"],
                                  sro_predictions=[[2., 3.], [0., 1.]]))
    np.testing.assert_array_equal(align_predictions(expected, predicted, 2), [[0., 1.], [2., 3.]])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "parent", "width", "nonfinite"])
def test_predictions_fail_closed(fault):
    expected = pd.DataFrame(dict(xrd_sample_id=["a"], parent_group_id=["p"]))
    predicted = pd.DataFrame(dict(xrd_sample_id=["a"], parent_group_id=["p"], sro_predictions=[[0., 1.]]))
    if fault == "missing":
        predicted = predicted.iloc[:0]
    elif fault == "duplicate":
        predicted = pd.concat([predicted, predicted])
    elif fault == "parent":
        predicted.loc[0, "parent_group_id"] = "wrong"
    elif fault == "width":
        predicted.at[0, "sro_predictions"] = [0.]
    else:
        predicted.at[0, "sro_predictions"] = [0., np.nan]
    with pytest.raises(ValueError):
        align_predictions(expected, predicted, 2)


def test_added_targets_are_not_silently_masked_out():
    old_y = np.array([[0., 0.], [0., 0.]])
    new_y = np.array([[0., 1.], [0., 1.]])
    old_mask = np.array([[True, False], [True, False]])
    new_mask = np.ones((2, 2), dtype=bool)
    predictions = np.zeros((2, 2))
    summary, parents = assess_revision(old_y, old_mask, new_y, new_mask, predictions,
                                       ["p", "p"], [1, 2], bootstrap=100)
    assert summary["A"]["masked_parent_macro_mae"] == 0.
    assert summary["B"]["masked_parent_macro_mae"] == .5
    assert summary["newly_valid"]["masked_parent_macro_mae"] == 1.
    assert summary["A"]["parent_count"] == 1
    assert len(parents) == 1
    assert summary["B_minus_A"]["mean"] == .5


def test_changed_common_truth_is_rejected():
    with pytest.raises(ValueError):
        assess_revision(np.zeros((1, 1)), np.ones((1, 1), bool), np.ones((1, 1)),
                        np.ones((1, 1), bool), np.zeros((1, 1)), ["p"], [1], bootstrap=10)


def test_bootstrap_uses_matched_parents_and_is_reproducible():
    difference = pd.Series([.1, -.3, .2], index=["a", "b", "c"])
    first = paired_interval(difference, 100, 2027)
    assert first == paired_interval(difference, 100, 2027)
    assert first["parent_count"] == 3
