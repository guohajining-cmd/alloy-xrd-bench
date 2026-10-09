"""Matched A/B/C assessment under frozen sample identities and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hea_validation_metrics import masked_parent_macro_sro


def align_predictions(expected, predictions, width):
    key = "xrd_sample_id"
    required = {key, "parent_group_id", "sro_predictions"}
    if not required.issubset(predictions.columns):
        raise ValueError("Missing prediction columns")
    if expected[key].isna().any() or predictions[key].isna().any():
        raise ValueError("Null sample ID")
    if expected[key].duplicated().any() or predictions[key].duplicated().any():
        raise ValueError("Duplicate sample ID")
    if set(expected[key]) != set(predictions[key]):
        raise ValueError("Prediction IDs must exactly cover the requested split")
    ordered = predictions.set_index(key).loc[expected[key]]
    if not np.array_equal(expected.parent_group_id.to_numpy(), ordered.parent_group_id.to_numpy()):
        raise ValueError("Prediction parent identity mismatch")
    try:
        array = np.stack(ordered.sro_predictions.to_numpy()).astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid prediction vectors") from exc
    if array.shape != (len(expected), width) or not np.isfinite(array).all():
        raise ValueError("Prediction shape or finite-value check failed")
    return array


def parent_errors(truth, prediction, mask, parent_ids):
    eligible = mask.any(axis=1)
    error = np.where(mask, np.abs(prediction - truth), 0).sum(axis=1)
    rows = error[eligible] / mask[eligible].sum(axis=1)
    return pd.Series(rows, index=np.asarray(parent_ids)[eligible]).groupby(level=0, sort=True).mean()


def paired_interval(difference, iterations=2000, seed=2027):
    values = np.asarray(difference, dtype=float)
    if not len(values) or not np.isfinite(values).all() or iterations < 1:
        raise ValueError("Bootstrap requires finite matched parent errors")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for index in range(iterations):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                parent_count=len(values), bootstrap_iterations=iterations, seed=seed,
                method="Paired parent-group percentile bootstrap; conditional on fitted models")


def assess_revision(old_truth, old_mask, new_truth, new_mask, old_prediction, parent_ids,
                    shells, corrected_prediction=None, bootstrap=2000):
    arrays = [old_truth, old_mask, new_truth, new_mask, old_prediction]
    if any(np.shape(value) != np.shape(old_truth) for value in arrays):
        raise ValueError("Unaligned assessment arrays")
    old_mask, new_mask = np.asarray(old_mask, bool), np.asarray(new_mask, bool)
    if np.any(old_mask & ~new_mask) or not np.allclose(old_truth[old_mask], new_truth[old_mask], atol=1e-10, rtol=0):
        raise ValueError("Revision changed or removed previously valid labels")
    if not np.isfinite(new_truth[new_mask]).all():
        raise ValueError("Non-finite revised truth")
    added = new_mask & ~old_mask
    if added.any() and not np.allclose(new_truth[added], 1., atol=1e-10, rtol=0):
        raise ValueError("Newly valid targets must be zero-count pairs (alpha=1)")
    def score(truth, prediction, mask):
        return masked_parent_macro_sro(truth, prediction, mask, parent_ids, shells) if mask.any() else None
    summary = dict(A=score(old_truth, old_prediction, old_mask),
                   B=score(new_truth, old_prediction, new_mask),
                   newly_valid=score(new_truth, old_prediction, added),
                   common_labels=score(new_truth, old_prediction, old_mask),
                   newly_valid_target_count=int(added.sum()),
                   affected_test_rows=int(added.any(axis=1).sum()),
                   caveat="B minus A changes evaluation support, not model performance; C minus B isolates retraining on the same corrected test targets.")
    parents = pd.DataFrame(dict(A=parent_errors(old_truth, old_prediction, old_mask, parent_ids),
                                B=parent_errors(new_truth, old_prediction, new_mask, parent_ids)))
    parents.index.name = "parent_group_id"
    matched = parents.dropna(subset=["A", "B"])
    summary["B_minus_A"] = paired_interval(matched.B - matched.A, bootstrap)
    if corrected_prediction is not None:
        summary["C"] = score(new_truth, corrected_prediction, new_mask)
        summary["C_common_labels"] = score(new_truth, corrected_prediction, old_mask)
        summary["C_newly_valid"] = score(new_truth, corrected_prediction, added)
        parents["C"] = parent_errors(new_truth, corrected_prediction, new_mask, parent_ids)
        matched = parents.dropna(subset=["B", "C"])
        summary["C_minus_B"] = paired_interval(matched.C - matched.B, bootstrap)
    return summary, parents
