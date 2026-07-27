"""Model-independent metrics for HEA validation experiments."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .scoring import require_complete_predictions


def masked_parent_macro_sro(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    parent_ids: Sequence[str],
    shells: Sequence[int],
) -> dict[str, object]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if truth.shape != prediction.shape or truth.shape != mask.shape:
        raise ValueError("truth, prediction, and mask must have identical shapes")
    if truth.ndim != 2 or len(parent_ids) != truth.shape[0]:
        raise ValueError("parent IDs must align with two-dimensional predictions")
    if len(shells) != truth.shape[1] or not mask.any():
        raise ValueError("shell labels must align and at least one target must be valid")
    if not np.isfinite(prediction[mask]).all():
        raise ValueError("valid predictions must be finite")

    absolute_error = np.abs(prediction - truth)
    squared_error = np.square(prediction - truth)
    parent_row_errors: dict[str, list[float]] = defaultdict(list)
    for index, parent_id in enumerate(parent_ids):
        if mask[index].any():
            parent_row_errors[str(parent_id)].append(
                float(absolute_error[index][mask[index]].mean())
            )
    parent_mae = {
        parent: float(np.mean(errors)) for parent, errors in parent_row_errors.items()
    }

    shell_spearman: dict[str, float | None] = {}
    shell_array = np.asarray(shells)
    for shell in sorted(set(shells)):
        shell_mask = mask & (shell_array == shell)[None, :]
        if shell_mask.sum() < 2:
            correlation = None
        else:
            value = float(spearmanr(truth[shell_mask], prediction[shell_mask]).statistic)
            correlation = value if math.isfinite(value) else None
        shell_spearman[f"shell{shell}"] = correlation

    return {
        "masked_parent_macro_mae": float(np.mean(list(parent_mae.values()))),
        "rmse": float(np.sqrt(squared_error[mask].mean())),
        "parent_count": len(parent_mae),
        "valid_target_count": int(mask.sum()),
        "shell_spearman": shell_spearman,
    }


def family_classification_metrics(
    truth: Sequence[str], prediction: Sequence[str], classes: Sequence[str]
) -> dict[str, object]:
    if len(truth) != len(prediction):
        raise ValueError("family truth and predictions must have equal length")
    allowed = set(classes)
    unknown = (set(truth) | set(prediction)) - allowed
    if unknown:
        raise ValueError(f"unknown family labels: {sorted(unknown)}")
    f1_values: list[float] = []
    recalls: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
    }


def occupancy_ranking_metrics(
    truth_scores: Sequence[float],
    predicted_scores: Sequence[float],
    group_ids: Sequence[str],
    *,
    identifiable: Sequence[bool],
) -> dict[str, object]:
    lengths = {len(truth_scores), len(predicted_scores), len(group_ids), len(identifiable)}
    if len(lengths) != 1 or len(truth_scores) == 0:
        raise ValueError("ranking inputs must be non-empty and aligned")
    truth = np.asarray(truth_scores, dtype=float)
    prediction = np.asarray(predicted_scores, dtype=float)
    keep = np.asarray(identifiable, dtype=bool)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("ranking scores must be finite")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, (group, eligible) in enumerate(zip(group_ids, keep)):
        if eligible:
            groups[str(group)].append(index)
    correct = 0.0
    comparisons = 0
    reciprocal_ranks: list[float] = []
    top3: list[float] = []
    for indices in groups.values():
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                truth_delta = truth[left] - truth[right]
                if truth_delta == 0:
                    continue
                prediction_delta = prediction[left] - prediction[right]
                comparisons += 1
                if prediction_delta == 0:
                    correct += 0.5
                else:
                    correct += float(np.sign(prediction_delta) == np.sign(truth_delta))
        if indices:
            true_best = indices[int(np.argmax(truth[indices]))]
            predicted_order = sorted(indices, key=lambda index: (-prediction[index], index))
            rank = predicted_order.index(true_best) + 1
            reciprocal_ranks.append(1.0 / rank)
            top3.append(float(rank <= 3))
    return {
        "pair_accuracy": correct / comparisons if comparisons else None,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "top3_accuracy": float(np.mean(top3)) if top3 else None,
        "identifiable_coverage": float(keep.mean()),
        "eligible_item_count": int(keep.sum()),
        "pair_comparison_count": comparisons,
    }


def prediction_completeness(
    expected_sample_ids: Sequence[str], predicted_sample_ids: Sequence[str]
) -> dict[str, int]:
    require_complete_predictions(expected_sample_ids, predicted_sample_ids)
    return {"expected_count": len(expected_sample_ids), "predicted_count": len(predicted_sample_ids)}


def robustness_retention(
    clean_scores: Mapping[str, float], perturbed_scores: Mapping[str, float]
) -> dict[str, float | int]:
    if not clean_scores or set(clean_scores) != set(perturbed_scores):
        raise ValueError("clean and perturbed scores must have identical non-empty IDs")
    ratios: list[float] = []
    for sample_id in sorted(clean_scores):
        clean = float(clean_scores[sample_id])
        perturbed = float(perturbed_scores[sample_id])
        if not math.isfinite(clean) or not math.isfinite(perturbed) or clean <= 0:
            raise ValueError("retention scores must be finite and clean scores positive")
        ratios.append(perturbed / clean)
    return {
        "mean_retention": float(np.mean(ratios)),
        "geometric_retention": float(math.prod(ratios) ** (1.0 / len(ratios))),
        "matched_count": len(ratios),
    }
