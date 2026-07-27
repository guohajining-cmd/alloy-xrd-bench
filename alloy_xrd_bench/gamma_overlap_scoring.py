"""Permutation-invariant scoring for gamma/gamma-prime peak components."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gamma_overlap_physics import canonical_cubic_hkl, pseudo_voigt_profile


REQUIRED_FIELDS = {
    "sample_id",
    "component_id",
    "phase",
    "h",
    "k",
    "l",
    "center_deg",
    "fwhm_deg",
    "integrated_intensity",
}

DEGENERATE_GAMMA_PRIME_HKLS = ((2, 2, 1), (3, 0, 0))
DEGENERATE_GAMMA_PRIME_GROUP = "gamma_prime:221|300"


def hkl_group_id(phase: str, h: int, k: int, l: int) -> str:
    canonical = canonical_cubic_hkl((int(h), int(k), int(l)))
    if phase == "gamma_prime" and canonical in DEGENERATE_GAMMA_PRIME_HKLS:
        return DEGENERATE_GAMMA_PRIME_GROUP
    return f"{phase}:{canonical[0]}{canonical[1]}{canonical[2]}"


def build_scoring_components(
    component_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create a scoring view that merges powder-degenerate HKL reflections."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in component_rows:
        group_id = str(row.get("hkl_group_id") or hkl_group_id(
            str(row["phase"]), int(row["h"]), int(row["k"]), int(row["l"])
        ))
        grouped[(str(row["sample_id"]), group_id)].append(row)
    result: list[dict[str, Any]] = []
    for (sample_id, group_id), rows in sorted(grouped.items()):
        first = dict(sorted(rows, key=lambda row: str(row["component_id"]))[0])
        allowed = [list(value) for value in sorted({
            canonical_cubic_hkl((int(row["h"]), int(row["k"]), int(row["l"])))
            for row in rows
        })]
        if group_id == DEGENERATE_GAMMA_PRIME_GROUP:
            allowed = [list(value) for value in DEGENERATE_GAMMA_PRIME_HKLS]
        weights = np.asarray(
            [max(float(row["integrated_intensity"]), 0.0) for row in rows], dtype=float
        )
        total = float(weights.sum())
        if total <= 0:
            weights = np.ones(len(rows), dtype=float)
        merged = first | {
            "sample_id": sample_id,
            "component_id": (
                f"hklgroup::{sample_id}::{group_id.replace(':', '_').replace('|', '_')}"
            ),
            "hkl_group_id": group_id,
            "allowed_hkls": allowed,
            "allowed_hkls_json": json.dumps(allowed, separators=(",", ":")),
            "degenerate_group": len(allowed) > 1,
            "center_deg": float(np.average(
                [float(row["center_deg"]) for row in rows], weights=weights
            )),
            "fwhm_deg": float(np.average(
                [float(row["fwhm_deg"]) for row in rows], weights=weights
            )),
            "eta": float(np.average(
                [float(row.get("eta", 0.5)) for row in rows], weights=weights
            )),
            "integrated_intensity": total,
            "merged_integrated_intensity": total,
        }
        if "score_eligible" in first:
            merged["score_eligible"] = any(bool(row.get("score_eligible")) for row in rows)
        result.append(merged)
    return result


def score_component_predictions(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_parent_ids: Mapping[str, str] | None = None,
) -> dict[str, float]:
    _validate_rows(truth_rows, predictions=False)
    _validate_rows(prediction_rows, predictions=True)
    truths = _group_by_sample(truth_rows)
    predictions = _group_by_sample(prediction_rows)
    for rows in truths.values():
        rows.sort(key=lambda row: str(row["component_id"]))
    for rows in predictions.values():
        rows.sort(key=lambda row: str(row["component_id"]))
    expected = set(str(value) for value in expected_sample_ids) if expected_sample_ids is not None else set(truths)
    parent_lookup = (
        {str(sample_id): str(parent_id) for sample_id, parent_id in expected_parent_ids.items()}
        if expected_parent_ids is not None
        else {}
    )
    if expected_parent_ids is not None and set(parent_lookup) != expected:
        missing = sorted(expected - set(parent_lookup))
        extra = sorted(set(parent_lookup) - expected)
        raise ValueError(
            f"expected parent mapping mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if expected_sample_ids is not None:
        for sample_id in expected:
            truths.setdefault(sample_id, [])
    unexpected = sorted(set(predictions) - expected)
    if unexpected:
        raise ValueError(f"unexpected prediction sample IDs: {unexpected[:5]}")

    sample_results: list[dict[str, Any]] = []
    for sample_id in sorted(expected):
        true = truths[sample_id]
        pred = predictions.get(sample_id, [])
        matches = _match_components(true, pred)
        sample_results.append(_sample_metrics(true, pred, matches))

    parent_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, result in zip(sorted(expected), sample_results):
        parent = parent_lookup.get(sample_id)
        if parent is None:
            parent = str(truths[sample_id][0].get("parent_pair_id", sample_id)) if truths[sample_id] else sample_id
        parent_results[parent].append(result)
    aggregated = [_aggregate_samples(rows) for rows in parent_results.values()]
    return _aggregate_parents(aggregated)


def window_component_truth(
    component_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project full-pattern component truth onto public window sample IDs."""

    import json

    by_id = {str(row["component_id"]): row for row in component_rows}
    result: list[dict[str, Any]] = []
    for window in window_rows:
        raw_ids = window.get("target_component_ids_json", "[]")
        if raw_ids is None or (isinstance(raw_ids, float) and np.isnan(raw_ids)):
            continue
        for component_id in json.loads(str(raw_ids)):
            component = by_id.get(str(component_id))
            if component is None or not bool(component.get("score_eligible", True)):
                continue
            result.append(
                {
                    **dict(component),
                    "sample_id": str(window["window_id"]),
                    "component_id": f"{window['window_id']}::{component_id}",
                    "parent_pair_id": str(window["parent_pair_id"]),
                }
            )
    return result


def _match_components(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    if not truth or not predictions:
        return []
    prohibitive = 1e6
    costs = np.full((len(truth), len(predictions)), prohibitive, dtype=float)
    for true_index, true in enumerate(truth):
        true_key = _phase_hkl(true)
        gate = max(0.15, 1.5 * float(true["fwhm_deg"]))
        for pred_index, pred in enumerate(predictions):
            if _phase_hkl(pred) != true_key:
                continue
            center_error = abs(float(pred["center_deg"]) - float(true["center_deg"]))
            if center_error > gate:
                continue
            center_cost = min(1.0, center_error / gate)
            width_cost = min(
                1.0,
                abs(float(pred["fwhm_deg"]) - float(true["fwhm_deg"]))
                / float(true["fwhm_deg"]),
            )
            profile_cost = _profile_distance(true, pred)
            costs[true_index, pred_index] = (center_cost + width_cost + profile_cost) / 3.0
    true_indices, pred_indices = linear_sum_assignment(costs)
    return [
        (int(true_index), int(pred_index))
        for true_index, pred_index in zip(true_indices, pred_indices)
        if costs[true_index, pred_index] < prohibitive
    ]


def _sample_metrics(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    matches: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    matched_prediction_indices = {j for _, j in matches}
    confidence_labels = [
        (float(row.get("confidence", 1.0)), int(index in matched_prediction_indices))
        for index, row in enumerate(predictions)
    ]
    center_errors = [abs(float(predictions[j]["center_deg"]) - float(truth[i]["center_deg"])) for i, j in matches]
    width_errors = [abs(float(predictions[j]["fwhm_deg"]) - float(truth[i]["fwhm_deg"])) for i, j in matches]
    area_errors = [abs(float(predictions[j]["integrated_intensity"]) - float(truth[i]["integrated_intensity"])) for i, j in matches]
    profile_errors = [_profile_distance(truth[i], predictions[j]) for i, j in matches]
    region_tp, region_fp, region_fn = _overlap_region_counts(truth, predictions)
    truth_overlap = len(truth) >= 2
    predicted_overlap = len(predictions) >= 2
    return {
        "tp": len(matches),
        "fp": len(predictions) - len(matches),
        "fn": len(truth) - len(matches),
        "truth_count": len(truth),
        "pred_count": len(predictions),
        "center_errors": center_errors,
        "width_errors": width_errors,
        "area_errors": area_errors,
        "profile_errors": profile_errors,
        "confidence_labels": confidence_labels,
        "high_confidence_matches": sum(
            float(predictions[j].get("confidence", 1.0)) >= 0.9 for _, j in matches
        ),
        "forward_reconstruction_l1": _forward_reconstruction_distance(truth, predictions),
        "overlap_region_tp": region_tp,
        "overlap_region_fp": region_fp,
        "overlap_region_fn": region_fn,
        "overlap_tp": int(truth_overlap and predicted_overlap),
        "overlap_fp": int(not truth_overlap and predicted_overlap),
        "overlap_fn": int(truth_overlap and not predicted_overlap),
    }


def _aggregate_samples(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    center = [value for row in rows for value in row["center_errors"]]
    width = [value for row in rows for value in row["width_errors"]]
    area = [value for row in rows for value in row["area_errors"]]
    profile = [value for row in rows for value in row["profile_errors"]]
    confidence_labels = [value for row in rows for value in row["confidence_labels"]]
    overlap_tp = sum(int(row["overlap_tp"]) for row in rows)
    overlap_fp = sum(int(row["overlap_fp"]) for row in rows)
    overlap_fn = sum(int(row["overlap_fn"]) for row in rows)
    region_tp = sum(int(row["overlap_region_tp"]) for row in rows)
    region_fp = sum(int(row["overlap_region_fp"]) for row in rows)
    region_fn = sum(int(row["overlap_region_fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    overlap_precision = overlap_tp / (overlap_tp + overlap_fp) if overlap_tp + overlap_fp else 0.0
    overlap_recall = overlap_tp / (overlap_tp + overlap_fn) if overlap_tp + overlap_fn else 0.0
    region_precision, region_recall, region_f1 = _precision_recall_f1(region_tp, region_fp, region_fn)
    calibration = _calibration_metrics(confidence_labels)
    return {
        "component_precision": precision,
        "component_recall": recall,
        "component_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "phase_hkl_accuracy": tp / max(tp + fn, tp + fp, 1),
        "count_accuracy": sum(row["truth_count"] == row["pred_count"] for row in rows) / len(rows),
        "center_mae_deg": float(np.mean(center)) if center else math.nan,
        "fwhm_mae_deg": float(np.mean(width)) if width else math.nan,
        "area_mae": float(np.mean(area)) if area else math.nan,
        "profile_distance": float(np.mean(profile)) if profile else math.nan,
        "confidence_ece": calibration["confidence_ece"],
        "coverage_at_90_confidence": (
            sum(int(row["high_confidence_matches"]) for row in rows)
            / max(sum(int(row["truth_count"]) for row in rows), 1)
        ),
        "aurc": calibration["aurc"],
        "risk_at_80_coverage": calibration["risk_at_80_coverage"],
        "forward_reconstruction_l1": float(np.mean([row["forward_reconstruction_l1"] for row in rows])),
        "overlap_region_precision": region_precision,
        "overlap_region_recall": region_recall,
        "overlap_region_f1": region_f1,
        "overlap_detection_precision": overlap_precision,
        "overlap_detection_recall": overlap_recall,
        "overlap_detection_f1": 2.0 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall) if overlap_precision + overlap_recall else 0.0,
    }


def _aggregate_parents(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "component_precision": 0.0,
            "component_recall": 0.0,
            "component_f1": 0.0,
            "phase_hkl_accuracy": 0.0,
            "count_accuracy": 0.0,
            "center_mae_deg": math.nan,
            "fwhm_mae_deg": math.nan,
            "area_mae": math.nan,
            "profile_distance": math.nan,
            "confidence_ece": math.nan,
            "coverage_at_90_confidence": 0.0,
            "aurc": math.nan,
            "risk_at_80_coverage": math.nan,
            "forward_reconstruction_l1": math.nan,
            "overlap_region_precision": 0.0,
            "overlap_region_recall": 0.0,
            "overlap_region_f1": 0.0,
            "overlap_detection_precision": 0.0,
            "overlap_detection_recall": 0.0,
            "overlap_detection_f1": 0.0,
        }
    result: dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    return result


def _calibration_metrics(rows: Sequence[tuple[float, int]]) -> dict[str, float]:
    if not rows:
        return {"confidence_ece": math.nan, "aurc": math.nan, "risk_at_80_coverage": math.nan}
    confidence = np.asarray([row[0] for row in rows], dtype=float)
    correct = np.asarray([row[1] for row in rows], dtype=float)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(rows) + 1)
    index_80 = max(0, int(math.ceil(0.8 * len(rows))) - 1)
    return {
        "confidence_ece": float(ece),
        "aurc": float(cumulative_risk.mean()),
        "risk_at_80_coverage": float(cumulative_risk[index_80]),
    }


def _phase_hkl(row: Mapping[str, Any]) -> tuple[str, object]:
    phase = str(row["phase"])
    if row.get("hkl_group_id"):
        return phase, str(row["hkl_group_id"])
    if phase not in {"gamma", "gamma_prime"}:
        return phase, (-1, -1, -1)
    return phase, canonical_cubic_hkl((int(row["h"]), int(row["k"]), int(row["l"])))


def _profile_distance(truth: Mapping[str, Any], prediction: Mapping[str, Any]) -> float:
    true_center = float(truth["center_deg"])
    pred_center = float(prediction["center_deg"])
    true_width = float(truth["fwhm_deg"])
    pred_width = float(prediction["fwhm_deg"])
    margin = max(1.0, 5.0 * true_width, 5.0 * pred_width)
    grid = np.linspace(min(true_center, pred_center) - margin, max(true_center, pred_center) + margin, 801)
    true_profile = pseudo_voigt_profile(
        grid,
        center_deg=true_center,
        fwhm_deg=true_width,
        integrated_intensity=1.0,
        eta=float(truth.get("eta", 0.5)),
    )
    pred_profile = pseudo_voigt_profile(
        grid,
        center_deg=pred_center,
        fwhm_deg=pred_width,
        integrated_intensity=1.0,
        eta=float(prediction.get("eta", 0.5)),
    )
    return min(1.0, 0.5 * float(np.trapezoid(np.abs(true_profile - pred_profile), grid)))


def _overlap_region_counts(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    support_fwhm: float = 1.5,
    step_deg: float = 0.02,
) -> tuple[int, int, int]:
    rows = list(truth) + list(predictions)
    if not rows:
        return 0, 0, 0
    left = min(float(row["center_deg"]) - support_fwhm * float(row["fwhm_deg"]) for row in rows)
    right = max(float(row["center_deg"]) + support_fwhm * float(row["fwhm_deg"]) for row in rows)
    grid = np.arange(left, right + step_deg / 2.0, step_deg)

    def mask(components: Sequence[Mapping[str, Any]]) -> np.ndarray:
        count = np.zeros(len(grid), dtype=np.int16)
        for row in components:
            center = float(row["center_deg"])
            radius = support_fwhm * float(row["fwhm_deg"])
            count += (grid >= center - radius) & (grid <= center + radius)
        return count >= 2

    true_mask = mask(truth)
    pred_mask = mask(predictions)
    return (
        int(np.count_nonzero(true_mask & pred_mask)),
        int(np.count_nonzero(~true_mask & pred_mask)),
        int(np.count_nonzero(true_mask & ~pred_mask)),
    )


def _forward_reconstruction_distance(
    truth: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> float:
    rows = list(truth) + list(predictions)
    if not rows:
        return 0.0
    margin = max(1.0, max(5.0 * float(row["fwhm_deg"]) for row in rows))
    left = min(float(row["center_deg"]) for row in rows) - margin
    right = max(float(row["center_deg"]) for row in rows) + margin
    grid = np.arange(left, right + 0.01, 0.02)

    def render(components: Sequence[Mapping[str, Any]]) -> np.ndarray:
        total = np.zeros(len(grid), dtype=float)
        for row in components:
            total += pseudo_voigt_profile(
                grid,
                center_deg=float(row["center_deg"]),
                fwhm_deg=float(row["fwhm_deg"]),
                integrated_intensity=float(row["integrated_intensity"]),
                eta=float(row.get("eta", 0.5)),
            )
        return total

    true_curve = render(truth)
    pred_curve = render(predictions)
    numerator = float(np.trapezoid(np.abs(true_curve - pred_curve), grid))
    denominator = float(np.trapezoid(np.abs(true_curve) + np.abs(pred_curve), grid))
    return numerator / denominator if denominator > 0 else 0.0


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _group_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    return grouped


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, predictions: bool) -> None:
    identifiers: set[tuple[str, str]] = set()
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"component row is missing fields: {sorted(missing)}")
        identifier = (str(row["sample_id"]), str(row["component_id"]))
        if identifier in identifiers:
            raise ValueError(f"duplicate component ID within sample: {identifier}")
        identifiers.add(identifier)
        for field in ("center_deg", "fwhm_deg", "integrated_intensity"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if float(row["fwhm_deg"]) <= 0 or float(row["integrated_intensity"]) < 0:
            raise ValueError("FWHM must be positive and integrated intensity non-negative")
        if predictions:
            if "confidence" not in row:
                raise ValueError("prediction row is missing fields: ['confidence']")
            confidence = float(row["confidence"])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("prediction confidence must be in [0, 1]")
