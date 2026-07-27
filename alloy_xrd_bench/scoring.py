"""Reference score composition for the SiteXRD-Bench HEA track."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping


CORE_DIMENSIONS = ("sro", "occupancy", "robustness", "ood", "trust")


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _score(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be finite and in [0, 100]")
    return value


def normalized_sro_score(model_mae: float, baseline_mae: float) -> float:
    """Return baseline-relative SRO skill on a 0--100 scale."""
    model_mae = float(model_mae)
    baseline_mae = float(baseline_mae)
    if not math.isfinite(model_mae) or model_mae < 0.0:
        raise ValueError("model_mae must be finite and non-negative")
    if not math.isfinite(baseline_mae) or baseline_mae <= 0.0:
        raise ValueError("baseline_mae must be finite and positive")
    return 100.0 * max(0.0, 1.0 - model_mae / baseline_mae)


def occupancy_score(
    pair_accuracy: float,
    mrr: float,
    top_k_accuracy: float,
    *,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Combine same-composition ranking and retrieval metrics."""
    metrics = (
        _unit_interval(pair_accuracy, "pair_accuracy"),
        _unit_interval(mrr, "mrr"),
        _unit_interval(top_k_accuracy, "top_k_accuracy"),
    )
    if len(weights) != 3 or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must contain three non-negative values")
    total = float(sum(weights))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("weights must sum to 1")
    return 100.0 * sum(weight * metric for weight, metric in zip(weights, metrics))


def robustness_score(
    clean_scores: Mapping[str, float],
    perturbed_scores: Mapping[str, float],
    *,
    clean_floor: float = 40.0,
) -> float:
    """Compute geometric performance retention after enforcing a clean floor."""
    if not clean_scores or set(clean_scores) != set(perturbed_scores):
        raise ValueError("clean_scores and perturbed_scores must have identical non-empty keys")
    floor = _score(clean_floor, "clean_floor")
    ratios: list[float] = []
    for name in sorted(clean_scores):
        clean = _score(clean_scores[name], f"clean_scores[{name}]")
        perturbed = _score(perturbed_scores[name], f"perturbed_scores[{name}]")
        if clean < floor:
            return 0.0
        ratios.append(min(1.0, perturbed / clean))
    return 100.0 * math.prod(ratios) ** (1.0 / len(ratios))


def trustworthiness_score(
    *,
    ece: float,
    empirical_coverage: float,
    target_coverage: float,
    aurc: float,
    oracle_aurc: float,
    reference_aurc: float,
    identifiability_auroc: float,
    coverage_tolerance: float = 0.10,
) -> float:
    """Aggregate calibration, coverage, selective risk, and ambiguity detection."""
    ece = _unit_interval(ece, "ece")
    empirical_coverage = _unit_interval(empirical_coverage, "empirical_coverage")
    target_coverage = _unit_interval(target_coverage, "target_coverage")
    identifiability_auroc = _unit_interval(identifiability_auroc, "identifiability_auroc")
    if coverage_tolerance <= 0.0:
        raise ValueError("coverage_tolerance must be positive")
    if not oracle_aurc <= aurc or not oracle_aurc < reference_aurc:
        raise ValueError("AURC values must satisfy oracle_aurc <= aurc and oracle_aurc < reference_aurc")

    calibration = max(0.0, 1.0 - ece)
    coverage = max(
        0.0,
        1.0 - abs(empirical_coverage - target_coverage) / coverage_tolerance,
    )
    selective = max(
        0.0,
        min(1.0, 1.0 - (aurc - oracle_aurc) / (reference_aurc - oracle_aurc)),
    )
    identifiability = max(0.0, min(1.0, 2.0 * (identifiability_auroc - 0.5)))
    components = (calibration, coverage, selective, identifiability)
    return 100.0 * math.prod(components) ** (1.0 / len(components))


def reference_overall_score(scores: Mapping[str, float]) -> float:
    """Return the geometric mean of supplied normalized capability scores."""
    if not scores:
        raise ValueError("scores must not be empty")
    values = [_score(value, name) for name, value in sorted(scores.items())]
    return math.prod(values) ** (1.0 / len(values))


def complete_capability(
    family_macro_f1: float,
    core_scores: Mapping[str, float],
    *,
    family_gate: float = 0.90,
    core_floor: float = 40.0,
) -> bool:
    """Apply the qualification gate used by the complete-capability badge."""
    family_macro_f1 = _unit_interval(family_macro_f1, "family_macro_f1")
    missing = set(CORE_DIMENSIONS) - set(core_scores)
    if missing:
        raise ValueError(f"missing core scores: {sorted(missing)}")
    return family_macro_f1 >= family_gate and all(
        _score(core_scores[name], name) >= core_floor for name in CORE_DIMENSIONS
    )


def require_complete_predictions(
    expected_sample_ids: Iterable[str], predicted_sample_ids: Iterable[str]
) -> None:
    """Reject partial, duplicate, or out-of-scope prediction submissions."""
    expected = list(expected_sample_ids)
    predicted = list(predicted_sample_ids)
    duplicate_expected = sorted(key for key, count in Counter(expected).items() if count > 1)
    duplicate_predicted = sorted(key for key, count in Counter(predicted).items() if count > 1)
    if duplicate_expected:
        raise ValueError(f"duplicate expected sample IDs: {duplicate_expected[:5]}")
    if duplicate_predicted:
        raise ValueError(f"duplicate prediction sample IDs: {duplicate_predicted[:5]}")
    missing = sorted(set(expected) - set(predicted))
    if missing:
        raise ValueError(f"missing prediction sample IDs: {missing[:5]}")
    extra = sorted(set(predicted) - set(expected))
    if extra:
        raise ValueError(f"unexpected prediction sample IDs: {extra[:5]}")


def diagnostic_flags(
    *,
    family_macro_f1: float,
    xrd_occupancy_score: float,
    composition_only_occupancy_score: float,
    family_score: float,
    occupancy_score_value: float,
    clean_score: float,
    perturbed_score: float,
    id_score: float,
    ood_score: float,
    nonidentifiable_confidence: float,
) -> list[str]:
    """Return fixed, human-readable failure modes for a scorecard."""
    flags: list[str] = []
    if _unit_interval(family_macro_f1, "family_macro_f1") < 0.90:
        flags.append("basic_diffraction_fail")
    if _score(xrd_occupancy_score, "xrd_occupancy_score") - _score(
        composition_only_occupancy_score, "composition_only_occupancy_score"
    ) < 5.0:
        flags.append("shortcut_risk")
    if _score(family_score, "family_score") >= 90.0 and _score(
        occupancy_score_value, "occupancy_score_value"
    ) < 40.0:
        flags.append("occupancy_blindness")
    clean = _score(clean_score, "clean_score")
    perturbed = _score(perturbed_score, "perturbed_score")
    if clean >= 40.0 and perturbed / clean < 0.75:
        flags.append("perturbation_brittleness")
    id_value = _score(id_score, "id_score")
    ood_value = _score(ood_score, "ood_score")
    if id_value >= 40.0 and ood_value / id_value < 0.65:
        flags.append("ood_collapse")
    if _unit_interval(nonidentifiable_confidence, "nonidentifiable_confidence") > 0.80:
        flags.append("overconfidence_risk")
    return flags
