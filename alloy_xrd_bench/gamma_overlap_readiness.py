"""Readiness gate for the frozen Gamma-Decomp v1.1 experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .gamma_overlap_public import validate_public_release


SEEDS = (2027, 2028, 2029)
TASKS = ("full", "window")
LEARNED_MODELS = ("peak-feature", "resnet-set", "transformer-set", "physics-constrained")
NEURAL_RUNS = (
    "resnet-set", "transformer-set", "physics-constrained", "reconstruction-only",
    "ablations/weak_peaks", "ablations/no_perturbation_training", "ablations/random_split",
)
REQUIRED_METRICS = {
    "component_f1",
    "phase_hkl_accuracy",
    "count_accuracy",
    "center_mae_deg",
    "fwhm_mae_deg",
    "area_mae",
    "forward_reconstruction_l1",
    "confidence_ece",
    "coverage_at_90_confidence",
    "aurc",
    "risk_at_80_coverage",
}
ALWAYS_FINITE_METRICS = {
    "component_f1",
    "phase_hkl_accuracy",
    "count_accuracy",
    "forward_reconstruction_l1",
    "coverage_at_90_confidence",
}
MATCHED_ONLY_METRICS = {"center_mae_deg", "fwhm_mae_deg", "area_mae"}
CALIBRATION_METRICS = {"confidence_ece", "aurc", "risk_at_80_coverage"}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score_artifact_issues(score_dir: Path, *, task: str) -> list[str]:
    issues: list[str] = []
    scorecard = _read_json(score_dir / "scorecard.json")
    if scorecard is None:
        return ["missing_scorecard"]
    metrics = scorecard.get("metrics", {})
    missing_metrics = sorted(REQUIRED_METRICS.difference(metrics))
    if missing_metrics:
        issues.append("missing_metrics:" + ",".join(missing_metrics))
    nonfinite_metrics = []
    for key in sorted(ALWAYS_FINITE_METRICS.intersection(metrics)):
        try:
            finite = math.isfinite(float(metrics[key]))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            nonfinite_metrics.append(key)
    if nonfinite_metrics:
        issues.append("nonfinite_metrics:" + ",".join(nonfinite_metrics))
    undefined_matched = {
        key for key in MATCHED_ONLY_METRICS
        if key in metrics and not _is_finite(metrics[key])
    }
    if undefined_matched and undefined_matched != MATCHED_ONLY_METRICS:
        issues.append("partially_undefined_matched_metrics")
    if undefined_matched:
        component_f1 = metrics.get("component_f1")
        phase_hkl_accuracy = metrics.get("phase_hkl_accuracy")
        if not (
            _is_finite(component_f1)
            and float(component_f1) == 0.0
            and _is_finite(phase_hkl_accuracy)
            and float(phase_hkl_accuracy) == 0.0
        ):
            issues.append("undefined_matched_metrics_without_abstention")
    undefined_calibration = {
        key for key in CALIBRATION_METRICS
        if key in metrics and not _is_finite(metrics[key])
    }
    if undefined_calibration and undefined_calibration != CALIBRATION_METRICS:
        issues.append("partially_undefined_calibration_metrics")
    if undefined_calibration:
        validation = _read_json(score_dir / "submission_validation.json") or {}
        try:
            component_count = int(validation.get("component_count", -1))
        except (TypeError, ValueError):
            component_count = -1
        if validation.get("status") != "pass" or component_count != 0:
            issues.append("undefined_calibration_with_predictions")
    if not (score_dir / "per_parent_pair.parquet").exists():
        issues.append("missing_per_parent_pair")
    instrument_path = score_dir / "per_instrument_slice.parquet"
    if not instrument_path.exists():
        issues.append("missing_per_instrument_slice")
    else:
        instrument = pd.read_parquet(instrument_path)
        labels = set(instrument.get("instrument_slice", pd.Series(dtype=str)).astype(str))
        if labels != {"instrument_id", "instrument_ood"}:
            issues.append("invalid_instrument_slices")
        if not _finite_column(instrument, "component_f1"):
            issues.append("nonfinite_instrument_component_f1")
    if task == "window":
        difficulty_path = score_dir / "per_difficulty.parquet"
        if not difficulty_path.exists():
            issues.append("missing_per_difficulty")
        else:
            difficulty = pd.read_parquet(difficulty_path)
            labels = set(difficulty.get("difficulty", pd.Series(dtype=str)).astype(str))
            required = {"single", "resolved", "partial", "strong", "severe"}
            if labels != required:
                issues.append("invalid_overlap_difficulty_slices")
            if not _finite_column(difficulty, "component_f1"):
                issues.append("nonfinite_difficulty_component_f1")
    return issues


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return bool(len(values)) and all(math.isfinite(float(value)) for value in values)


def _comparison_artifact_issues(comparison_dir: Path) -> list[str]:
    issues: list[str] = []
    comparison = _read_json(comparison_dir / "comparison.json")
    if comparison is None:
        return ["missing_comparison"]
    if int(comparison.get("bootstrap_replicates", -1)) != 2000:
        issues.append("invalid_bootstrap_replicates")
    if int(comparison.get("bootstrap_seed", -1)) != 2027:
        issues.append("invalid_bootstrap_seed")
    if not (comparison_dir / "bootstrap_samples.parquet").exists():
        issues.append("missing_bootstrap_samples")
    if not (comparison_dir / "paired_parent_metrics.parquet").exists():
        issues.append("missing_paired_parent_metrics")
    return issues


def _training_artifact_issues(training_dir: Path) -> list[str]:
    issues = []
    if not (training_dir / "best_model.pt").exists():
        issues.append("missing_best_model")
    if not (training_dir / "training_summary.json").exists():
        issues.append("missing_training_summary")
    return issues


def assess_gamma_readiness(
    release_dir: str | Path,
    run_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    release = Path(release_dir).resolve()
    run = Path(run_root).resolve()
    missing: list[str] = []
    failures: list[str] = []
    invalid_scores: list[dict[str, Any]] = []
    invalid_comparisons: list[dict[str, Any]] = []
    invalid_training_artifacts: list[dict[str, Any]] = []

    def require_score(score_dir: Path, task: str) -> None:
        scorecard = score_dir / "scorecard.json"
        if not scorecard.exists():
            missing.append(str(scorecard))
            return
        issues = _score_artifact_issues(score_dir, task=task)
        if issues:
            invalid_scores.append({"score_dir": str(score_dir), "issues": issues})
    release_report = validate_public_release(release)
    if release_report["status"] != "pass":
        failures.append("RELEASE_AUDIT_FAILED")

    oracle_ok = True
    for task in TASKS:
        oracle_path = run / "oracle" / task / "validation_score" / "scorecard.json"
        oracle = _read_json(oracle_path)
        if oracle is None:
            missing.append(str(oracle_path))
            oracle_ok = False
            continue
        require_score(oracle_path.parent, task)
        metrics = oracle.get("metrics", {})
        maxima = ("component_f1", "phase_hkl_accuracy", "count_accuracy")
        minima = ("center_mae_deg", "fwhm_mae_deg", "area_mae", "forward_reconstruction_l1")
        if any(float(metrics.get(key, -1.0)) != 1.0 for key in maxima) or any(
            float(metrics.get(key, 1.0)) != 0.0 for key in minima
        ):
            oracle_ok = False
    if not oracle_ok:
        failures.append("ORACLE_NOT_OPTIMAL")

    for task in TASKS:
        for split in ("validation", "test"):
            require_score(run / "no-spectrum" / task / split / "score", task)
        for seed in SEEDS:
            for split in ("validation", "test"):
                require_score(run / "peak-feature" / task / str(seed) / split / "score", task)
            for model in ("resnet-set", "transformer-set", "physics-constrained"):
                require_score(run / model / task / str(seed) / "validation_score", task)
                require_score(run / model / task / str(seed) / "test_score", task)
            require_score(run / "reconstruction-only" / task / str(seed) / "validation_score", task)
            require_score(run / "reconstruction-only" / task / str(seed) / "test_score", task)
            for model_path in NEURAL_RUNS:
                training_dir = run / model_path / task / str(seed)
                training_issues = _training_artifact_issues(training_dir)
                if training_issues:
                    invalid_training_artifacts.append({
                        "training_dir": str(training_dir), "issues": training_issues,
                    })
    for split in ("validation", "test"):
        require_score(run / "bic" / "window" / split / "score", "window")

    positive_comparisons = []
    for task in TASKS:
        for model in LEARNED_MODELS:
            for seed in SEEDS:
                comparison_dir = run / "comparisons" / model / task / str(seed)
                comparison_issues = _comparison_artifact_issues(comparison_dir)
                if comparison_issues:
                    invalid_comparisons.append({
                        "comparison_dir": str(comparison_dir),
                        "issues": comparison_issues,
                    })
                    continue
                comparison = _read_json(comparison_dir / "comparison.json")
                if comparison and bool(comparison.get("five_point_gate_passed")):
                    positive_comparisons.append(f"{task}:{model}:{seed}")
    if not positive_comparisons:
        failures.append("NO_SIGNIFICANT_MODEL_IMPROVEMENT")

    for task in TASKS:
        for seed in SEEDS:
            for ablation in ("weak_peaks", "no_perturbation_training", "random_split"):
                for split_dir in ("validation_score", "test_score"):
                    require_score(
                        run / "ablations" / ablation / task / str(seed) / split_dir,
                        task,
                    )
    if missing:
        failures.append("INCOMPLETE_EXPERIMENT_MATRIX")
    if invalid_scores:
        failures.append("INVALID_SCORE_ARTIFACTS")
    if invalid_comparisons:
        failures.append("INVALID_COMPARISON_ARTIFACTS")
    if invalid_training_artifacts:
        failures.append("INVALID_TRAINING_ARTIFACTS")

    report = {
        "schema_version": "gamma-decomp-readiness-v1.1",
        "gamma_benchmark_ready": not failures,
        "release_audit": release_report,
        "oracle_optimal": oracle_ok,
        "positive_comparisons": sorted(positive_comparisons),
        "no_prior_ablation_status": "not_applicable_spectrum_only_input_contract",
        "missing_artifacts": sorted(set(missing)),
        "invalid_score_artifacts": invalid_scores,
        "invalid_comparison_artifacts": invalid_comparisons,
        "invalid_training_artifacts": invalid_training_artifacts,
        "failure_codes": sorted(set(failures)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = assess_gamma_readiness(args.release, args.run_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gamma_benchmark_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
