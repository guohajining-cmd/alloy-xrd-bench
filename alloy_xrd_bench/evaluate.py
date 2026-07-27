"""Independent evaluator for SiteXRD-Bench HEA prediction submissions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .hea_validation_data import vectorize_sro
from .hea_validation_metrics import (
    family_classification_metrics,
    masked_parent_macro_sro,
    occupancy_ranking_metrics,
)
from .scoring import normalized_sro_score, occupancy_score, require_complete_predictions


PREDICTION_SCHEMA_VERSION = "sitexrd-predictions-v1"
TASK_FILES = {
    "family": "family.parquet",
    "sro": "sro.parquet",
    "occupancy": "occupancy.parquet",
    "uncertainty": "uncertainty.parquet",
}
COMMON_COLUMNS = {"sample_id", "parent_group_id", "model_id", "seed"}


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("calibration labels and probabilities must be aligned vectors")
    if bins <= 0 or not np.isfinite(probabilities).all():
        raise ValueError("calibration probabilities must be finite and bins positive")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("calibration probabilities must be in [0, 1]")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("calibration labels must be binary")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    value = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            value += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean())
            )
    return value


def selective_risk_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    expected_calibration_error(labels, probabilities)
    predicted = probabilities >= 0.5
    correct = predicted == labels.astype(bool)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(~correct[order]) / np.arange(1, len(labels) + 1)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    aurc = float(np.trapezoid(cumulative_risk, coverage))
    error_rate = float((~correct).mean())
    oracle_order = np.argsort(~correct, kind="stable")
    oracle_risk = np.cumsum(~correct[oracle_order]) / np.arange(1, len(labels) + 1)
    oracle_aurc = float(np.trapezoid(oracle_risk, coverage))
    auroc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    return {
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": float(correct.mean()),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "reference_aurc": error_rate,
        "identifiability_auroc": auroc,
        "risk_curve": cumulative_risk.tolist(),
        "coverage_curve": coverage.tolist(),
    }


def paired_parent_bootstrap(
    reference_errors: Mapping[str, float],
    candidate_errors: Mapping[str, float],
    *,
    replicates: int = 2000,
    seed: int = 2027,
) -> dict[str, float | int]:
    if set(reference_errors) != set(candidate_errors) or not reference_errors:
        raise ValueError("paired bootstrap requires identical non-empty parent IDs")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    parent_ids = sorted(reference_errors)
    reference = np.asarray([reference_errors[key] for key in parent_ids], dtype=float)
    candidate = np.asarray([candidate_errors[key] for key in parent_ids], dtype=float)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all() or np.any(reference < 0) or np.any(candidate < 0):
        raise ValueError("bootstrap errors must be finite and non-negative")
    reference_mean = float(reference.mean())
    if reference_mean <= 0.0:
        raise ValueError("reference mean error must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    difference_draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(parent_ids), len(parent_ids))
        ref = float(reference[selected].mean())
        cand = float(candidate[selected].mean())
        draws[index] = 100.0 * (1.0 - cand / ref) if ref > 0 else 0.0
        difference_draws[index] = ref - cand
    point = 100.0 * (1.0 - float(candidate.mean()) / reference_mean)
    candidate_mean = float(candidate.mean())
    return {
        "parent_count": len(parent_ids),
        "replicates": replicates,
        "seed": seed,
        "reference_mae": reference_mean,
        "candidate_mae": candidate_mean,
        "mae_difference": reference_mean - candidate_mean,
        "mae_difference_ci95_low": float(np.quantile(difference_draws, 0.025)),
        "mae_difference_ci95_high": float(np.quantile(difference_draws, 0.975)),
        "skill_points": point,
        "relative_ridge_improvement_percent": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_of_improvement": float((draws > 0.0).mean()),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _truth(dataset: Path, metadata: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(dataset / "manifests" / "task_samples.parquet")
    split_column = str(metadata["split_column"])
    split_value = str(metadata["split_value"])
    if split_column not in samples:
        raise ValueError(f"truth manifest missing split column {split_column!r}")
    samples = samples.loc[samples[split_column].astype(str).eq(split_value)].copy()
    pairs = pd.DataFrame()
    needs_pairs = bool(set(metadata.get("tasks", [])) & {"occupancy", "uncertainty"})
    calibrated = dataset / "manifests" / "occupancy_pairs_calibrated.parquet"
    fallback = dataset / "manifests" / "occupancy_pairs.parquet"
    pair_path = calibrated if calibrated.exists() else fallback
    if needs_pairs and pair_path.exists():
        pairs = pd.read_parquet(pair_path)
    if not pairs.empty:
        if split_column not in pairs:
            if split_column != "id_split" or "id_split" not in pairs:
                raise ValueError(f"occupancy truth missing split column {split_column!r}")
        pairs = pairs.loc[pairs[split_column].astype(str).eq(split_value)].copy()
        if "noise_identifiable" not in pairs:
            if "low_identifiability" not in pairs:
                raise ValueError("occupancy truth lacks calibrated identifiability labels")
            pairs["noise_identifiable"] = ~pairs["low_identifiability"].astype(bool)
    return samples, pairs


def _metadata(predictions: Path) -> dict[str, Any]:
    metadata = _load_json(predictions / "metadata.json")
    required = {"schema_version", "model_id", "seed", "split_column", "split_value", "tasks"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"metadata missing fields: {sorted(missing)}")
    if metadata["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError("unsupported prediction schema version")
    tasks = list(metadata["tasks"])
    unknown = set(tasks).difference(TASK_FILES)
    if unknown or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid tasks: {tasks}")
    return metadata


def _read_task(predictions: Path, task: str, metadata: Mapping[str, Any]) -> pd.DataFrame:
    path = predictions / TASK_FILES[task]
    if not path.exists():
        raise ValueError(f"missing prediction file for task {task}: {path.name}")
    frame = pd.read_parquet(path)
    missing = COMMON_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{task} predictions missing columns: {sorted(missing)}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate prediction sample IDs in {task}")
    if not frame["model_id"].astype(str).eq(str(metadata["model_id"])).all():
        raise ValueError(f"model_id mismatch in {task}")
    if not frame["seed"].astype(int).eq(int(metadata["seed"])).all():
        raise ValueError(f"seed mismatch in {task}")
    return frame


def _json_vector(value: Any, length: int, name: str) -> np.ndarray:
    raw = json.loads(value) if isinstance(value, str) else value
    result = np.asarray(raw, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return result


def validate_submission(
    dataset: Path, predictions: Path, config_path: Path
) -> dict[str, Any]:
    metadata = _metadata(predictions)
    config = _load_config(config_path)
    samples, pairs = _truth(dataset, metadata)
    counts: dict[str, int] = {}
    for task in metadata["tasks"]:
        frame = _read_task(predictions, task, metadata)
        if task in {"family", "sro"}:
            expected = samples["structure_state_id"].astype(str).tolist()
        else:
            if pairs.empty:
                raise ValueError(f"no occupancy truth available for task {task}")
            id_column = "candidate_structure_state_id" if task == "occupancy" else "pair_id"
            expected = pairs[id_column].astype(str).tolist()
        require_complete_predictions(expected, frame["sample_id"].astype(str).tolist())
        if task == "family":
            required = {"predicted_label", "probabilities_json"}
            if required.difference(frame):
                raise ValueError("family predictions missing label or probabilities")
            for value in frame["probabilities_json"]:
                probabilities = json.loads(value)
                numeric = np.asarray(list(probabilities.values()), dtype=float)
                if not numeric.size or not np.isfinite(numeric).all() or np.any(numeric < 0) or not math.isclose(float(numeric.sum()), 1.0, abs_tol=1e-6):
                    raise ValueError("family probabilities must be finite, non-negative, and sum to one")
        elif task == "sro":
            required = {"sro_predictions_json"}
            if required.difference(frame):
                raise ValueError("sro predictions missing sro_predictions_json")
            for value in frame["sro_predictions_json"]:
                _json_vector(value, len(config["sro_targets"]), "sro_predictions_json")
            has_lower = "sro_lower_json" in frame
            has_upper = "sro_upper_json" in frame
            if has_lower != has_upper:
                raise ValueError("SRO interval lower and upper must be supplied together")
        elif task == "occupancy":
            if "predicted_score" not in frame or not np.isfinite(frame["predicted_score"].astype(float)).all():
                raise ValueError("occupancy predicted_score must be finite")
        elif task == "uncertainty":
            if {"identifiable_probability", "abstain"}.difference(frame):
                raise ValueError("uncertainty predictions missing probability or abstain")
            probabilities = frame["identifiable_probability"].astype(float).to_numpy()
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("identifiable_probability must be finite and in [0, 1]")
        counts[task] = len(frame)
    return {"status": "pass", "model_id": metadata["model_id"], "seed": metadata["seed"], "split_column": metadata["split_column"], "split_value": metadata["split_value"], "task_counts": counts}


def _aligned(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    return frame.assign(sample_id=frame["sample_id"].astype(str)).set_index("sample_id").loc[id_column].reset_index()


def _parent_errors(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray, parents: Sequence[str]
) -> dict[str, float]:
    result: dict[str, list[float]] = {}
    for index, parent in enumerate(parents):
        if mask[index].any():
            result.setdefault(str(parent), []).append(float(np.abs(truth[index][mask[index]] - prediction[index][mask[index]]).mean()))
    return {parent: float(np.mean(values)) for parent, values in result.items()}


def score_submission(
    dataset: Path,
    predictions: Path,
    output: Path,
    config_path: Path,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 2027,
    baseline_mae: float | None = None,
) -> dict[str, Any]:
    validation = validate_submission(dataset, predictions, config_path)
    metadata = _metadata(predictions)
    config = _load_config(config_path)
    samples, pairs = _truth(dataset, metadata)
    metrics: dict[str, Any] = {"schema_version": "sitexrd-metrics-v1", "validation": validation}
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    if "family" in metadata["tasks"]:
        frame = _aligned(_read_task(predictions, "family", metadata), samples["structure_state_id"].astype(str).tolist())
        mapping = config["family_mapping"]
        truth_labels = [str(mapping[value]) for value in samples["structure_family"]]
        classes = sorted(set(truth_labels))
        predicted = frame["predicted_label"].astype(str).tolist()
        result = family_classification_metrics(truth_labels, predicted, classes)
        labels = list(zip(truth_labels, predicted, samples["parent_group_id"].astype(str)))
        rng = np.random.default_rng(bootstrap_seed)
        parents = sorted(set(samples["parent_group_id"].astype(str)))
        for draw in range(bootstrap_replicates):
            selected = rng.choice(parents, size=len(parents), replace=True)
            sampled = [item for parent in selected for item in labels if item[2] == parent]
            value = family_classification_metrics([item[0] for item in sampled], [item[1] for item in sampled], classes)["macro_f1"]
            bootstrap_rows.append({"task": "family", "draw": draw, "metric": "macro_f1", "value": value})
        metrics["family"] = result
        rows.extend({"task": "family_class", "group_id": label, "metric": key, "value": value} for label, values in result["per_class"].items() for key, value in values.items())

    if "sro" in metadata["tasks"]:
        frame = _aligned(_read_task(predictions, "sro", metadata), samples["structure_state_id"].astype(str).tolist())
        truth_values, masks = zip(*(vectorize_sro(row["sro_shell1_json"], row["sro_shell2_json"], config["sro_targets"]) for row in samples.to_dict(orient="records")))
        truth = np.stack(truth_values)
        mask = np.stack(masks)
        prediction = np.stack([_json_vector(value, len(config["sro_targets"]), "sro_predictions_json") for value in frame["sro_predictions_json"]])
        parents = samples["parent_group_id"].astype(str).tolist()
        shells = [int(target[5]) for target in config["sro_targets"]]
        result = masked_parent_macro_sro(truth, prediction, mask, parents, shells)
        if float(result["masked_parent_macro_mae"]) < 1e-8:
            result["masked_parent_macro_mae"] = 0.0
        if float(result["rmse"]) < 1e-8:
            result["rmse"] = 0.0
        parent_errors = _parent_errors(truth, prediction, mask, parents)
        rows.extend({"task": "sro_parent", "group_id": parent, "metric": "mae", "value": value} for parent, value in parent_errors.items())
        rng = np.random.default_rng(bootstrap_seed)
        values = np.asarray(list(parent_errors.values()), dtype=float)
        for draw in range(bootstrap_replicates):
            bootstrap_rows.append({"task": "sro", "draw": draw, "metric": "parent_macro_mae", "value": float(rng.choice(values, size=len(values), replace=True).mean())})
        if baseline_mae is not None:
            result["normalized_score"] = normalized_sro_score(float(result["masked_parent_macro_mae"]), baseline_mae)
        if "sro_lower_json" in frame and "sro_upper_json" in frame:
            lower = np.stack([_json_vector(value, len(config["sro_targets"]), "sro_lower_json") for value in frame["sro_lower_json"]])
            upper = np.stack([_json_vector(value, len(config["sro_targets"]), "sro_upper_json") for value in frame["sro_upper_json"]])
            if np.any(lower[mask] > upper[mask]):
                raise ValueError("SRO interval lower bound exceeds upper bound")
            result["interval_coverage"] = float(((truth >= lower) & (truth <= upper))[mask].mean())
            result["mean_interval_width"] = float((upper - lower)[mask].mean())
        metrics["sro"] = result

    if "occupancy" in metadata["tasks"]:
        frame = _aligned(_read_task(predictions, "occupancy", metadata), pairs["candidate_structure_state_id"].astype(str).tolist())
        eligible = pairs["noise_identifiable"].astype(bool).to_numpy()
        result = occupancy_ranking_metrics(pairs["sro_distance"].astype(float).to_numpy(), frame["predicted_score"].astype(float).to_numpy(), pairs["parent_group_id"].astype(str).tolist(), identifiable=eligible)
        if all(result[key] is not None for key in ["pair_accuracy", "mrr", "top3_accuracy"]):
            result["normalized_score"] = occupancy_score(result["pair_accuracy"], result["mrr"], result["top3_accuracy"])
        metrics["occupancy"] = result

    if "uncertainty" in metadata["tasks"]:
        frame = _aligned(_read_task(predictions, "uncertainty", metadata), pairs["pair_id"].astype(str).tolist())
        labels = pairs["noise_identifiable"].astype(int).to_numpy()
        probabilities = frame["identifiable_probability"].astype(float).to_numpy()
        result = selective_risk_metrics(labels, probabilities)
        abstain = frame["abstain"].astype(bool).to_numpy()
        result["non_abstained_coverage"] = float((~abstain).mean())
        metrics["uncertainty"] = result

    flags: list[str] = []
    if metrics.get("family", {}).get("macro_f1", 1.0) < 0.90:
        flags.append("basic_diffraction_fail")
    if metrics.get("occupancy", {}).get("eligible_item_count", 1) == 0:
        flags.append("occupancy_not_identifiable")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows, columns=["task", "group_id", "metric", "value"]).to_parquet(output / "metrics_by_slice.parquet", index=False)
    pd.DataFrame(bootstrap_rows, columns=["task", "draw", "metric", "value"]).to_parquet(output / "bootstrap_samples.parquet", index=False)
    (output / "diagnostic_flags.json").write_text(json.dumps({"flags": flags}, indent=2), encoding="utf-8")
    scorecard = ["# SiteXRD-Bench HEA Scorecard", "", f"- Model: `{metadata['model_id']}`", f"- Seed: `{metadata['seed']}`", f"- Slice: `{metadata['split_column']}={metadata['split_value']}`", f"- Submission validation: **{validation['status'].upper()}**", f"- Diagnostic flags: {', '.join(flags) if flags else 'none'}", ""]
    if "sro" in metrics:
        scorecard.append(f"- SRO parent-macro MAE: {metrics['sro']['masked_parent_macro_mae']:.6f}")
    if "family" in metrics:
        scorecard.append(f"- Family Macro-F1: {metrics['family']['macro_f1']:.6f}")
    if "occupancy" in metrics:
        scorecard.append(f"- Identifiable occupancy candidates: {metrics['occupancy']['eligible_item_count']}")
    (output / "scorecard.md").write_text("\n".join(scorecard) + "\n", encoding="utf-8")
    return {"status": "pass", "output": str(output.resolve()), "flags": flags}


def compare_scores(reference: Path, candidate: Path, *, replicates: int, seed: int) -> dict[str, Any]:
    def errors(path: Path) -> tuple[dict[str, float], int]:
        direct = path / "metrics_by_slice.parquet"
        paths = [direct] if direct.exists() else sorted(path.rglob("metrics_by_slice.parquet"))
        if not paths:
            raise ValueError(f"no score artifacts found under {path}")
        runs = []
        for score_path in paths:
            frame = pd.read_parquet(score_path)
            selected = frame.loc[
                frame["task"].eq("sro_parent") & frame["metric"].eq("mae")
            ].copy()
            if selected.empty or selected["group_id"].astype(str).duplicated().any():
                raise ValueError(f"invalid parent metrics in {score_path}")
            selected["run"] = score_path.parent.as_posix()
            runs.append(selected[["group_id", "value", "run"]])
        combined = pd.concat(runs, ignore_index=True)
        if not np.isfinite(combined["value"].astype(float)).all():
            raise ValueError("non-finite parent errors")
        aggregated = combined.groupby(combined["group_id"].astype(str))["value"].mean()
        return aggregated.astype(float).to_dict(), len(paths)

    reference_errors, reference_count = errors(reference)
    candidate_errors, candidate_count = errors(candidate)
    result = paired_parent_bootstrap(
        reference_errors, candidate_errors, replicates=replicates, seed=seed
    )
    result["status"] = "pass"
    result["reference_run_count"] = reference_count
    result["candidate_run_count"] = candidate_count
    result["passes_five_point_gate"] = bool(result["skill_points"] >= 5.0 and result["ci95_low"] > 0.0)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["validate", "score"]:
        item = subparsers.add_parser(command)
        item.add_argument("--dataset", type=Path, required=True)
        item.add_argument("--predictions", type=Path, required=True)
        item.add_argument("--config", type=Path, default=Path("configs/hea_validation_v1.json"))
        if command == "score":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--bootstrap-replicates", type=int, default=2000)
            item.add_argument("--bootstrap-seed", type=int, default=2027)
            item.add_argument("--baseline-mae", type=float)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--bootstrap-replicates", type=int, default=2000)
    compare.add_argument("--bootstrap-seed", type=int, default=2027)
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_submission(args.dataset.resolve(), args.predictions.resolve(), args.config.resolve())
        elif args.command == "score":
            result = score_submission(args.dataset.resolve(), args.predictions.resolve(), args.output.resolve(), args.config.resolve(), bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed, baseline_mae=args.baseline_mae)
        else:
            result = compare_scores(args.reference.resolve(), args.candidate.resolve(), replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
