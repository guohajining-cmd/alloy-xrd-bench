"""Evaluate a frozen HEA SRO checkpoint on matched perturbed XRD views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .hea_validation_data import vectorize_sro
from .hea_validation_models import (
    PhysicsConstrainedTransformer,
    ResNet1D,
    SpectrumTransformer,
    peak_features,
)


def _perturbation_severity(value: Any, config: dict[str, Any]) -> float:
    if not isinstance(value, str) or not value:
        return float("nan")
    params = json.loads(value)
    direct = (
        "instrument_fwhm_deg",
        "microstrain_fraction",
        "background_fraction",
        "gaussian_noise_fraction",
    )
    inverse = ("crystallite_size_nm", "poisson_scale")
    values = []
    max_shift = max(abs(float(item)) for item in config["zero_shift_deg"])
    values.append(abs(float(params["zero_shift_deg"])) / max_shift if max_shift else 0.0)
    for name in direct:
        low, high = (float(item) for item in config[name])
        values.append((float(params[name]) - low) / (high - low) if high > low else 0.0)
    for name in inverse:
        low, high = (float(item) for item in config[name])
        values.append((high - float(params[name])) / (high - low) if high > low else 0.0)
    low, high = (float(item) for item in config["march_dollase_r"])
    maximum_texture = max(abs(low - 1.0), abs(high - 1.0))
    values.append(abs(float(params["march_dollase_r"]) - 1.0) / maximum_texture if maximum_texture else 0.0)
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _load_raw_spectra(
    dataset: Path, observations: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((len(observations), 4001), dtype=np.float32)
    common_axis: np.ndarray | None = None
    for relative_path, indices in observations.groupby("grid_shard_path", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        rows = observations.loc[positions]
        with h5py.File((dataset / str(relative_path)).resolve(), "r") as handle:
            grid_indices = rows["grid_row_index"].to_numpy(dtype=int)
            spectra = np.asarray(handle["intensity"])[grid_indices].astype(np.float32)
            ids = handle["xrd_sample_id"][grid_indices]
            axis = (
                np.asarray(handle["two_theta"], dtype=np.float32)
                if "two_theta" in handle
                else np.linspace(20.0, 100.0, 4001, dtype=np.float32)
            )
        if common_axis is None:
            common_axis = axis
        elif not np.allclose(common_axis, axis, rtol=0.0, atol=1e-6):
            raise ValueError("observation shards use inconsistent two-theta axes")
        actual = [value.decode() if isinstance(value, bytes) else str(value) for value in ids]
        if actual != rows["xrd_sample_id"].astype(str).tolist():
            raise ValueError(f"observation sample ID mismatch in {relative_path}")
        if spectra.shape != (len(rows), 4001) or not np.isfinite(spectra).all() or np.any(spectra < 0):
            raise ValueError(f"invalid observation spectra in {relative_path}")
        output[positions] = spectra
    if common_axis is None:
        raise ValueError("no observation spectra")
    return output, common_axis


def _load_spectra(dataset: Path, observations: pd.DataFrame) -> np.ndarray:
    raw, _ = _load_raw_spectra(dataset, observations)
    transformed = np.log1p(raw)
    scales = transformed.max(axis=1, keepdims=True)
    return np.divide(
        transformed, scales, out=np.zeros_like(transformed), where=scales > 0
    )


def _model_from_checkpoint(checkpoint: dict[str, Any], output_dim: int) -> torch.nn.Module:
    name = str(checkpoint.get("model"))
    if name == "resnet1d":
        model = ResNet1D(output_dim)
    elif name == "transformer":
        model = SpectrumTransformer(output_dim)
    elif name == "physics-transformer":
        model = PhysicsConstrainedTransformer(
            output_dim, peak_count=int(checkpoint.get("peak_count", 16))
        )
    else:
        raise ValueError(f"unsupported checkpoint model {name!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def validate_heldout_scope(tasks, merged, split_column, clean_metrics_path):
    """Fail closed for legacy inputs that cannot establish matched held-out scope."""
    if clean_metrics_path is not None:
        raise ValueError(
            "An aggregate clean metric does not establish matched states; use the "
            "clean-prediction checks in scripts/evaluate_sro_corrected_perturbations.py"
        )
    if split_column not in tasks:
        raise ValueError(f"Cannot verify held-out scope without split column {split_column}")
    if (tasks.groupby("parent_group_id")[split_column].nunique(dropna=False) != 1).any():
        raise ValueError("Parent groups cross split boundaries")
    truth_split = split_column + "_truth" if split_column + "_truth" in merged else split_column
    if truth_split not in merged or not merged[truth_split].eq("test").all():
        raise ValueError("Legacy evaluator requires test-only observations; use the matched corrected evaluator")
    if truth_split != split_column and not merged[truth_split].eq(merged[split_column]).all():
        raise ValueError("Observation split disagrees with source manifest")


def evaluate_perturbed_checkpoint(
    dataset: Path,
    observation_run: str,
    checkpoint_path: Path,
    config_path: Path,
    output: Path,
    *,
    observation_dataset: Path | None = None,
    perturbation_config_path: Path | None = None,
    clean_metrics_path: Path | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    observation_dataset = (observation_dataset or dataset).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("sro_label_revision"):
        raise ValueError("Use scripts/evaluate_sro_corrected_perturbations.py for source-bound revised labels")
    checkpoint = (joblib.load(checkpoint_path) if checkpoint_path.suffix.lower() == ".joblib"
                  else torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    split_column = str(checkpoint.get("split_column", "id_split"))
    perturbation_config_path = perturbation_config_path or Path("configs/xrd_perturbation_v1.json")
    perturbation_config = json.loads(perturbation_config_path.read_text(encoding="utf-8"))
    observations = pd.read_parquet(
        observation_dataset / "xrd" / "runs" / observation_run / "xrd_samples.parquet"
    ).reset_index(drop=True)
    tasks = pd.read_parquet(dataset / "manifests" / "task_samples.parquet")
    truth_columns = [
        "structure_state_id",
        "parent_group_id",
        "sro_shell1_json",
        "sro_shell2_json",
    ]
    if split_column not in tasks:
        raise ValueError(f"Cannot verify held-out scope without split column {split_column}")
    truth_columns.append(split_column)
    if tasks["structure_state_id"].astype(str).duplicated().any():
        raise ValueError("duplicate structure states in task manifest")
    merged = observations.merge(
        tasks[truth_columns],
        on="structure_state_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_truth"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("observation states missing SRO truth")
    if "parent_group_id_truth" in merged and not merged["parent_group_id"].astype(str).eq(
        merged["parent_group_id_truth"].astype(str)
    ).all():
        raise ValueError("observation parent group does not match task truth")
    validate_heldout_scope(tasks, merged, split_column, clean_metrics_path)
    if checkpoint_path.suffix.lower() == ".joblib":
        if checkpoint.get("model") != "peak-xgboost":
            raise ValueError("unsupported classical checkpoint model")
        raw_spectra, axis = _load_raw_spectra(observation_dataset, merged)
        features = np.stack(
            [
                peak_features(row, axis, max_peaks=int(checkpoint["max_peaks"]))
                for row in raw_spectra
            ]
        )
        prediction = checkpoint["estimator"].predict(features)
    else:
        spectra = _load_spectra(observation_dataset, merged)
        model = _model_from_checkpoint(checkpoint, len(config["sro_targets"]))
        predictions = []
        with torch.no_grad():
            for batch in DataLoader(torch.from_numpy(spectra), batch_size=batch_size, shuffle=False):
                output_value = model(batch.unsqueeze(1))
                if isinstance(output_value, dict):
                    output_value = output_value["sro"]
                predictions.append(output_value.numpy())
        prediction = np.concatenate(predictions)
    truth_and_masks = [
        vectorize_sro(row.sro_shell1_json, row.sro_shell2_json, config["sro_targets"])
        for row in merged.itertuples()
    ]
    truth = np.stack([item[0] for item in truth_and_masks])
    mask = np.stack([item[1] for item in truth_and_masks])
    if not mask.any(axis=1).all():
        raise ValueError("each perturbed observation must have at least one SRO target")
    observation_mae = np.asarray(
        [np.abs(truth[i, mask[i]] - prediction[i, mask[i]]).mean() for i in range(len(merged))]
    )
    if "perturbation_json" in merged:
        severity = np.asarray(
            [_perturbation_severity(value, perturbation_config) for value in merged["perturbation_json"]],
            dtype=float,
        )
    else:
        severity = np.full(len(merged), np.nan)
    scored = pd.DataFrame(
        {
            "xrd_sample_id": merged["xrd_sample_id"].astype(str),
            "structure_state_id": merged["structure_state_id"].astype(str),
            "parent_group_id": merged["parent_group_id"].astype(str),
            "observation_mae": observation_mae,
            "perturbation_severity": severity,
            "sro_predictions_json": [json.dumps(values.tolist()) for values in prediction],
        }
    )
    state_errors = scored.groupby(["parent_group_id", "structure_state_id"])["observation_mae"].mean()
    parent_errors = state_errors.groupby(level="parent_group_id").mean()
    perturbed_mae = float(parent_errors.mean())
    severity_curve = []
    finite_severity = scored["perturbation_severity"].notna()
    if finite_severity.any():
        bins = [0.0, 0.25, 0.5, 0.75, 1.0000001]
        scored_severity = scored.loc[finite_severity].copy()
        scored_severity["severity_bin"] = pd.cut(
            scored_severity["perturbation_severity"], bins=bins, include_lowest=True, right=False
        )
        for interval, rows in scored_severity.groupby("severity_bin", observed=True):
            state = rows.groupby(["parent_group_id", "structure_state_id"])["observation_mae"].mean()
            parent = state.groupby(level="parent_group_id").mean()
            severity_curve.append(
                {
                    "severity_min": float(interval.left),
                    "severity_max": min(1.0, float(interval.right)),
                    "observation_count": len(rows),
                    "parent_group_count": int(rows["parent_group_id"].nunique()),
                    "parent_macro_mae": float(parent.mean()),
                }
            )
    report: dict[str, Any] = {
        "schema_version": "sitexrd-hea-perturbation-eval-v1",
        "status": "pass",
        "evaluation_scope": "held_out_test",
        "clean_reference_scope": None,
        "model": checkpoint["model"],
        "seed": int(checkpoint.get("seed", 0)),
        "split_column": str(checkpoint.get("split_column", "id_split")),
        "observation_run": observation_run,
        "source_dataset": str(dataset),
        "observation_dataset": str(observation_dataset),
        "missing_join_count": 0,
        "observation_count": len(scored),
        "structure_state_count": int(scored["structure_state_id"].nunique()),
        "parent_group_count": int(scored["parent_group_id"].nunique()),
        "perturbed_parent_macro_mae": perturbed_mae,
        "aggregation": "observation_to_structure_state_to_parent",
        "severity_definition": "mean_normalized_frozen_perturbation_magnitude",
        "severity_curve": severity_curve,
    }
    output.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output / "perturbation_predictions.parquet", index=False)
    (output / "perturbation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def aggregate_perturbation_reports(
    root: Path,
    *,
    minimum_seeds: int = 3,
    required_models: Sequence[str] | None = None,
    composition_ridge_metrics: Path | None = None,
) -> dict[str, Any]:
    paths = sorted(root.rglob("perturbation_summary.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    valid = [report for report in reports if report.get("status") == "pass"]
    if composition_ridge_metrics is not None:
        return {"status": "fail", "reason": "unmatched_composition_reference",
                "detail": "Full-test Ridge metrics cannot define a matched perturbation reference"}
    if any(report.get("evaluation_scope") != "held_out_test"
           or report.get("clean_reference_scope") != "identical_structure_states"
           or "clean_to_perturbed_retention" not in report for report in valid):
        return {"status": "fail", "reason": "unverified_matched_evaluation_scope",
                "detail": "Legacy mixed-split or unpaired summaries must be re-evaluated"}
    seeds = sorted({int(report["seed"]) for report in valid})
    expected_seeds = list(range(2027, 2027 + minimum_seeds))
    missing_by_model: dict[str, list[int]] = {}
    metrics_by_model: dict[str, Any] = {}
    for model in required_models or []:
        model_reports = [report for report in valid if report.get("model") == model]
        model_seeds = {int(report["seed"]) for report in model_reports}
        missing_by_model[model] = [seed for seed in expected_seeds if seed not in model_seeds]
        if model_reports:
            model_mae = np.asarray(
                [report["perturbed_parent_macro_mae"] for report in model_reports], dtype=float
            )
            model_retention = np.asarray(
                [report["clean_to_perturbed_retention"] for report in model_reports], dtype=float
            )
            metrics_by_model[model] = {
                "seed_count": len(model_seeds),
                "perturbed_parent_macro_mae_mean": float(model_mae.mean()),
                "perturbed_parent_macro_mae_std": float(model_mae.std()),
                "clean_to_perturbed_retention_mean": float(model_retention.mean()),
                "clean_to_perturbed_retention_std": float(model_retention.std()),
            }
    required_incomplete = any(missing_by_model.values())
    if len(valid) != len(paths) or required_incomplete or (not required_models and len(seeds) < minimum_seeds):
        return {
            "status": "fail",
            "reason": "insufficient_valid_perturbation_seeds",
            "seed_count": min(
                (value["seed_count"] for value in metrics_by_model.values()), default=len(seeds)
            ),
            "minimum_seeds": minimum_seeds,
            "missing_seeds_by_model": missing_by_model,
        }
    mae = np.asarray([report["perturbed_parent_macro_mae"] for report in valid], dtype=float)
    retention = np.asarray(
        [report["clean_to_perturbed_retention"] for report in valid], dtype=float
    )
    if not np.isfinite(mae).all() or not np.isfinite(retention).all():
        raise ValueError("perturbation reports contain non-finite metrics")
    return {
        "schema_version": "sitexrd-hea-perturbation-summary-v1",
        "status": "pass",
        "seed_count": min(
            (value["seed_count"] for value in metrics_by_model.values()), default=len(seeds)
        ),
        "seeds": seeds,
        "perturbed_parent_macro_mae_mean": float(mae.mean()),
        "perturbed_parent_macro_mae_std": float(mae.std()),
        "clean_to_perturbed_retention_mean": float(retention.mean()),
        "clean_to_perturbed_retention_std": float(retention.std()),
        "metrics_by_model": metrics_by_model,
        "missing_seeds_by_model": missing_by_model,
        "composition_ridge_negative_control": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--observation-dataset", type=Path)
    evaluate.add_argument("--observation-run", required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=Path("configs/hea_validation_v1.json"))
    evaluate.add_argument("--perturbation-config", type=Path, default=Path("configs/xrd_perturbation_v1.json"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--clean-metrics", type=Path)
    evaluate.add_argument("--batch-size", type=int, default=128)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--minimum-seeds", type=int, default=3)
    summarize.add_argument("--required-models", nargs="*")
    summarize.add_argument("--composition-ridge-metrics", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        report = evaluate_perturbed_checkpoint(
            args.dataset,
            args.observation_run,
            args.checkpoint,
            args.config,
            args.output,
            observation_dataset=args.observation_dataset,
            perturbation_config_path=args.perturbation_config,
            clean_metrics_path=args.clean_metrics,
            batch_size=args.batch_size,
        )
    else:
        report = aggregate_perturbation_reports(
            args.input.resolve(),
            minimum_seeds=args.minimum_seeds,
            required_models=args.required_models,
            composition_ridge_metrics=args.composition_ridge_metrics,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
