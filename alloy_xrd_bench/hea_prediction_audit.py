"""Audit the frozen HEA SRO model, split, seed, and prediction-ID matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SPLIT_COLUMNS = (
    "id_split",
    "composition_ood_split",
    "chemistry_ood_split",
    "sro_ood_split",
)
STOCHASTIC_MODELS = (
    "peak-xgboost",
    "resnet1d",
    "transformer",
    "physics-transformer",
)
SEEDS = (2027, 2028, 2029, 2030, 2031)
REQUIRED_COLUMNS = {
    "sample_id",
    "xrd_sample_id",
    "parent_group_id",
    "model",
    "seed",
    "split_column",
    "sro_predictions",
}


def _required_runs() -> list[tuple[str, int, str]]:
    runs = [("composition-ridge", 2027, split) for split in SPLIT_COLUMNS]
    runs.extend(
        (model, seed, split)
        for split in SPLIT_COLUMNS
        for model in STOCHASTIC_MODELS
        for seed in SEEDS
    )
    return runs


def audit_prediction_matrix(dataset: Path, predictions_root: Path) -> dict[str, Any]:
    tasks = pd.read_parquet(dataset / "manifests" / "task_samples.parquet")
    missing_columns = {
        "structure_state_id", "xrd_sample_id", "parent_group_id", *SPLIT_COLUMNS
    }.difference(tasks.columns)
    if missing_columns:
        raise ValueError(f"task manifest missing columns: {sorted(missing_columns)}")
    required_runs = _required_runs()
    expected_keys = set(required_runs)
    observed_keys: set[tuple[str, int, str]] = set()
    for path in predictions_root.glob("submissions/*/*/*_test/sro.parquet"):
        try:
            model = path.parents[2].name
            seed = int(path.parents[1].name)
            split = path.parent.name.removesuffix("_test")
        except ValueError:
            continue
        observed_keys.add((model, seed, split))

    missing_runs: list[dict[str, Any]] = []
    invalid_runs: list[dict[str, Any]] = []
    missing_ids: list[dict[str, Any]] = []
    unexpected_ids: list[dict[str, Any]] = []
    duplicate_ids: list[dict[str, Any]] = []
    for model, seed, split in required_runs:
        run = predictions_root / "submissions" / model / str(seed) / f"{split}_test"
        prediction_path = run / "sro.parquet"
        metadata_path = run / "metadata.json"
        identity = {"model": model, "seed": seed, "split_column": split}
        if not prediction_path.exists() or not metadata_path.exists():
            missing_runs.append(identity)
            continue
        frame = pd.read_parquet(prediction_path)
        absent = REQUIRED_COLUMNS.difference(frame.columns)
        if absent:
            invalid_runs.append({**identity, "reason": "missing_columns", "columns": sorted(absent)})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            str(metadata.get("model_id")) != model
            or int(metadata.get("seed", -1)) != seed
            or str(metadata.get("split_column")) != split
            or not frame["model"].astype(str).eq(model).all()
            or not frame["seed"].astype(int).eq(seed).all()
            or not frame["split_column"].astype(str).eq(split).all()
        ):
            invalid_runs.append({**identity, "reason": "metadata_mismatch"})
            continue
        duplicate = frame["sample_id"].astype(str).duplicated(keep=False)
        if duplicate.any():
            duplicate_ids.append({**identity, "ids": sorted(frame.loc[duplicate, "sample_id"].astype(str).unique())})
        truth = tasks.loc[tasks[split].astype(str).eq("test")]
        truth_by_id = truth.set_index(truth["structure_state_id"].astype(str), drop=False)
        actual_ids = frame["sample_id"].astype(str)
        expected = set(truth_by_id.index)
        actual = set(actual_ids)
        if expected - actual:
            missing_ids.append({**identity, "ids": sorted(expected - actual)})
        if actual - expected:
            unexpected_ids.append({**identity, "ids": sorted(actual - expected)})
        aligned = frame.loc[actual_ids.isin(expected)].copy()
        if not aligned.empty:
            aligned_truth = truth_by_id.loc[aligned["sample_id"].astype(str)]
            if not np.array_equal(
                aligned["xrd_sample_id"].astype(str).to_numpy(),
                aligned_truth["xrd_sample_id"].astype(str).to_numpy(),
            ) or not np.array_equal(
                aligned["parent_group_id"].astype(str).to_numpy(),
                aligned_truth["parent_group_id"].astype(str).to_numpy(),
            ):
                invalid_runs.append({**identity, "reason": "manifest_identity_mismatch"})

    extra_runs = [
        {"model": model, "seed": seed, "split_column": split}
        for model, seed, split in sorted(observed_keys - expected_keys)
    ]
    missing_count = len(missing_runs) + sum(len(item["ids"]) for item in missing_ids)
    duplicate_count = sum(len(item["ids"]) for item in duplicate_ids)
    unexpected_count = len(extra_runs) + sum(len(item["ids"]) for item in unexpected_ids)
    status = "pass" if not (missing_count or duplicate_count or unexpected_count or invalid_runs) else "fail"
    return {
        "schema_version": "sitexrd-hea-prediction-completeness-v1.1",
        "status": status,
        "required_run_count": len(required_runs),
        "validated_run_count": len(required_runs) - len(missing_runs) - len(invalid_runs),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "unexpected_count": unexpected_count,
        "invalid_count": len(invalid_runs),
        "missing_runs": missing_runs,
        "invalid_runs": invalid_runs,
        "missing_prediction_ids": missing_ids,
        "duplicate_prediction_ids": duplicate_ids,
        "unexpected_prediction_ids": unexpected_ids,
        "extra_runs": extra_runs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_prediction_matrix(args.dataset.resolve(), args.predictions_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
