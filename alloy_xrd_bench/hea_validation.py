"""Command line runner for pre-release HEA model validation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader, TensorDataset

from .hea_validation_data import audit_parent_group_leakage, load_validation_rows
from .hea_validation_metrics import (
    family_classification_metrics,
    masked_parent_macro_sro,
    prediction_completeness,
)
from .hea_validation_models import (
    PerTargetExtraTrees,
    PerTargetRidge,
    PerTargetXGBoost,
    PhysicsConstrainedTransformer,
    ResNet1D,
    SpectrumTransformer,
    peak_feature_matrix,
    seed_everything,
    spectrum_matrix,
    masked_mse,
    normalize_reconstruction_target,
)


PREDICTION_SCHEMA_VERSION = "sitexrd-predictions-v1"


def _load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_audit(dataset: Path, config_path: Path) -> tuple[dict[str, object], bool]:
    config = _load_config(config_path)
    rows = load_validation_rows(dataset, config)
    leakage_by_split = {
        split: audit_parent_group_leakage(rows, split)
        for split in config["split_columns"]
    }
    leaking_groups = sorted(
        {group for groups in leakage_by_split.values() for group in groups}
    )
    mask = np.stack([row.sro_mask for row in rows])
    structures_path = dataset / "manifests" / "structures.parquet"
    canonical_count = (
        len(pd.read_parquet(structures_path, columns=["dataset_structure_id"]))
        if structures_path.exists()
        else len({row.parent_group_id for row in rows})
    )
    payload: dict[str, object] = {
        "status": "pass" if not leaking_groups else "fail",
        "schema_version": config["schema_version"],
        "result_qualification": config["result_qualification"],
        "benchmark_ready": bool(config["benchmark_ready"]),
        "row_count": len(rows),
        "canonical_count": canonical_count,
        "source_dataset_immutable": True,
        "parent_group_count": len({row.parent_group_id for row in rows}),
        "family_counts": dict(sorted(Counter(row.family for row in rows).items())),
        "sro_valid_value_count": int(mask.sum()),
        "sro_target_coverage": {
            target: float(mask[:, index].mean())
            for index, target in enumerate(config["sro_targets"])
        },
        "parent_group_leakage_count": len(leaking_groups),
        "parent_group_leakage": leakage_by_split,
    }
    return payload, not leaking_groups


def run_controls(
    dataset: Path, config_path: Path, *, split: str, seed: int
) -> tuple[dict[str, object], bool]:
    config = _load_config(config_path)
    rows = load_validation_rows(dataset, config)
    if split != "all":
        rows = [row for row in rows if row.splits["id_split"] == split]
    if not rows:
        raise ValueError(f"no rows selected for split {split!r}")

    truth = np.stack([row.sro_values for row in rows])
    mask = np.stack([row.sro_mask for row in rows])
    parents = [row.parent_group_id for row in rows]
    shells = [int(target[5]) for target in config["sro_targets"]]
    oracle_sro = masked_parent_macro_sro(truth, truth, mask, parents, shells)
    rng = np.random.default_rng(seed)
    shuffled = truth.copy()
    for column in range(truth.shape[1]):
        indices = np.flatnonzero(mask[:, column])
        shuffled[indices, column] = truth[rng.permutation(indices), column]
    shuffled_sro = masked_parent_macro_sro(truth, shuffled, mask, parents, shells)

    classes = sorted(set(config["family_mapping"].values()))
    families = [row.family for row in rows]
    shuffled_families = list(np.asarray(families)[rng.permutation(len(families))])
    family_oracle = family_classification_metrics(families, families, classes)
    family_shuffled = family_classification_metrics(
        families, shuffled_families, classes
    )
    payload: dict[str, object] = {
        "result_qualification": config["result_qualification"],
        "split": split,
        "seed": seed,
        "sample_count": len(rows),
        "sro": {"oracle": oracle_sro, "shuffled": shuffled_sro},
        "family": {"oracle": family_oracle, "shuffled": family_shuffled},
    }

    passed = (
        oracle_sro["masked_parent_macro_mae"] == 0.0
        and shuffled_sro["masked_parent_macro_mae"] > 0.0
        and all(truth == prediction for truth, prediction in zip(families, families))
    )
    payload["control_gate_passed"] = passed
    return payload, passed


def _fit_family_classifier(
    train_features: np.ndarray,
    train_labels: list[str],
    test_features: np.ndarray,
    classes: list[str],
    seed: int,
) -> tuple[list[str], np.ndarray]:
    observed = sorted(set(train_labels))
    probabilities = np.zeros((len(test_features), len(classes)), dtype=float)
    if len(observed) == 1:
        probabilities[:, classes.index(observed[0])] = 1.0
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
        classifier.fit(train_features, train_labels)
        observed_probabilities = classifier.predict_proba(test_features)
        for index, label in enumerate(classifier.classes_):
            probabilities[:, classes.index(str(label))] = observed_probabilities[:, index]
    predictions = [classes[index] for index in np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def _write_formal_submission(
    output: Path,
    *,
    model_name: str,
    seed: int,
    split_column: str,
    rows: Sequence[object],
    sro_prediction: np.ndarray,
    family_prediction: Sequence[str] | None = None,
    family_probabilities: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> Path:
    submission = output / "submissions" / model_name / str(seed) / f"{split_column}_test"
    submission.mkdir(parents=True, exist_ok=True)
    tasks = ["sro"]
    common = {
        "sample_id": [row.structure_state_id for row in rows],
        "xrd_sample_id": [row.xrd_sample_id for row in rows],
        "parent_group_id": [row.parent_group_id for row in rows],
        "model": model_name,
        "model_id": model_name,
        "seed": seed,
        "split_column": split_column,
    }
    pd.DataFrame(
        {
            **common,
            "sro_predictions": [values.tolist() for values in sro_prediction],
            "sro_predictions_json": [json.dumps(values.tolist()) for values in sro_prediction],
        }
    ).to_parquet(submission / "sro.parquet", index=False)
    if family_prediction is not None and family_probabilities is not None and classes is not None:
        pd.DataFrame(
            {
                **common,
                "predicted_label": list(family_prediction),
                "probabilities_json": [
                    json.dumps(dict(zip(classes, values))) for values in family_probabilities
                ],
            }
        ).to_parquet(submission / "family.parquet", index=False)
    (submission / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "model_id": model_name,
                "seed": seed,
                "split_column": split_column,
                "split_value": "test",
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return submission.resolve()


def materialize_formal_submission(
    dataset: Path,
    prediction_path: Path,
    output: Path,
    *,
    model_name: str,
    seed: int,
    split_column: str,
) -> Path:
    """Recover a formal submission from a completed, identity-checked prediction artifact."""
    manifest = pd.read_parquet(dataset / "manifests" / "task_samples.parquet")
    required_manifest = {"structure_state_id", "xrd_sample_id", "parent_group_id", split_column}
    missing_manifest = required_manifest.difference(manifest.columns)
    if missing_manifest:
        raise ValueError(f"task manifest missing columns: {sorted(missing_manifest)}")
    expected = manifest.loc[manifest[split_column].astype(str).eq("test")].copy()
    if expected["xrd_sample_id"].astype(str).duplicated().any():
        raise ValueError("test manifest contains duplicate xrd_sample_id values")

    predictions = pd.read_parquet(prediction_path)
    required_predictions = {
        "model", "seed", "split_column", "xrd_sample_id", "parent_group_id", "sro_predictions"
    }
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(f"prediction artifact missing columns: {sorted(missing_predictions)}")
    if predictions["xrd_sample_id"].astype(str).duplicated().any():
        raise ValueError("prediction artifact contains duplicate xrd_sample_id values")
    if not predictions["model"].astype(str).eq(model_name).all():
        raise ValueError("prediction artifact model does not match requested model")
    if not predictions["seed"].astype(int).eq(seed).all():
        raise ValueError("prediction artifact seed does not match requested seed")
    if not predictions["split_column"].astype(str).eq(split_column).all():
        raise ValueError("prediction artifact split does not match requested split")

    expected_ids = set(expected["xrd_sample_id"].astype(str))
    prediction_ids = set(predictions["xrd_sample_id"].astype(str))
    if prediction_ids != expected_ids:
        raise ValueError(
            f"prediction IDs do not match the test manifest: missing={len(expected_ids - prediction_ids)}, "
            f"extra={len(prediction_ids - expected_ids)}"
        )
    ordered = expected.merge(
        predictions[["xrd_sample_id", "parent_group_id", "sro_predictions"]],
        on="xrd_sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", "_prediction"),
    )
    if not ordered["parent_group_id_manifest"].astype(str).eq(
        ordered["parent_group_id_prediction"].astype(str)
    ).all():
        raise ValueError("prediction parent_group_id does not match the task manifest")
    ordered = ordered.rename(columns={"parent_group_id_manifest": "parent_group_id"})
    vectors = [np.asarray(value, dtype=np.float64) for value in ordered["sro_predictions"]]
    if not vectors or any(value.ndim != 1 or not np.isfinite(value).all() for value in vectors):
        raise ValueError("prediction artifact contains invalid SRO vectors")
    if len({len(value) for value in vectors}) != 1:
        raise ValueError("prediction artifact contains inconsistent SRO vector lengths")
    return _write_formal_submission(
        output,
        model_name=model_name,
        seed=seed,
        split_column=split_column,
        rows=list(ordered.itertuples(index=False)),
        sro_prediction=np.stack(vectors),
    )


def _train_composition_baseline(
    model_name: str,
    dataset: Path,
    config_path: Path,
    output: Path,
    *,
    seed: int,
    split_column: str,
) -> dict[str, object]:
    config = _load_config(config_path)
    rows = load_validation_rows(dataset, config)
    if split_column not in config["split_columns"]:
        raise ValueError(f"unsupported split column {split_column!r}")
    leaking = audit_parent_group_leakage(rows, split_column)
    if leaking:
        raise ValueError(f"parent-group leakage in {split_column}: {leaking[:5]}")
    train_rows = [row for row in rows if row.splits[split_column] == "train"]
    test_rows = [row for row in rows if row.splits[split_column] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("composition baseline requires non-empty ID train and test rows")
    seed_everything(seed)
    train_x = np.stack([row.composition for row in train_rows])
    test_x = np.stack([row.composition for row in test_rows])
    train_y = np.stack([row.sro_values for row in train_rows])
    train_mask = np.stack([row.sro_mask for row in train_rows])
    test_y = np.stack([row.sro_values for row in test_rows])
    test_mask = np.stack([row.sro_mask for row in test_rows])
    budget = config["model_budgets"][model_name.replace("-", "_")]
    if model_name == "composition-ridge":
        estimator = PerTargetRidge(alpha=float(budget["alpha"]))
    elif model_name == "composition-extratrees":
        estimator = PerTargetExtraTrees(
            n_estimators=int(budget["n_estimators"]),
            min_samples_leaf=int(budget["min_samples_leaf"]),
            random_state=seed,
            n_jobs=int(budget.get("n_jobs", -1)),
        )
    else:
        raise ValueError(f"unsupported composition model {model_name!r}")
    sro_prediction = estimator.fit(train_x, train_y, train_mask).predict(test_x)

    classes = sorted(set(config["family_mapping"].values()))
    family_prediction, family_probabilities = _fit_family_classifier(
        train_x,
        [row.family for row in train_rows],
        test_x,
        classes,
        seed,
    )
    sample_ids = [row.xrd_sample_id for row in test_rows]
    prediction_completeness(sample_ids, sample_ids)
    shells = [int(target[5]) for target in config["sro_targets"]]
    metrics = {
        "model": model_name,
        "seed": seed,
        "slice": f"{split_column}_test",
        "result_qualification": config["result_qualification"],
        "sro": masked_parent_macro_sro(
            test_y,
            sro_prediction,
            test_mask,
            [row.parent_group_id for row in test_rows],
            shells,
        ),
        "family": family_classification_metrics(
            [row.family for row in test_rows], family_prediction, classes
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.replace('-', '_')}_seed{seed}_{split_column}_test"
    prediction_path = (output / f"{stem}.parquet").resolve()
    metrics_path = (output / f"{stem}.metrics.json").resolve()
    pd.DataFrame(
        {
            "model": model_name,
            "seed": seed,
            "split_column": split_column,
            "slice": f"{split_column}_test",
            "xrd_sample_id": sample_ids,
            "parent_group_id": [row.parent_group_id for row in test_rows],
            "family_prediction": family_prediction,
            "family_probabilities_json": [
                json.dumps(dict(zip(classes, values))) for values in family_probabilities
            ],
            "sro_predictions": [values.tolist() for values in sro_prediction],
        }
    ).to_parquet(prediction_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    submission_path = _write_formal_submission(
        output,
        model_name=model_name,
        seed=seed,
        split_column=split_column,
        rows=test_rows,
        sro_prediction=sro_prediction,
        family_prediction=family_prediction,
        family_probabilities=family_probabilities,
        classes=classes,
    )
    return {
        "model": model_name,
        "seed": seed,
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def train_composition_ridge(
    dataset: Path,
    config_path: Path,
    output: Path,
    *,
    seed: int,
    split_column: str = "id_split",
) -> dict[str, object]:
    return _train_composition_baseline(
        "composition-ridge", dataset, config_path, output, seed=seed, split_column=split_column
    )


def train_composition_extratrees(
    dataset: Path,
    config_path: Path,
    output: Path,
    *,
    seed: int,
    split_column: str = "id_split",
) -> dict[str, object]:
    return _train_composition_baseline(
        "composition-extratrees", dataset, config_path, output, seed=seed, split_column=split_column
    )


def train_peak_xgboost(
    dataset: Path,
    config_path: Path,
    output: Path,
    *,
    seed: int,
    split_column: str = "id_split",
) -> dict[str, object]:
    config = _load_config(config_path)
    rows = load_validation_rows(dataset, config)
    if split_column not in config["split_columns"]:
        raise ValueError(f"unsupported split column {split_column!r}")
    leaking = audit_parent_group_leakage(rows, split_column)
    if leaking:
        raise ValueError(f"parent-group leakage in {split_column}: {leaking[:5]}")
    train_rows = [row for row in rows if row.splits[split_column] == "train"]
    test_rows = [row for row in rows if row.splits[split_column] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("peak baseline requires non-empty ID train and test rows")
    seed_everything(seed)
    budget = config["model_budgets"]["peak_xgboost"]
    max_peaks = int(budget["max_peaks"])
    train_x = peak_feature_matrix(train_rows, max_peaks=max_peaks)
    test_x = peak_feature_matrix(test_rows, max_peaks=max_peaks)
    train_y = np.stack([row.sro_values for row in train_rows])
    train_mask = np.stack([row.sro_mask for row in train_rows])
    test_y = np.stack([row.sro_values for row in test_rows])
    test_mask = np.stack([row.sro_mask for row in test_rows])
    xgboost = PerTargetXGBoost(
        n_estimators=int(budget["n_estimators"]),
        max_depth=int(budget["max_depth"]),
        learning_rate=float(budget["learning_rate"]),
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=seed,
        n_jobs=8,
    ).fit(train_x, train_y, train_mask)
    sro_prediction = xgboost.predict(test_x)
    classes = sorted(set(config["family_mapping"].values()))
    family_prediction, family_probabilities = _fit_family_classifier(
        train_x,
        [row.family for row in train_rows],
        test_x,
        classes,
        seed,
    )
    sample_ids = [row.xrd_sample_id for row in test_rows]
    prediction_completeness(sample_ids, sample_ids)
    shells = [int(target[5]) for target in config["sro_targets"]]
    metrics = {
        "model": "peak-xgboost",
        "seed": seed,
        "slice": f"{split_column}_test",
        "result_qualification": config["result_qualification"],
        "sro": masked_parent_macro_sro(
            test_y,
            sro_prediction,
            test_mask,
            [row.parent_group_id for row in test_rows],
            shells,
        ),
        "family": family_classification_metrics(
            [row.family for row in test_rows], family_prediction, classes
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"peak_xgboost_seed{seed}_{split_column}_test"
    prediction_path = (output / f"{stem}.parquet").resolve()
    metrics_path = (output / f"{stem}.metrics.json").resolve()
    checkpoint_path = (output / f"{stem}.joblib").resolve()
    joblib.dump(
        {
            "model": "peak-xgboost",
            "seed": seed,
            "split_column": split_column,
            "max_peaks": max_peaks,
            "estimator": xgboost,
        },
        checkpoint_path,
    )
    pd.DataFrame(
        {
            "model": "peak-xgboost",
            "seed": seed,
            "split_column": split_column,
            "slice": f"{split_column}_test",
            "xrd_sample_id": sample_ids,
            "parent_group_id": [row.parent_group_id for row in test_rows],
            "family_prediction": family_prediction,
            "family_probabilities_json": [
                json.dumps(dict(zip(classes, values))) for values in family_probabilities
            ],
            "sro_predictions": [values.tolist() for values in sro_prediction],
        }
    ).to_parquet(prediction_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    submission_path = _write_formal_submission(
        output,
        model_name="peak-xgboost",
        seed=seed,
        split_column=split_column,
        rows=test_rows,
        sro_prediction=sro_prediction,
        family_prediction=family_prediction,
        family_probabilities=family_probabilities,
        classes=classes,
    )
    return {
        "model": "peak-xgboost",
        "seed": seed,
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def _predict_sro(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        torch.from_numpy(features), batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda"
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.unsqueeze(1).to(device, non_blocking=True))
            if isinstance(output, dict):
                output = output["sro"]
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)


def train_neural_sro(
    model_name: str,
    dataset: Path,
    config_path: Path,
    output: Path,
    *,
    seed: int,
    split_column: str = "id_split",
    max_samples_per_split: int | None = None,
) -> dict[str, object]:
    config = _load_config(config_path)
    rows = load_validation_rows(dataset, config)
    if split_column not in config["split_columns"]:
        raise ValueError(f"unsupported split column {split_column!r}")
    leaking = audit_parent_group_leakage(rows, split_column)
    if leaking:
        raise ValueError(f"parent-group leakage in {split_column}: {leaking[:5]}")
    train_rows = [row for row in rows if row.splits[split_column] == "train"]
    validation_rows = [row for row in rows if row.splits[split_column] == "validation"]
    test_rows = [row for row in rows if row.splits[split_column] == "test"]
    if max_samples_per_split is not None:
        limit = int(max_samples_per_split)
        if limit <= 0:
            raise ValueError("max_samples_per_split must be positive")
        train_rows = train_rows[:limit]
        validation_rows = validation_rows[:limit]
        test_rows = test_rows[:limit]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("neural training requires non-empty ID train, validation, and test rows")
    seed_everything(seed)
    budget = config["model_budgets"][model_name]
    batch_size = int(budget["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = spectrum_matrix(train_rows)
    validation_x = spectrum_matrix(validation_rows)
    test_x = spectrum_matrix(test_rows)
    train_y = np.stack([row.sro_values for row in train_rows]).astype(np.float32)
    train_mask = np.stack([row.sro_mask for row in train_rows])
    validation_y = np.stack([row.sro_values for row in validation_rows]).astype(np.float32)
    validation_mask = np.stack([row.sro_mask for row in validation_rows])
    test_y = np.stack([row.sro_values for row in test_rows]).astype(np.float32)
    test_mask = np.stack([row.sro_mask for row in test_rows])
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(train_mask),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model: torch.nn.Module
    if model_name == "resnet1d":
        model = ResNet1D(len(config["sro_targets"]))
    elif model_name == "transformer":
        model = SpectrumTransformer(len(config["sro_targets"]))
    elif model_name == "physics-transformer":
        model = PhysicsConstrainedTransformer(
            len(config["sro_targets"]), peak_count=int(budget["peak_count"])
        )
    else:
        raise ValueError(f"unsupported neural model {model_name!r}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(budget["learning_rate"]))
    shells = [int(target[5]) for target in config["sro_targets"]]
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(budget["max_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for features, truth, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            device_features = features.unsqueeze(1).to(device, non_blocking=True)
            output_value = model(device_features)
            prediction = output_value["sro"] if isinstance(output_value, dict) else output_value
            loss = masked_mse(
                prediction,
                truth.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )
            reconstruction_loss = None
            if isinstance(output_value, dict):
                reconstruction_loss = torch.nn.functional.mse_loss(
                    output_value["reconstruction"],
                    normalize_reconstruction_target(device_features.squeeze(1)),
                )
                loss = loss + float(budget["reconstruction_weight"]) * reconstruction_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = _predict_sro(
            model, validation_x, batch_size=batch_size, device=device
        )
        validation_metrics = masked_parent_macro_sro(
            validation_y,
            validation_prediction,
            validation_mask,
            [row.parent_group_id for row in validation_rows],
            shells,
        )
        validation_mae = float(validation_metrics["masked_parent_macro_mae"])
        history.append({
            "epoch": epoch,
            "train_total_loss": float(np.mean(losses)),
            "validation_parent_macro_mae": validation_mae,
            "forward_reconstruction_weight": float(budget.get("reconstruction_weight", 0.0)),
        })
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(budget["patience"]):
                break
    if best_state is None:
        raise RuntimeError("neural training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_prediction = _predict_sro(model, test_x, batch_size=batch_size, device=device)
    metrics = {
        "model": model_name,
        "seed": seed,
        "slice": f"{split_column}_test",
        "result_qualification": (
            "engineering-smoke" if max_samples_per_split is not None else config["result_qualification"]
        ),
        "max_samples_per_split": max_samples_per_split,
        "best_epoch": best_epoch,
        "validation_parent_macro_mae": best_mae,
        "sro": masked_parent_macro_sro(
            test_y,
            test_prediction,
            test_mask,
            [row.parent_group_id for row in test_rows],
            shells,
        ),
        "history": history,
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_seed{seed}_{split_column}_test"
    checkpoint_path = (output / f"{stem}.pt").resolve()
    prediction_path = (output / f"{stem}.parquet").resolve()
    metrics_path = (output / f"{stem}.metrics.json").resolve()
    torch.save(
        {
            "model": model_name, "seed": seed, "split_column": split_column,
            "best_epoch": best_epoch, "state_dict": best_state,
            "peak_count": int(budget.get("peak_count", 0)),
            "reconstruction_weight": float(budget.get("reconstruction_weight", 0.0)),
            "max_samples_per_split": max_samples_per_split,
        },
        checkpoint_path,
    )
    pd.DataFrame(
        {
            "model": model_name,
            "seed": seed,
            "split_column": split_column,
            "slice": f"{split_column}_test",
            "xrd_sample_id": [row.xrd_sample_id for row in test_rows],
            "parent_group_id": [row.parent_group_id for row in test_rows],
            "family_probabilities_json": "{}",
            "sro_predictions": [values.tolist() for values in test_prediction],
        }
    ).to_parquet(prediction_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    submission_path = _write_formal_submission(
        output,
        model_name=model_name,
        seed=seed,
        split_column=split_column,
        rows=test_rows,
        sro_prediction=test_prediction,
    )
    return {
        "model": model_name,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def generate_report(run_root: Path, config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    def evidence(name: str, fallback: str | None = None) -> dict[str, object]:
        candidates = [run_root / "evidence" / name]
        if fallback:
            candidates.append(run_root / fallback)
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    audit = evidence("canonical_audit.json", "audit.json")
    controls = evidence("controls.json", "controls.json")
    family = evidence("family_support.json")
    identifiability = evidence("identifiability.json")
    ood = evidence("ood_summary.json")
    perturbation = evidence("perturbation_summary.json")
    bootstrap = evidence("bootstrap_summary.json")
    try:
        from .hea_benchmark_readiness import assess_readiness

        readiness = assess_readiness(run_root / "evidence")
    except (ValueError, json.JSONDecodeError):
        readiness = {"core_benchmark_ready": False, "failure_codes": ["INVALID_EVIDENCE"]}
    model_metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "predictions").glob("*.metrics.json"))
    ]
    composition = next(
        (item for item in model_metrics if item["model"] == "composition-ridge"), None
    )
    xrd_models = [item for item in model_metrics if item["model"] != "composition-ridge"]
    xrd_beats_composition = bool(
        composition
        and xrd_models
        and min(item["sro"]["masked_parent_macro_mae"] for item in xrd_models)
        < composition["sro"]["masked_parent_macro_mae"]
    )
    occupancy_coverage = float(
        identifiability.get(
            "identifiable_pair_fraction",
            float(identifiability.get("identifiable_pair_count", 0))
            / max(float(identifiability.get("pair_count", 0)), 1.0),
        )
    )
    occupancy_supported = occupancy_coverage >= 0.05
    publication_ready = bool(readiness.get("core_benchmark_ready", False))
    rows = []
    for item in model_metrics:
        rows.append(
            {
                "model": item["model"],
                "seed": item["seed"],
                "split_column": item.get(
                    "split_column", str(item.get("slice", "id_split_test")).removesuffix("_test")
                ),
                "sro_parent_macro_mae": item["sro"]["masked_parent_macro_mae"],
                "sro_rmse": item["sro"]["rmse"],
            }
        )
    model_aggregates: dict[str, dict[str, float | int | None]] = {}
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        mae_values = np.asarray(
            [row["sro_parent_macro_mae"] for row in model_rows], dtype=float
        )
        rmse_values = np.asarray([row["sro_rmse"] for row in model_rows], dtype=float)
        model_aggregates[model_name] = {
            "seed_count": len(model_rows),
            "sro_parent_macro_mae_mean": float(mae_values.mean()),
            "sro_parent_macro_mae_std": float(mae_values.std()),
            "sro_rmse_mean": float(rmse_values.mean()),
            "sro_rmse_std": float(rmse_values.std()),
            "uncertainty": "random_seed_standard_deviation",
        }
    composition_aggregate = model_aggregates.get("composition-ridge")
    split_model_aggregates: dict[str, dict[str, dict[str, float | int]]] = {}
    for split_name in sorted({row["split_column"] for row in rows}):
        split_model_aggregates[split_name] = {}
        for model_name in sorted({row["model"] for row in rows}):
            selected = [
                row for row in rows
                if row["split_column"] == split_name and row["model"] == model_name
            ]
            if not selected:
                continue
            values = np.asarray([row["sro_parent_macro_mae"] for row in selected], dtype=float)
            split_model_aggregates[split_name][model_name] = {
                "seed_count": len(selected),
                "sro_parent_macro_mae_mean": float(values.mean()),
                "sro_parent_macro_mae_std": float(values.std()),
            }
    xrd_aggregate_values = [
        value
        for name, value in model_aggregates.items()
        if name != "composition-ridge"
    ]
    xrd_beats_composition = bool(
        composition_aggregate
        and xrd_aggregate_values
        and min(value["sro_parent_macro_mae_mean"] for value in xrd_aggregate_values)
        < composition_aggregate["sro_parent_macro_mae_mean"]
    )
    pd.DataFrame(rows).to_csv(run_root / "model_metrics.csv", index=False)
    aggregate = {
        "result_qualification": config["result_qualification"],
        "release_scope": "sro_core",
        "primary_tasks": ["sro"],
        "diagnostic_tasks": ["family", "occupancy_identifiability"],
        "publication_ready": publication_ready,
        "core_benchmark_ready": publication_ready,
        "failure_codes": readiness.get("failure_codes", []),
        "data_leakage_passed": audit.get("parent_group_leakage_count") == 0,
        "control_gate_passed": bool(controls.get("control_gate_passed", False)),
        "xrd_beats_composition_only": xrd_beats_composition,
        "model_gain_is_release_gate": False,
        "occupancy_identifiable_coverage": occupancy_coverage,
        "occupancy_statistical_support": occupancy_supported,
        "diagnostic_family_status": readiness.get("diagnostic_family_status", "diagnostic_not_generated"),
        "diagnostic_occupancy_status": readiness.get("diagnostic_occupancy_status", "diagnostic_not_generated"),
        "experimental_noise_calibrated": False,
        "family": family,
        "occupancy_identifiability": identifiability,
        "ood_validation_status": ood.get("status", "not_run"),
        "perturbation_validation_status": perturbation.get("status", "not_run"),
        "parent_bootstrap": bootstrap,
        "models": rows,
        "model_aggregates": model_aggregates,
        "split_model_aggregates": split_model_aggregates,
    }
    aggregate_path = run_root / "validation_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# SiteXRD-Bench HEA v1.1 SRO Core",
        "",
        f"Core benchmark readiness: **{'READY' if publication_ready else 'NOT READY'}**.",
        "Release scope: **SRO only**. Family and occupancy are nonblocking diagnostics.",
        "",
        "## SRO Core Gates",
        "",
        f"- Parent-group leakage audit: {'PASS' if aggregate['data_leakage_passed'] else 'FAIL'}",
        f"- Oracle and shuffled controls: {'PASS' if aggregate['control_gate_passed'] else 'FAIL'}",
        f"- OOD evaluation: {str(aggregate['ood_validation_status']).upper()}",
        f"- Matched perturbation evaluation: {str(aggregate['perturbation_validation_status']).upper()}",
        f"- XRD beats composition-only: {'YES' if xrd_beats_composition else 'NO'} (scientific result, not a release gate)",
        "",
        "## SRO Results",
        "",
        "| Split | Model | Seed | Parent-macro MAE | RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split_column']} | {row['model']} | {row['seed']} | {row['sro_parent_macro_mae']:.4f} | {row['sro_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics And Limitations",
            "",
            f"- Family diagnostic: {aggregate['diagnostic_family_status']}. Labels are automated and unadjudicated; no Family leaderboard score is published.",
            f"- Occupancy identifiability diagnostic: {aggregate['diagnostic_occupancy_status']}. Ambiguous pairs are excluded from ranking metrics.",
            "- Noise reference: frozen simulated perturbations. Real repeated-measurement calibration is deferred to v1.2.",
            "- Random-seed standard deviations and parent-bootstrap confidence intervals are reported separately.",
        ]
    )
    report_path = run_root / "validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report_path": str(report_path.resolve()),
        "summary_path": str(aggregate_path.resolve()),
        "model_count": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit HEA validation manifests")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--config", type=Path, required=True)
    controls = subparsers.add_parser("controls", help="run oracle and shuffled controls")
    controls.add_argument("--dataset", type=Path, required=True)
    controls.add_argument("--config", type=Path, required=True)
    controls.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    controls.add_argument("--seed", type=int, default=2027)
    train = subparsers.add_parser("train", help="train a frozen validation baseline")
    train.add_argument(
        "--model",
        choices=["composition-ridge", "composition-extratrees", "peak-xgboost", "resnet1d", "transformer", "physics-transformer"],
        required=True,
    )
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=2027)
    train.add_argument("--split-column", default="id_split")
    train.add_argument("--max-samples-per-split", type=int)
    materialize = subparsers.add_parser(
        "materialize-submission", help="recover a formal submission from a completed prediction parquet"
    )
    materialize.add_argument("--dataset", type=Path, required=True)
    materialize.add_argument("--predictions", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--model", required=True)
    materialize.add_argument("--seed", type=int, required=True)
    materialize.add_argument("--split-column", required=True)
    report = subparsers.add_parser("report", help="aggregate validation artifacts")
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        payload, passed = run_audit(args.dataset, args.config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 2
    if args.command == "controls":
        payload, passed = run_controls(
            args.dataset, args.config, split=args.split, seed=args.seed
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 2
    if args.command == "train":
        if args.model == "composition-ridge":
            payload = train_composition_ridge(
                args.dataset, args.config, args.output, seed=args.seed, split_column=args.split_column
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.model == "composition-extratrees":
            payload = train_composition_extratrees(
                args.dataset, args.config, args.output, seed=args.seed, split_column=args.split_column
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.model == "peak-xgboost":
            payload = train_peak_xgboost(
                args.dataset, args.config, args.output, seed=args.seed, split_column=args.split_column
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.model in {"resnet1d", "transformer", "physics-transformer"}:
            payload = train_neural_sro(
                args.model,
                args.dataset,
                args.config,
                args.output,
                seed=args.seed,
                split_column=args.split_column,
                max_samples_per_split=args.max_samples_per_split,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
    if args.command == "report":
        payload = generate_report(args.run_root, args.config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "materialize-submission":
        submission = materialize_formal_submission(
            args.dataset,
            args.predictions,
            args.output,
            model_name=args.model,
            seed=args.seed,
            split_column=args.split_column,
        )
        print(json.dumps({"status": "pass", "submission": str(submission)}, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
