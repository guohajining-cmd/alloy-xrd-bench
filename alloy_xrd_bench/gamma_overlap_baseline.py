"""Training-only physics baseline for local overlap windows."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .gamma_overlap_physics import pseudo_voigt_profile
from .gamma_overlap_scoring import hkl_group_id, score_component_predictions, window_component_truth


WINDOW_BASELINE_VERSION = "bic-background-null-v2"
BASELINE_COMPONENT_COLUMNS = [
    "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
    "center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence",
]


def build_training_candidate_catalog(
    phase_reflection_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build phase/HKL center priors using training parents only."""

    train_parents: set[str] = set()
    for row in split_rows:
        if str(row["split"]) != "train":
            continue
        train_parents.add(str(row["gamma_parent_id"]))
        train_parents.add(str(row["gamma_prime_parent_id"]))
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in phase_reflection_rows:
        if str(row["parent_structure_id"]) not in train_parents:
            continue
        phase = str(row["phase"])
        group_id = hkl_group_id(phase, int(row["h"]), int(row["k"]), int(row["l"]))
        key = (phase, group_id)
        entry = grouped.setdefault(key, {
            "h": int(row["h"]), "k": int(row["k"]), "l": int(row["l"]), "centers": [],
        })
        entry["centers"].append(float(row["center_deg"]))
    return [
        {
            "phase": key[0], "hkl_group_id": key[1],
            "h": values["h"], "k": values["k"], "l": values["l"],
            "center_deg": float(np.median(values["centers"])),
            "training_parent_count": len(values["centers"]),
        }
        for key, values in sorted(grouped.items(), key=lambda item: (np.median(item[1]["centers"]), item[0]))
    ]


def fit_window_baseline(
    *,
    window_id: str,
    two_theta: np.ndarray,
    observed: np.ndarray,
    candidate_catalog: Sequence[Mapping[str, Any]],
    max_components: int = 2,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Choose background-only or up to max_components profiles by BIC."""

    grid = np.asarray(two_theta, dtype=float)
    signal = np.asarray(observed, dtype=float)
    if grid.ndim != 1 or signal.shape != grid.shape or len(grid) < 10:
        raise ValueError("window arrays must be aligned one-dimensional vectors")
    if not np.isfinite(grid).all() or not np.isfinite(signal).all() or not np.all(np.diff(grid) > 0):
        raise ValueError("window arrays must be finite with an increasing grid")
    if int(max_components) != max_components or max_components < 0:
        raise ValueError("max_components must be a non-negative integer")
    candidates = [
        dict(row)
        for row in candidate_catalog
        if float(grid.min()) - 0.25 <= float(row["center_deg"]) <= float(grid.max()) + 0.25
    ]
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update({
        "sample_id": window_id, "baseline_version": WINDOW_BASELINE_VERSION,
        "status": "all_fits_failed", "attempted_fit_count": 0, "failed_fit_count": 0,
        "selected_component_count": None, "selected_bic": None,
    })
    best: tuple[float, list[dict[str, Any]]] | None = None
    for count in range(0, min(int(max_components), len(candidates)) + 1):
        for selected in combinations(candidates, count):
            diagnostics["attempted_fit_count"] += 1
            fitted = _fit_candidate_set(grid, signal, selected)
            if fitted is None:
                diagnostics["failed_fit_count"] += 1
                continue
            bic, predictions = fitted
            if best is None or bic < best[0]:
                best = bic, predictions
    if best is None:
        return []
    diagnostics.update(status="selected", selected_component_count=len(best[1]), selected_bic=best[0])
    return [
        {
            "sample_id": window_id,
            "component_id": f"baseline_{window_id}_{index:02d}",
            **row,
            "confidence": 1.0,
        }
        for index, row in enumerate(sorted(best[1], key=lambda item: item["center_deg"]))
    ]


def run_window_baseline(
    *,
    release_dir: str | Path,
    output_path: str | Path,
    split: str = "validation",
    max_windows: int | None = None,
) -> dict[str, Any]:
    release = Path(release_dir).resolve()
    phase_rows = pd.read_parquet(release / "phase_reflections.parquet").to_dict(orient="records")
    splits = pd.read_parquet(release / "splits.parquet").to_dict(orient="records")
    catalog = build_training_candidate_catalog(phase_rows, splits)
    windows = pd.read_parquet(release / "overlap_windows.parquet")
    windows = windows[windows["split"] == split].sort_values("window_id")
    if max_windows is not None:
        windows = windows.head(int(max_windows))
    predictions: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    for row in windows.to_dict(orient="records"):
        diagnostic: dict[str, Any] = {}
        with np.load(release / str(row["window_path"])) as artifact:
            predictions.extend(
                fit_window_baseline(
                    window_id=str(row["window_id"]),
                    two_theta=artifact["two_theta"],
                    observed=artifact["observed"],
                    candidate_catalog=catalog,
                    diagnostics=diagnostic,
                )
            )
        fit_diagnostics.append(diagnostic)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).reindex(columns=BASELINE_COMPONENT_COLUMNS).to_parquet(output, index=False)
    pd.DataFrame(fit_diagnostics).to_parquet(output.with_name(f"{output.stem}_fit_diagnostics.parquet"), index=False)
    report: dict[str, Any] = {
        "baseline_version": WINDOW_BASELINE_VERSION,
        "split": split,
        "window_count": len(windows),
        "prediction_count": len(predictions),
        "candidate_count": len(catalog),
        "failed_window_count": sum(row["status"] == "all_fits_failed" for row in fit_diagnostics),
        "failed_candidate_fit_count": sum(row["failed_fit_count"] for row in fit_diagnostics),
        "output_path": str(output.resolve()),
    }
    if split != "private_test":
        component_rows = pd.read_parquet(release / "components.parquet").to_dict(orient="records")
        truth = window_component_truth(component_rows, windows.to_dict(orient="records"))
        report["metrics"] = score_component_predictions(
            truth,
            predictions,
            expected_sample_ids=[str(row["window_id"]) for row in windows.to_dict(orient="records")],
            expected_parent_ids={
                str(row["window_id"]): str(row["parent_pair_id"])
                for row in windows.to_dict(orient="records")
            },
        )
    summary_path = output.with_suffix(".summary.json")
    import json

    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_public_window_baseline(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    split: str = "validation",
    baseline_kind: str = "bic",
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Fit a public-release window baseline and score its standard submission."""

    if baseline_kind not in {"single", "bic"}:
        raise ValueError("baseline_kind must be single or bic")
    from .gamma_overlap_public import evaluate_public_submission

    release = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    submission = output / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    frozen_catalog = release / "baselines/training_candidate_catalog.parquet"
    if frozen_catalog.exists():
        catalog = pd.read_parquet(frozen_catalog).sort_values("center_deg").to_dict(orient="records")
    else:
        train = pd.read_parquet(release / "window/components.parquet")
        train = train.loc[train["split"].astype(str).eq("train") & train["score_eligible"].astype(bool)]
        catalog = (
            train.groupby(["phase", "h", "k", "l"], as_index=False)["center_deg"]
            .median()
            .sort_values("center_deg")
            .to_dict(orient="records")
        )
    samples = pd.read_parquet(release / "window/samples.parquet")
    samples = samples.loc[samples["split"].astype(str).eq(split)].sort_values("sample_id")
    if max_windows is not None:
        samples = samples.head(int(max_windows))
    predictions: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    for row in samples.to_dict(orient="records"):
        diagnostic: dict[str, Any] = {}
        with np.load(release / str(row["pattern_path"])) as artifact:
            predictions.extend(
                fit_window_baseline(
                    window_id=str(row["sample_id"]),
                    two_theta=artifact["two_theta"],
                    observed=artifact["observed"],
                    candidate_catalog=catalog,
                    max_components=1 if baseline_kind == "single" else 2,
                    diagnostics=diagnostic,
                )
            )
        fit_diagnostics.append(diagnostic)
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame = prediction_frame.reindex(columns=BASELINE_COMPONENT_COLUMNS)
    pd.DataFrame(fit_diagnostics).to_parquet(output / "fit_diagnostics.parquet", index=False)
    prediction_frame.to_parquet(submission / "submission_components.parquet", index=False)
    count_by_sample = prediction_frame.groupby("sample_id").size() if len(prediction_frame) else {}
    pd.DataFrame({
        "sample_id": samples["sample_id"].astype(str),
        "predicted_component_count": [int(count_by_sample.get(value, 0)) for value in samples["sample_id"].astype(str)],
    }).to_parquet(submission / "submission_samples.parquet", index=False)
    score = evaluate_public_submission(
        release, "window", split, submission, output / "score"
    )
    return {
        "status": "pass",
        "baseline_kind": baseline_kind,
        "baseline_version": WINDOW_BASELINE_VERSION,
        "failed_window_count": sum(row["status"] == "all_fits_failed" for row in fit_diagnostics),
        "failed_candidate_fit_count": sum(row["failed_fit_count"] for row in fit_diagnostics),
        "sample_count": len(samples),
        "prediction_count": len(prediction_frame),
        "candidate_count": len(catalog),
        "metrics": score["metrics"],
        "output_dir": str(output),
    }


def run_public_no_spectrum_prior(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    task: str,
    split: str = "validation",
) -> dict[str, Any]:
    """Score a train-only component prior that never reads an XRD pattern."""

    if task not in {"full", "window"}:
        raise ValueError("task must be full or window")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    from .gamma_overlap_public import evaluate_public_submission

    release = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    submission = output / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(release / task / "samples.parquet")
    train_samples = samples.loc[samples["split"].astype(str).eq("train")]
    evaluation_samples = samples.loc[samples["split"].astype(str).eq(split)].sort_values("sample_id")
    train = pd.read_parquet(release / task / "components.parquet")
    train = train.loc[
        train["split"].astype(str).eq("train") & train["score_eligible"].astype(bool)
    ].copy()
    train_count = max(len(train_samples), 1)
    per_sample_counts = (
        train.groupby("sample_id").size().reindex(train_samples["sample_id"], fill_value=0)
    )
    prior_count = int(np.rint(float(np.median(per_sample_counts)))) if len(per_sample_counts) else 0
    class_columns = ["phase", "hkl_group_id"]
    catalog = []
    for (phase, group_id), rows in train.groupby(class_columns, sort=True):
        first = rows.sort_values(["h", "k", "l"]).iloc[0]
        catalog.append({
            "phase": str(phase),
            "hkl_group_id": str(group_id),
            "h": int(first["h"]),
            "k": int(first["k"]),
            "l": int(first["l"]),
            "center_deg": float(rows["center_deg"].median()),
            "fwhm_deg": float(rows["fwhm_deg"].median()),
            "integrated_intensity": float(rows["integrated_intensity"].median()),
            "eta": float(rows["eta"].median()),
            "confidence": float(rows["sample_id"].nunique() / train_count),
        })
    catalog.sort(key=lambda row: (-row["confidence"], row["phase"], row["hkl_group_id"]))
    selected = catalog[:prior_count]
    predictions = []
    for sample_id in evaluation_samples["sample_id"].astype(str):
        for index, component in enumerate(selected):
            predictions.append({
                "sample_id": sample_id,
                "component_id": f"prior_{index:02d}",
                **component,
            })
    columns = [
        "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
        "center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence",
    ]
    pd.DataFrame(predictions).reindex(columns=columns).to_parquet(
        submission / "submission_components.parquet", index=False
    )
    pd.DataFrame({
        "sample_id": evaluation_samples["sample_id"].astype(str),
        "predicted_component_count": prior_count,
    }).to_parquet(submission / "submission_samples.parquet", index=False)
    score = evaluate_public_submission(release, task, split, submission, output / "score")
    report = {
        "status": "pass",
        "baseline_kind": "no-spectrum",
        "task": task,
        "split": split,
        "train_sample_count": len(train_samples),
        "sample_count": len(evaluation_samples),
        "prior_component_count": prior_count,
        "metrics": score["metrics"],
        "output_dir": str(output),
    }
    import json

    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _public_peak_features(
    two_theta: np.ndarray, observed: np.ndarray, *, max_peaks: int = 24
) -> np.ndarray:
    grid = np.asarray(two_theta, dtype=float)
    signal = np.asarray(observed, dtype=float)
    scale = max(float(np.quantile(signal, 0.99)), 1e-8)
    normalized = np.clip(signal / scale, 0.0, None)
    peaks, properties = find_peaks(normalized, prominence=0.001)
    order = np.argsort(properties.get("prominences", np.zeros(len(peaks))))[::-1][:max_peaks]
    selected = peaks[order]
    widths = np.zeros(len(selected), dtype=float)
    if len(selected):
        widths = peak_widths(normalized, selected, rel_height=0.5)[0]
        widths *= float(np.median(np.diff(grid)))
    peak_block = np.zeros((max_peaks, 3), dtype=np.float32)
    for index, (peak, width) in enumerate(zip(selected, widths)):
        peak_block[index] = [
            (float(grid[peak]) - 20.0) / 80.0,
            float(normalized[peak]),
            float(width) / 1.2,
        ]
    total = max(float(normalized.sum()), 1e-8)
    global_features = np.asarray([
        (float(grid.min()) - 20.0) / 80.0,
        (float(grid.max()) - 20.0) / 80.0,
        float(normalized.mean()),
        float(normalized.std()),
        float(np.quantile(normalized, 0.5)),
        float(np.quantile(normalized, 0.9)),
        float(np.quantile(normalized, 0.99)),
        float(np.sum(grid * normalized) / total / 100.0),
        float(np.trapezoid(normalized, grid) / 80.0),
        float(len(peaks) / max(len(grid), 1)),
    ], dtype=np.float32)
    return np.concatenate([peak_block.reshape(-1), global_features])


def run_public_peak_feature_baseline(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    task: str,
    split: str = "validation",
    seed: int = 2027,
    n_estimators: int = 300,
    max_peaks: int = 24,
) -> dict[str, Any]:
    """Train a class-wise component model from explicit peak summary features."""

    if task not in {"full", "window"}:
        raise ValueError("task must be full or window")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    from .gamma_overlap_public import evaluate_public_submission

    release = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    submission = output / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    all_samples = pd.read_parquet(release / task / "samples.parquet")
    train_samples = all_samples.loc[all_samples["split"].astype(str).eq("train")].sort_values("sample_id")
    evaluation_samples = all_samples.loc[all_samples["split"].astype(str).eq(split)].sort_values("sample_id")
    labels = pd.read_parquet(release / task / "components.parquet")
    train_labels = labels.loc[
        labels["split"].astype(str).eq("train") & labels["score_eligible"].astype(bool)
    ].copy()
    catalog = []
    medians = []
    for (phase, group_id), rows in train_labels.groupby(["phase", "hkl_group_id"], sort=True):
        first = rows.sort_values(["h", "k", "l"]).iloc[0]
        catalog.append((str(phase), str(group_id), int(first["h"]), int(first["k"]), int(first["l"])))
        medians.append(np.asarray([
            (float(rows["center_deg"].median()) - 20.0) / 80.0,
            (float(rows["fwhm_deg"].median()) - 0.03) / 1.17,
            float(rows["integrated_intensity"].median()) / 100.0,
            float(rows["eta"].median()),
        ], dtype=np.float32))
    class_index = {(phase, group_id): index for index, (phase, group_id, *_rest) in enumerate(catalog)}

    def features(samples: pd.DataFrame) -> np.ndarray:
        rows = []
        for sample in samples.to_dict(orient="records"):
            with np.load(release / str(sample["pattern_path"])) as artifact:
                rows.append(_public_peak_features(
                    artifact["two_theta"], artifact["observed"], max_peaks=max_peaks
                ))
        return np.stack(rows)

    train_x = features(train_samples)
    evaluation_x = features(evaluation_samples)
    target = np.zeros((len(train_samples), len(catalog), 5), dtype=np.float32)
    if medians:
        target[:, :, 1:] = np.stack(medians)[None, :, :]
    sample_index = {
        sample_id: index for index, sample_id in enumerate(train_samples["sample_id"].astype(str))
    }
    for row in train_labels.to_dict(orient="records"):
        row_index = sample_index[str(row["sample_id"])]
        column = class_index[(str(row["phase"]), str(row["hkl_group_id"]))]
        target[row_index, column] = [
            1.0,
            np.clip((float(row["center_deg"]) - 20.0) / 80.0, 0.0, 1.0),
            np.clip((float(row["fwhm_deg"]) - 0.03) / 1.17, 0.0, 1.0),
            np.clip(float(row["integrated_intensity"]) / 100.0, 0.0, 1.0),
            np.clip(float(row["eta"]), 0.0, 1.0),
        ]
    estimator = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=-1,
    ).fit(train_x, target.reshape(len(train_x), -1))
    prediction = estimator.predict(evaluation_x).reshape(len(evaluation_x), len(catalog), 5)
    query_count = 20 if task == "full" else 3
    predictions = []
    for sample_offset, sample_id in enumerate(evaluation_samples["sample_id"].astype(str)):
        probabilities = np.clip(prediction[sample_offset, :, 0], 0.0, 1.0)
        selected = np.flatnonzero(probabilities >= 0.5)
        selected = selected[np.argsort(probabilities[selected])[::-1]][:query_count]
        for query_index, class_id in enumerate(selected):
            phase, group_id, h, k, l = catalog[int(class_id)]
            values = np.clip(prediction[sample_offset, class_id, 1:], 0.0, 1.0)
            predictions.append({
                "sample_id": sample_id,
                "component_id": f"peak_feature_{query_index:02d}",
                "phase": phase,
                "h": h,
                "k": k,
                "l": l,
                "hkl_group_id": group_id,
                "center_deg": 20.0 + 80.0 * float(values[0]),
                "fwhm_deg": 0.03 + 1.17 * float(values[1]),
                "integrated_intensity": 100.0 * float(values[2]),
                "eta": float(values[3]),
                "confidence": float(probabilities[class_id]),
            })
    columns = [
        "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
        "center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence",
    ]
    frame = pd.DataFrame(predictions).reindex(columns=columns)
    frame.to_parquet(submission / "submission_components.parquet", index=False)
    counts = frame.groupby("sample_id").size() if len(frame) else {}
    pd.DataFrame({
        "sample_id": evaluation_samples["sample_id"].astype(str),
        "predicted_component_count": [
            int(counts.get(sample_id, 0)) for sample_id in evaluation_samples["sample_id"].astype(str)
        ],
    }).to_parquet(submission / "submission_samples.parquet", index=False)
    joblib.dump({
        "estimator": estimator,
        "catalog": catalog,
        "task": task,
        "seed": seed,
        "max_peaks": max_peaks,
    }, output / "model.joblib")
    score = evaluate_public_submission(release, task, split, submission, output / "score")
    report = {
        "status": "pass",
        "baseline_kind": "peak-feature",
        "task": task,
        "split": split,
        "seed": seed,
        "train_sample_count": len(train_samples),
        "sample_count": len(evaluation_samples),
        "feature_count": int(train_x.shape[1]),
        "metrics": score["metrics"],
        "output_dir": str(output),
    }
    import json

    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _fit_candidate_set(
    grid: np.ndarray,
    observed: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]] | None:
    count = len(candidates)
    total_area = max(float(np.trapezoid(np.clip(observed, 0.0, None), grid)), 1e-6)
    maximum = max(float(observed.max()), 1e-6)
    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for candidate in candidates:
        center = float(candidate["center_deg"])
        initial.extend([center, 0.15, total_area / count, 0.5])
        lower.extend([center - 0.25, 0.03, 0.0, 0.0])
        upper.extend([center + 0.25, 0.80, total_area * 5.0, 1.0])
    initial.extend([max(float(observed.min()), 0.0), 0.0])
    lower.extend([0.0, -maximum])
    upper.extend([maximum, maximum])
    normalized_x = np.linspace(-1.0, 1.0, len(grid))

    def model(parameters: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid)
        for index in range(count):
            center, fwhm, area, eta = parameters[index * 4 : index * 4 + 4]
            result += pseudo_voigt_profile(
                grid,
                center_deg=float(center),
                fwhm_deg=float(fwhm),
                integrated_intensity=float(area),
                eta=float(eta),
            )
        return result + parameters[-2] + parameters[-1] * normalized_x

    try:
        result = least_squares(
            lambda parameters: model(parameters) - observed,
            np.asarray(initial),
            bounds=(np.asarray(lower), np.asarray(upper)),
            max_nfev=1000,
        )
    except (ValueError, FloatingPointError):
        return None
    if not result.success or not np.isfinite(result.x).all():
        return None
    residual = model(result.x) - observed
    if not np.isfinite(residual).all():
        return None
    rss = max(float(np.dot(residual, residual)), np.finfo(float).tiny)
    parameter_count = len(result.x)
    bic = len(grid) * np.log(rss / len(grid)) + parameter_count * np.log(len(grid))
    predictions = []
    for index, candidate in enumerate(candidates):
        center, fwhm, area, eta = result.x[index * 4 : index * 4 + 4]
        predictions.append(
            {
            "phase": str(candidate["phase"]),
            "h": int(candidate["h"]),
            "k": int(candidate["k"]),
            "l": int(candidate["l"]),
            "hkl_group_id": str(candidate.get("hkl_group_id") or hkl_group_id(
                str(candidate["phase"]), int(candidate["h"]), int(candidate["k"]), int(candidate["l"])
            )),
                "center_deg": float(center),
                "fwhm_deg": float(fwhm),
                "integrated_intensity": float(area),
                "eta": float(eta),
            }
        )
    return float(bic), predictions
