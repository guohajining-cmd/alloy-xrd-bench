"""Noise-calibrated identifiability for HEA occupancy counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def weak_peak_weighted_l1(
    first: np.ndarray,
    second: np.ndarray,
    *,
    weak_threshold: float = 5.0,
    weak_weight: float = 5.0,
    intensity_scale: float = 100.0,
) -> float:
    """Return an L1 distance that preserves evidence in weak-intensity regions."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("weak-peak distance requires aligned vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("weak-peak distance requires finite spectra")
    if weak_weight < 1.0 or weak_threshold < 0.0 or intensity_scale <= 0.0:
        raise ValueError("invalid weak-peak distance parameters")
    weak = np.maximum(left, right) <= weak_threshold
    weights = np.where(weak, weak_weight, 1.0)
    return float(np.average(np.abs(left - right), weights=weights) / intensity_scale)


def nearest_prototype_accuracy(
    prototype_a: np.ndarray,
    prototype_b: np.ndarray,
    views_a: np.ndarray,
    views_b: np.ndarray,
) -> float:
    """Evaluate a fixed nearest-clean-prototype oracle on nuisance views."""

    first = np.asarray(prototype_a, dtype=float)
    second = np.asarray(prototype_b, dtype=float)
    a_views = np.asarray(views_a, dtype=float)
    b_views = np.asarray(views_b, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("oracle prototypes must be aligned vectors")
    if a_views.ndim != 2 or b_views.ndim != 2 or a_views.shape[1:] != first.shape or b_views.shape[1:] != first.shape:
        raise ValueError("oracle views must align with prototypes")
    if not all(np.isfinite(value).all() for value in (first, second, a_views, b_views)):
        raise ValueError("oracle spectra must be finite")
    a_correct = (
        np.mean(np.abs(a_views - first), axis=1)
        <= np.mean(np.abs(a_views - second), axis=1)
    )
    b_correct = (
        np.mean(np.abs(b_views - second), axis=1)
        < np.mean(np.abs(b_views - first), axis=1)
    )
    return float(np.concatenate([a_correct, b_correct]).mean())


def assign_identifiability_classes(
    pairs: pd.DataFrame,
    *,
    identifiable_ratio: float = 1.0,
    identifiable_accuracy: float = 0.80,
    weak_ratio: float = 0.50,
    weak_accuracy: float = 0.60,
) -> pd.DataFrame:
    """Assign conservative three-level labels from signal/noise and oracle evidence."""

    required = {"identifiability_ratio", "oracle_accuracy"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"identifiability classes missing columns: {sorted(missing)}")
    result = pairs.copy()
    ratio = result["identifiability_ratio"].astype(float)
    accuracy = result["oracle_accuracy"].astype(float)
    available = np.isfinite(accuracy)
    identifiable = available & (ratio >= identifiable_ratio) & (
        accuracy >= identifiable_accuracy
    )
    weak = available & ~identifiable & (ratio >= weak_ratio) & (
        accuracy >= weak_accuracy
    )
    result["identifiability_class"] = np.select(
        [identifiable, weak],
        ["identifiable", "weakly_identifiable"],
        default="non_identifiable",
    )
    result["identifiability_status"] = np.where(
        available, "calibrated", "missing_views"
    )
    result["noise_identifiable"] = identifiable
    return result


def add_noise_distances(
    clean: np.ndarray, observed: np.ndarray, *, intensity_scale: float = 100.0
) -> pd.DataFrame:
    clean_values = np.asarray(clean, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.float64)
    if clean_values.shape != observed_values.shape or clean_values.ndim != 2:
        raise ValueError("clean and observed arrays must have the same 2D shape")
    l1 = np.mean(np.abs(observed_values - clean_values), axis=1) / intensity_scale
    numerator = np.sum(clean_values * observed_values, axis=1)
    denominator = np.linalg.norm(clean_values, axis=1) * np.linalg.norm(
        observed_values, axis=1
    )
    similarity = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )
    return pd.DataFrame(
        {
            "noise_l1_distance": l1,
            "noise_cosine_distance": np.maximum(0.0, 1.0 - similarity),
        }
    )


def calibrate_parent_noise_floors(
    distances: pd.DataFrame, *, quantile: float = 0.95, min_observations: int = 3
) -> pd.DataFrame:
    required = {
        "parent_group_id",
        "noise_l1_distance",
        "noise_cosine_distance",
    }
    missing = required.difference(distances.columns)
    if missing:
        raise ValueError(f"missing distance columns: {sorted(missing)}")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    grouped = distances.groupby("parent_group_id", sort=True)
    floors = grouped[["noise_l1_distance", "noise_cosine_distance"]].quantile(
        quantile
    )
    floors = floors.rename(
        columns={
            "noise_l1_distance": "noise_l1_q95",
            "noise_cosine_distance": "noise_cosine_q95",
        }
    )
    floors["noise_observation_count"] = grouped.size()
    global_l1 = float(distances["noise_l1_distance"].quantile(quantile))
    global_cosine = float(distances["noise_cosine_distance"].quantile(quantile))
    insufficient = floors["noise_observation_count"] < min_observations
    floors.loc[insufficient, "noise_l1_q95"] = global_l1
    floors.loc[insufficient, "noise_cosine_q95"] = global_cosine
    floors["used_global_fallback"] = insufficient
    floors["global_noise_l1_q95"] = global_l1
    floors["global_noise_cosine_q95"] = global_cosine
    return floors


def label_pair_identifiability(
    pairs: pd.DataFrame,
    floors: pd.DataFrame,
    *,
    min_effect_ratio: float = 1.0,
    minimum_denominator: float = 1e-12,
) -> pd.DataFrame:
    if "parent_group_id" not in pairs or "xrd_l1_distance" not in pairs:
        raise ValueError("pairs require parent_group_id and xrd_l1_distance")
    floor_frame = floors.reset_index()
    result = pairs.merge(floor_frame, on="parent_group_id", how="left", validate="many_to_one")
    if result["global_noise_l1_q95"].isna().all():
        raise ValueError("noise calibration has no global fallback")
    global_fallback = float(result["global_noise_l1_q95"].dropna().iloc[0])
    has_parent_floor = result["noise_l1_q95"].notna()
    parent_used_fallback = result.get(
        "used_global_fallback", pd.Series(False, index=result.index)
    ).fillna(True).astype(bool)
    result["noise_floor_source"] = np.where(
        has_parent_floor & ~parent_used_fallback,
        "parent_specific",
        "global_fallback",
    )
    result["noise_l1_floor"] = result["noise_l1_q95"].fillna(global_fallback)
    result["identifiability_ratio"] = result["xrd_l1_distance"] / np.maximum(
        result["noise_l1_floor"], minimum_denominator
    )
    result["noise_identifiable"] = result["identifiability_ratio"] >= min_effect_ratio
    return result


def audit_parent_noise_coverage(
    pairs: pd.DataFrame,
    floors: pd.DataFrame,
    *,
    minimum_variants: int = 5,
) -> dict[str, object]:
    """Report whether every occupancy parent has its own frozen noise views."""

    if "parent_group_id" not in pairs:
        raise ValueError("pairs require parent_group_id")
    if floors.index.name != "parent_group_id":
        if "parent_group_id" not in floors:
            raise ValueError("floors require parent_group_id")
        floor_frame = floors.set_index("parent_group_id")
    else:
        floor_frame = floors
    if "noise_observation_count" not in floor_frame:
        raise ValueError("floors require noise_observation_count")
    parents = sorted(set(pairs["parent_group_id"].astype(str)))
    counts = floor_frame["noise_observation_count"]
    count_by_parent = {str(key): int(value) for key, value in counts.items()}
    missing = [
        parent
        for parent in parents
        if count_by_parent.get(parent, 0) < int(minimum_variants)
    ]
    covered = len(parents) - len(missing)
    return {
        "occupancy_parent_group_count": len(parents),
        "parent_specific_calibration_group_count": covered,
        "parent_specific_coverage_fraction": covered / len(parents) if parents else 0.0,
        "minimum_variants_per_parent": int(minimum_variants),
        "coverage_complete": not missing,
        "missing_parent_group_ids": missing,
    }


def _clean_source_index(dataset: Path, source_ids: set[str]) -> pd.DataFrame:
    matches = []
    candidates = [dataset / "manifests" / "xrd_samples.parquet"]
    candidates.extend(dataset.glob("xrd/runs/*/xrd_samples.parquet"))
    for path in candidates:
        if "observation" in path.as_posix() or not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["xrd_sample_id", "grid_shard_path", "grid_row_index"],
        )
        selected = frame.loc[frame["xrd_sample_id"].isin(source_ids)].copy()
        if not selected.empty:
            matches.append(selected)
    if not matches:
        raise ValueError("no clean XRD sources match the observation manifest")
    sources = pd.concat(matches, ignore_index=True).drop_duplicates("xrd_sample_id")
    missing = source_ids.difference(sources["xrd_sample_id"])
    if missing:
        raise ValueError(f"missing {len(missing)} clean XRD source IDs")
    return sources.set_index("xrd_sample_id")


def compute_observation_distances(
    dataset: Path,
    observations: pd.DataFrame,
    *,
    observation_dataset: Path | None = None,
) -> pd.DataFrame:
    observation_root = observation_dataset or dataset
    sources = _clean_source_index(
        dataset, set(observations["source_xrd_sample_id"].astype(str))
    )
    rows = []
    for observation_path, chunk in observations.groupby("grid_shard_path", sort=True):
        with h5py.File(observation_root / observation_path, "r") as handle:
            observed = np.asarray(handle["intensity"])[
                chunk["grid_row_index"].to_numpy(dtype=int)
            ]
        clean = np.empty_like(observed)
        source_rows = sources.loc[chunk["source_xrd_sample_id"]].reset_index(drop=True)
        for clean_path, positions in source_rows.groupby("grid_shard_path").groups.items():
            position_array = np.asarray(list(positions), dtype=int)
            clean_indices = source_rows.loc[position_array, "grid_row_index"].to_numpy(dtype=int)
            with h5py.File(dataset / clean_path, "r") as handle:
                clean_shard = np.asarray(handle["intensity"])
            clean[position_array] = clean_shard[clean_indices]
        metrics = add_noise_distances(clean, observed)
        metrics.insert(0, "parent_group_id", chunk["parent_group_id"].to_numpy())
        metrics.insert(0, "structure_state_id", chunk["structure_state_id"].to_numpy())
        metrics.insert(0, "xrd_sample_id", chunk["xrd_sample_id"].to_numpy())
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def run_calibration(
    dataset: Path,
    output_root: Path,
    *,
    observation_run: str,
    observation_dataset: Path | None = None,
    quantile: float = 0.95,
) -> dict:
    observation_root = observation_dataset or dataset
    observations = pd.read_parquet(
        observation_root / "xrd" / "runs" / observation_run / "xrd_samples.parquet"
    )
    distances = compute_observation_distances(
        dataset, observations, observation_dataset=observation_root
    )
    floors = calibrate_parent_noise_floors(distances, quantile=quantile)
    pairs = pd.read_parquet(dataset / "manifests" / "occupancy_pairs.parquet")
    calibrated = label_pair_identifiability(pairs, floors)
    coverage = audit_parent_noise_coverage(pairs, floors, minimum_variants=5)
    if "oracle_accuracy" not in calibrated:
        calibrated["oracle_accuracy"] = np.nan
    calibrated = assign_identifiability_classes(calibrated)
    support = (
        calibrated.loc[calibrated["noise_identifiable"]]
        .groupby(["id_split", "parent_group_id"])
        .size()
    )
    supported_parents = support.loc[support >= 2].groupby(level=0).size().to_dict()
    summary = {
        "schema_version": "hea-identifiability-calibration-v1.1",
        "status": "diagnostic",
        "release_gate": False,
        "experimental_noise_calibrated": False,
        "noise_reference": "frozen_simulated_perturbations",
        "source_dataset_root": str(dataset),
        "observation_run": observation_run,
        "observation_dataset_root": str(observation_root),
        "noise_quantile": quantile,
        "observation_count": int(len(distances)),
        "parent_group_count": int(floors.shape[0]),
        "global_noise_l1_q95": float(floors["global_noise_l1_q95"].iloc[0]),
        "global_noise_cosine_q95": float(floors["global_noise_cosine_q95"].iloc[0]),
        "pair_count": int(len(calibrated)),
        "identifiable_pair_count": int(calibrated["noise_identifiable"].sum()),
        "ambiguous_pair_count": int((~calibrated["noise_identifiable"]).sum()),
        "identifiable_pair_fraction": float(calibrated["noise_identifiable"].mean()),
        "identifiable_parent_group_count": int(
            calibrated.loc[calibrated["noise_identifiable"], "parent_group_id"].nunique()
        ),
        "identifiability_class_counts": {
            str(key): int(value)
            for key, value in calibrated["identifiability_class"].value_counts().items()
        },
        "oracle_calibrated_pair_count": int(
            calibrated["oracle_accuracy"].notna().sum()
        ),
        "supported_parent_groups_with_two_candidates_by_split": {
            str(key): int(value) for key, value in supported_parents.items()
        },
        **coverage,
        "global_fallback_pair_count": int(
            calibrated["noise_floor_source"].eq("global_fallback").sum()
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(exist_ok=True)
    (output_root / "manifests").mkdir(exist_ok=True)
    distances.to_parquet(output_root / "reports" / "observation_noise_distances.parquet", index=False)
    floors.reset_index().to_parquet(output_root / "reports" / "parent_noise_floors.parquet", index=False)
    calibrated.to_parquet(output_root / "manifests" / "occupancy_pairs_calibrated.parquet", index=False)
    report_path = output_root / "reports" / "identifiability_calibration.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--observation-run", default="hea_observation_pilot_v1")
    parser.add_argument("--observation-dataset", type=Path, default=None)
    parser.add_argument("--quantile", default=0.95, type=float)
    args = parser.parse_args(argv)
    source = args.dataset.resolve()
    output = args.output_root.resolve()
    if source == output or source in output.parents:
        raise ValueError("output root must not be the source dataset or one of its descendants")
    summary = run_calibration(
        source,
        output,
        observation_run=args.observation_run,
        observation_dataset=(
            args.observation_dataset.resolve()
            if args.observation_dataset is not None
            else None
        ),
        quantile=args.quantile,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
