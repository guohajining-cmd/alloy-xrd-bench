"""Data contracts for pre-release HEA benchmark validation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationRow:
    structure_state_id: str
    parent_group_id: str
    xrd_sample_id: str
    grid_shard_path: str
    grid_row_index: int
    family: str
    composition: np.ndarray
    sro_values: np.ndarray
    sro_mask: np.ndarray
    splits: Mapping[str, str]


def vectorize_sro(
    shell1_json: str,
    shell2_json: str,
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize sparse shell dictionaries while retaining missing-target masks."""
    shells = {
        "shell1": json.loads(shell1_json or "{}"),
        "shell2": json.loads(shell2_json or "{}"),
    }
    values = np.zeros(len(targets), dtype=np.float32)
    mask = np.zeros(len(targets), dtype=bool)
    for index, target in enumerate(targets):
        shell, pair = target.split(":", 1)
        if pair in shells[shell]:
            value = float(shells[shell][pair])
            if not np.isfinite(value):
                raise ValueError(f"non-finite SRO value for {target}")
            values[index] = value
            mask[index] = True
    return values, mask


def _decode_sample_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_grid_row(path: Path, row_index: int, expected_sample_id: str) -> np.ndarray:
    """Read and validate one spectrum from an HDF5 grid shard."""
    with h5py.File(path, "r") as handle:
        count = len(handle["xrd_sample_id"])
        if not 0 <= row_index < count:
            raise IndexError(f"grid row index {row_index} outside [0, {count})")
        actual_id = _decode_sample_id(handle["xrd_sample_id"][row_index])
        if actual_id != expected_sample_id:
            raise ValueError(
                f"sample ID mismatch: expected {expected_sample_id!r}, found {actual_id!r}"
            )
        intensity = np.asarray(handle["intensity"][row_index], dtype=np.float32)

    if intensity.shape != (4001,):
        raise ValueError(f"expected intensity shape (4001,), found {intensity.shape}")
    if not np.isfinite(intensity).all():
        raise ValueError("intensity contains non-finite values")
    if np.any(intensity < 0):
        raise ValueError("intensity contains negative values")
    return intensity


def audit_parent_group_leakage(
    rows: Sequence[ValidationRow], split_name: str
) -> list[str]:
    """Return parent groups assigned to more than one split."""
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if split_name not in row.splits:
            raise KeyError(f"missing split column {split_name!r}")
        assignments[row.parent_group_id].add(row.splits[split_name])
    return sorted(group for group, splits in assignments.items() if len(splits) > 1)


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes dataset root: {relative_path!r}")
    return resolved


def load_validation_rows(
    dataset_root: Path,
    config: Mapping[str, object],
) -> list[ValidationRow]:
    """Load the frozen clean-spectrum manifest and composition features."""
    root = Path(dataset_root).resolve()
    manifest = pd.read_parquet(root / "manifests" / "task_samples.parquet")
    composition = pd.read_parquet(
        root / "manifests" / "composition_only_features.parquet"
    )
    keys = ["structure_state_id", "parent_group_id"]
    if composition.duplicated(keys).any():
        raise ValueError("duplicate composition feature key")
    fraction_columns = [f"fraction_{element}" for element in config["elements"]]
    merged = manifest.merge(
        composition[keys + fraction_columns],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        missing = merged.loc[merged["_merge"] != "both", "structure_state_id"].tolist()
        raise ValueError(f"missing composition features: {missing[:5]}")

    family_mapping = config["family_mapping"]
    targets = config["sro_targets"]
    split_columns = config["split_columns"]
    rows: list[ValidationRow] = []
    for record in merged.to_dict(orient="records"):
        raw_family = record["structure_family"]
        if raw_family not in family_mapping:
            raise ValueError(f"unmapped structure family: {raw_family!r}")
        values, mask = vectorize_sro(
            record["sro_shell1_json"], record["sro_shell2_json"], targets
        )
        composition_values = np.asarray(
            [record[column] for column in fraction_columns], dtype=np.float32
        )
        if not np.isfinite(composition_values).all():
            raise ValueError(
                f"non-finite composition for {record['structure_state_id']!r}"
            )
        rows.append(
            ValidationRow(
                structure_state_id=str(record["structure_state_id"]),
                parent_group_id=str(record["parent_group_id"]),
                xrd_sample_id=str(record["xrd_sample_id"]),
                grid_shard_path=str(
                    _resolve_under_root(root, str(record["grid_shard_path"]))
                ),
                grid_row_index=int(record["grid_row_index"]),
                family=str(family_mapping[raw_family]),
                composition=composition_values,
                sro_values=values,
                sro_mask=mask,
                splits={column: str(record[column]) for column in split_columns},
            )
        )
    return rows
