"""Derived, non-release data views used by Gamma-Decomp ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import pandas as pd


def _random_assignments(sample_ids: Sequence[str], seed: int) -> dict[str, str]:
    if len(sample_ids) < 3:
        raise ValueError("random split ablation requires at least three samples")
    ordered = sorted(
        (str(value) for value in sample_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    count = len(ordered)
    train_count = max(1, int(count * 0.70))
    validation_count = max(1, int(count * 0.15))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    assignments = {}
    for index, sample_id in enumerate(ordered):
        assignments[sample_id] = (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
    return assignments


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def build_random_split_view(
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 2027,
) -> dict[str, Any]:
    source = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"random split view already exists: {output}")
    output.mkdir(parents=True)
    split_maps: dict[str, dict[str, str]] = {}
    leakage_counts: dict[str, int] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for task in ("full", "window"):
        task_output = output / task
        task_output.mkdir()
        shutil.copytree(
            source / task / "patterns",
            task_output / "patterns",
            copy_function=_hardlink_or_copy,
        )
        samples = pd.read_parquet(source / task / "samples.parquet").copy()
        assignments = _random_assignments(samples["sample_id"].astype(str).tolist(), seed)
        samples["split"] = samples["sample_id"].astype(str).map(assignments)
        samples.to_parquet(task_output / "samples.parquet", index=False)
        components = pd.read_parquet(source / task / "components.parquet").copy()
        components["split"] = components["sample_id"].astype(str).map(assignments)
        if components["split"].isna().any():
            raise ValueError(f"{task} component references an unknown sample")
        components.to_parquet(task_output / "components.parquet", index=False)
        groups = pd.read_parquet(source / task / "evaluation_groups.parquet").copy()
        groups["split"] = groups["sample_id"].astype(str).map(assignments)
        groups.to_parquet(task_output / "evaluation_groups.parquet", index=False)
        parent_split_counts = groups.groupby("parent_pair_id")["split"].nunique()
        leakage_counts[task] = int((parent_split_counts > 1).sum())
        split_maps[task] = assignments
        split_counts[task] = {
            key: int(value) for key, value in samples["split"].value_counts().to_dict().items()
        }

    slices_output = output / "slices"
    slices_output.mkdir()
    all_assignments = {**split_maps["full"], **split_maps["window"]}
    instrument_slices = pd.read_parquet(
        source / "slices" / "instrument_slices.parquet"
    ).copy()
    instrument_slices["split"] = instrument_slices["sample_id"].astype(str).map(
        all_assignments
    )
    if instrument_slices["split"].isna().any():
        raise ValueError("instrument slice references an unknown sample")
    instrument_slices.to_parquet(
        slices_output / "instrument_slices.parquet", index=False
    )

    overlap_slices = pd.read_parquet(source / "slices" / "overlap_slices.parquet").copy()
    overlap_slices["split"] = overlap_slices["sample_id"].astype(str).map(
        split_maps["window"]
    )
    if overlap_slices["split"].isna().any():
        raise ValueError("overlap slice references an unknown window sample")
    overlap_slices.to_parquet(slices_output / "overlap_slices.parquet", index=False)
    shutil.copy2(source / "slices" / "instrument_thresholds.json", slices_output)
    report = {
        "schema_version": "gamma-decomp-random-split-ablation-v1",
        "status": "pass",
        "seed": int(seed),
        "release_source": str(source),
        "split_counts": split_counts,
        "leaking_parent_group_counts": leakage_counts,
        "purpose": "deliberately_leaky_random_split_ablation",
    }
    (output / "ablation_metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    report = build_random_split_view(args.release, args.output, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
