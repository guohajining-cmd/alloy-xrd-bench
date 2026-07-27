"""Supported HEA family-task heads and parent-group support audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


LATTICE_MAP = {
    "fcc_like": "fcc",
    "l12_like": "fcc",
    "bcc_like": "bcc",
    "b2_like": "bcc",
}
ORDERING_MAP = {
    "fcc_like": "disordered",
    "bcc_like": "disordered",
    "l12_like": "l12",
    "b2_like": "b2",
}
RARE_MAP = {"hcp_like": "hcp", "complex": "complex", "unknown": "unknown"}


def lattice_family(label: str) -> str | None:
    return LATTICE_MAP.get(str(label))


def ordering_prototype(label: str) -> str | None:
    return ORDERING_MAP.get(str(label))


def rare_open_set(label: str) -> str | None:
    return RARE_MAP.get(str(label))


def add_family_task_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"structure_family", "parent_group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"family task input missing columns: {sorted(missing)}")
    result = frame.copy()
    result["automated_family_label"] = result["structure_family"].astype(str)
    result["lattice_family"] = result["structure_family"].map(lattice_family)
    result["ordering_prototype"] = result["structure_family"].map(
        ordering_prototype
    )
    result["rare_open_set"] = result["structure_family"].map(rare_open_set)
    return result


def audit_family_support(
    frame: pd.DataFrame,
    *,
    split_column: str,
    minimums: Mapping[str, int],
    pending_review_count: int = 80,
) -> dict[str, object]:
    if split_column not in frame or "lattice_family" not in frame:
        raise ValueError("family support requires split and task-label columns")
    parent_rows = frame[
        ["parent_group_id", split_column, "lattice_family"]
    ].drop_duplicates(["parent_group_id", split_column])
    if parent_rows.groupby("parent_group_id")[split_column].nunique().gt(1).any():
        raise ValueError("parent-group leakage in family support audit")
    eligible = parent_rows.dropna(subset=["lattice_family"])
    table = eligible.groupby([split_column, "lattice_family"]).size()
    counts = {
        split: {
            family: int(table.get((split, family), 0))
            for family in ("fcc", "bcc")
        }
        for split in minimums
    }
    failures = [
        f"{split}:{family}"
        for split, minimum in minimums.items()
        for family in ("fcc", "bcc")
        if counts[split][family] < int(minimum)
    ]
    rare = (
        frame.dropna(subset=["rare_open_set"])
        .drop_duplicates(["parent_group_id", split_column])
        .groupby([split_column, "rare_open_set"])
        .size()
    )
    return {
        "status": "diagnostic",
        "release_gate": False,
        "ground_truth_status": "automated_unadjudicated",
        "pending_review_count": int(pending_review_count),
        "split_column": split_column,
        "counts": counts,
        "minimums": {key: int(value) for key, value in minimums.items()},
        "lattice_family_gate_passed": not failures,
        "family_support_failures": failures,
        "rare_open_set_counts": {
            f"{split}:{label}": int(value)
            for (split, label), value in rare.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-column", default="id_split")
    parser.add_argument("--train-min", type=int, default=500)
    parser.add_argument("--validation-min", type=int, default=100)
    parser.add_argument("--test-min", type=int, default=100)
    parser.add_argument("--pending-review-count", type=int, default=80)
    args = parser.parse_args(argv)
    labelled = add_family_task_labels(pd.read_parquet(args.labels))
    report = audit_family_support(
        labelled,
        split_column=args.split_column,
        minimums={
            "train": args.train_min,
            "validation": args.validation_min,
            "test": args.test_min,
        },
        pending_review_count=args.pending_review_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["lattice_family_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
