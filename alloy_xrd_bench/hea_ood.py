"""Audit completeness and aggregate HEA OOD evaluator scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_SPLITS = ["composition_ood_split", "chemistry_ood_split", "sro_ood_split"]


def audit_ood_scores(
    scores_root: Path,
    model: str,
    seeds: Sequence[int],
    split_columns: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    rows = []
    missing = []
    invalid = []
    for seed in seeds:
        for split in split_columns:
            path = scores_root / model / str(seed) / f"{split}_test" / "metrics.json"
            if not path.exists():
                missing.append({"seed": int(seed), "split_column": split})
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                mae = float(payload["sro"]["masked_parent_macro_mae"])
            except (KeyError, TypeError, ValueError):
                invalid.append(str(path))
                continue
            if not np.isfinite(mae):
                invalid.append(str(path))
                continue
            rows.append({"seed": int(seed), "split_column": split, "sro_parent_macro_mae": mae})
    aggregates = {}
    for split in split_columns:
        values = np.asarray(
            [row["sro_parent_macro_mae"] for row in rows if row["split_column"] == split],
            dtype=float,
        )
        if values.size:
            aggregates[split] = {
                "seed_count": int(values.size),
                "sro_parent_macro_mae_mean": float(values.mean()),
                "sro_parent_macro_mae_std": float(values.std()),
            }
    return {
        "schema_version": "sitexrd-hea-ood-audit-v1",
        "status": "pass" if not missing and not invalid else "fail",
        "model": model,
        "required_seeds": [int(seed) for seed in seeds],
        "required_split_columns": list(split_columns),
        "evaluated_run_count": len(rows),
        "missing_runs": missing,
        "invalid_runs": invalid,
        "metrics_by_split": aggregates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--split-columns", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_ood_scores(
        args.scores_root.resolve(), args.model, args.seeds, args.split_columns
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
