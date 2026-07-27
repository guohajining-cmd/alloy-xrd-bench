"""Assign benchmark split labels to XRD manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .validate import load_benchmark_manifests


SUPPORTED_SPLITS = {"id", "composition_ood", "chemistry_ood", "phase_ood", "private_test"}


def apply_splits(
    *,
    root: str | Path,
    config_path: str | Path,
    default_split: str = "id",
) -> dict[str, Any]:
    """Apply split rules to every per-potential xrd_samples manifest."""

    if default_split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported default split: {default_split}")
    root_path = Path(root).resolve()
    config = _load_split_config(Path(config_path))
    manifests, _ = load_benchmark_manifests(root_path)
    contexts = _xrd_contexts(manifests)
    updated = 0
    counts = {split: 0 for split in SUPPORTED_SPLITS}

    data_root = root_path / "data" if (root_path / "data").exists() else root_path
    for manifest_path in sorted((data_root / "potentials").glob("*/generated/manifests/xrd_samples.parquet")):
        rows = pd.read_parquet(manifest_path).to_dict(orient="records")
        for row in rows:
            xrd_id = str(row.get("xrd_sample_id", ""))
            split = _assign_split(contexts.get(xrd_id, {}), config, default_split)
            if row.get("split") != split:
                updated += 1
            row["split"] = split
            counts[split] += 1
        pd.DataFrame(rows).to_parquet(manifest_path)

    return {"updated_rows": updated, "split_counts": counts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign Alloy-XRD-Bench split labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply = subparsers.add_parser("apply", help="Apply split rules to xrd_samples manifests")
    apply.add_argument("--root", default=".", help="Benchmark workspace root")
    apply.add_argument("--config", required=True, help="Split YAML config")
    apply.add_argument("--default-split", default="id", choices=sorted(SUPPORTED_SPLITS))

    args = parser.parse_args(argv)
    if args.command == "apply":
        report = apply_splits(
            root=args.root,
            config_path=args.config,
            default_split=args.default_split,
        )
        print(f"updated_rows: {report['updated_rows']}")
        for split, count in sorted(report["split_counts"].items()):
            print(f"{split}: {count}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _load_split_config(path: Path) -> dict[str, dict[str, set[str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = payload.get("splits", {})
    config: dict[str, dict[str, set[str]]] = {}
    for split, rule in rules.items():
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split {split}")
        config[split] = {
            key: {str(value) for value in values}
            for key, values in (rule or {}).items()
        }
    return config


def _xrd_contexts(manifests: Mapping[str, list[dict[str, Any]]]) -> dict[str, dict[str, str]]:
    initials = {
        str(row["initial_structure_id"]): row
        for row in manifests["initial_structures"]
    }
    relaxations = {
        str(row["relaxation_run_id"]): row
        for row in manifests["relaxations"]
    }
    relaxed = {
        str(row["relaxed_structure_id"]): row
        for row in manifests["relaxed_structures"]
    }
    contexts: dict[str, dict[str, str]] = {}
    for xrd in manifests["xrd_samples"]:
        xrd_id = str(xrd["xrd_sample_id"])
        relaxed_id = str(xrd["relaxed_structure_id"])
        relaxed_row = relaxed.get(relaxed_id, {})
        relaxation_id = str(relaxed_row.get("relaxation_run_id", ""))
        relaxation_row = relaxations.get(relaxation_id, {})
        initial_id = str(relaxation_row.get("initial_structure_id", ""))
        initial_row = initials.get(initial_id, {})
        contexts[xrd_id] = {
            "xrd_sample_id": xrd_id,
            "relaxed_structure_id": relaxed_id,
            "relaxation_run_id": relaxation_id,
            "initial_structure_id": initial_id,
            "alloy_system": str(initial_row.get("alloy_system", "")),
            "phase": str(initial_row.get("phase", "")),
        }
    return contexts


def _assign_split(
    context: Mapping[str, str],
    config: Mapping[str, Mapping[str, set[str]]],
    default_split: str,
) -> str:
    for split, rule in config.items():
        if _rule_matches(context, rule):
            return split
    return default_split


def _rule_matches(context: Mapping[str, str], rule: Mapping[str, set[str]]) -> bool:
    field_map = {
        "xrd_sample_ids": "xrd_sample_id",
        "relaxed_structure_ids": "relaxed_structure_id",
        "relaxation_run_ids": "relaxation_run_id",
        "initial_structure_ids": "initial_structure_id",
        "alloy_systems": "alloy_system",
        "phases": "phase",
    }
    for rule_key, context_key in field_map.items():
        allowed = rule.get(rule_key)
        if allowed and context.get(context_key) in allowed:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
