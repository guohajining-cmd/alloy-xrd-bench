"""Command line validation for Alloy-XRD-Bench workspaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .contract import (
    MANIFEST_SCHEMAS,
    ValidationIssue,
    validate_artifact_hashes,
    validate_manifest_records,
    validate_potential_scoped_artifacts,
    validate_provenance_graph,
    validate_relaxation_quality,
    validate_split_isolation,
    validate_structure_json_hashes,
    validate_xrd_geometry,
)


PER_POTENTIAL_MANIFESTS = (
    "relaxations",
    "relaxed_structures",
    "xrd_samples",
    "provenance_edges",
)


def load_benchmark_manifests(root: str | Path) -> tuple[dict[str, list[dict[str, Any]]], list[ValidationIssue]]:
    """Load manifests from a benchmark workspace or release root."""

    root_path = Path(root).resolve()
    data_root = _data_root_for(root_path)
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in MANIFEST_SCHEMAS}
    issues: list[ValidationIssue] = []

    catalog_path = _first_existing(
        [
            data_root / "registry" / "potentials_catalog.yml",
            data_root / "manifests" / "potentials_catalog.yml",
            root_path / "manifests" / "potentials_catalog.yml",
        ]
    )
    if catalog_path is None:
        issues.append(_missing_manifest("potentials_catalog", data_root / "registry" / "potentials_catalog.yml"))
    else:
        try:
            payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
            rows = payload.get("potentials", payload)
            manifests["potentials_catalog"].extend(_records_from_yaml_rows(rows))
        except Exception as exc:  # pragma: no cover - defensive path, hard to trigger portably
            issues.append(_load_error("potentials_catalog", catalog_path, exc))

    initial_path = _first_existing(
        [
            data_root / "shared" / "manifests" / "initial_structures.parquet",
            data_root / "manifests" / "initial_structures.parquet",
            root_path / "manifests" / "initial_structures.parquet",
        ]
    )
    if initial_path is None:
        issues.append(_missing_manifest("initial_structures", data_root / "shared" / "manifests" / "initial_structures.parquet"))
    else:
        _extend_from_parquet(manifests, issues, "initial_structures", initial_path)

    gamma_pair_path = _first_existing(
        [
            data_root / "shared" / "manifests" / "gamma_prime_pairs.parquet",
            data_root / "manifests" / "gamma_prime_pairs.parquet",
            root_path / "manifests" / "gamma_prime_pairs.parquet",
        ]
    )
    if gamma_pair_path is not None:
        _extend_from_parquet(manifests, issues, "gamma_prime_pairs", gamma_pair_path)

    potential_manifest_dirs = sorted((data_root / "potentials").glob("*/generated/manifests"))
    for manifest_name in PER_POTENTIAL_MANIFESTS:
        found = False
        for manifest_dir in potential_manifest_dirs:
            manifest_path = manifest_dir / f"{manifest_name}.parquet"
            if manifest_path.exists():
                found = True
                _extend_from_parquet(manifests, issues, manifest_name, manifest_path)

        fallback_path = _first_existing(
            [
                data_root / "manifests" / f"{manifest_name}.parquet",
                root_path / "manifests" / f"{manifest_name}.parquet",
            ]
        )
        if fallback_path is not None:
            found = True
            _extend_from_parquet(manifests, issues, manifest_name, fallback_path)

        if not found:
            issues.append(
                _missing_manifest(
                    manifest_name,
                    data_root / "potentials" / "{potential_id}" / "generated" / "manifests" / f"{manifest_name}.parquet",
                )
            )

    return manifests, issues


def validate_benchmark_root(root: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark workspace, returning a JSON-ready report."""

    root_path = Path(root).resolve()
    manifests, issues = load_benchmark_manifests(root_path)
    data_root = _data_root_for(root_path)

    for manifest_name, records in manifests.items():
        issues.extend(validate_manifest_records(manifest_name, records))
    issues.extend(validate_provenance_graph(manifests))
    issues.extend(validate_potential_scoped_artifacts(manifests))
    issues.extend(_validate_registered_potential_directories(data_root, manifests["potentials_catalog"]))
    issues.extend(validate_split_isolation(manifests["xrd_samples"]))
    issues.extend(validate_relaxation_quality(manifests["relaxations"]))
    issues.extend(validate_xrd_geometry(manifests["xrd_samples"]))
    issues.extend(
        validate_structure_json_hashes(
            manifests["initial_structures"],
            root=root_path,
            path_field="artifact_path",
            id_field="initial_structure_id",
            file_hash_field="artifact_file_sha256",
        )
    )
    issues.extend(
        validate_structure_json_hashes(
            manifests["relaxed_structures"],
            root=root_path,
            path_field="artifact_path_json",
            id_field="relaxed_structure_id",
            file_hash_field="artifact_json_file_sha256",
        )
    )
    issues.extend(_validate_artifact_paths_exist(root_path, manifests))

    unique_issues = _deduplicate_issues(issues)
    return {
        "root": str(root_path),
        "issue_count": len(unique_issues),
        "manifest_counts": {name: len(records) for name, records in manifests.items()},
        "issues": [asdict(issue) for issue in unique_issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Alloy-XRD-Bench workspace or release.")
    parser.add_argument("root", help="Benchmark workspace or release root")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_benchmark_root(args.root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Root: {report['root']}")
        print(f"Issues: {report['issue_count']}")
        for name, count in report["manifest_counts"].items():
            print(f"{name}: {count}")
        for issue in report["issues"]:
            location = issue.get("record_id") or issue.get("manifest") or "workspace"
            print(f"- {issue['code']} [{location}]: {issue['message']}")

    return 0 if report["issue_count"] == 0 else 1


def _extend_from_parquet(
    manifests: dict[str, list[dict[str, Any]]],
    issues: list[ValidationIssue],
    manifest_name: str,
    path: Path,
) -> None:
    try:
        rows = pd.read_parquet(path).to_dict(orient="records")
        manifests[manifest_name].extend(rows)
    except Exception as exc:  # pragma: no cover - defensive path, hard to trigger portably
        issues.append(_load_error(manifest_name, path, exc))


def _validate_artifact_paths_exist(
    root: Path,
    manifests: Mapping[str, list[dict[str, Any]]],
) -> list[ValidationIssue]:
    checks = (
        ("relaxations", "relaxation_run_id", "log_path"),
        ("relaxed_structures", "relaxed_structure_id", "artifact_path_cif"),
        ("xrd_samples", "xrd_sample_id", "grid_path"),
        ("xrd_samples", "xrd_sample_id", "peak_path"),
    )
    issues: list[ValidationIssue] = []
    for manifest_name, id_field, path_field in checks:
        for index, record in enumerate(manifests[manifest_name]):
            record_id = "" if record.get(id_field) is None else str(record[id_field])
            artifact_value = "" if record.get(path_field) is None else str(record[path_field])
            artifact_path = (root / artifact_value).resolve()
            if not _is_relative_to(artifact_path, root):
                issues.append(
                    ValidationIssue(
                        code="artifact_path_outside_root",
                        message=f"{path_field} resolves outside validation root: {artifact_value}",
                        manifest=manifest_name,
                        record_index=index,
                        record_id=record_id,
                    )
                )
            elif not artifact_path.exists():
                issues.append(
                    ValidationIssue(
                        code="missing_artifact",
                        message=f"{path_field} is missing: {artifact_value}",
                        manifest=manifest_name,
                        record_index=index,
                        record_id=record_id,
                    )
                )
    return issues


def _validate_registered_potential_directories(
    data_root: Path,
    potentials: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    potential_root = data_root / "potentials"
    if not potential_root.exists():
        return []

    registered_ids = {
        str(row.get("potential_id"))
        for row in potentials
        if row.get("potential_id") not in {None, ""}
    }
    issues: list[ValidationIssue] = []
    for path in sorted(potential_root.iterdir()):
        if not path.is_dir() or path.name in registered_ids:
            continue
        issues.append(
            ValidationIssue(
                code="unregistered_potential_directory",
                message=(
                    f"Potential directory {path.name} exists under data/potentials "
                    "but is not declared in potentials_catalog.yml"
                ),
                manifest="potentials_catalog",
                record_id=path.name,
            )
        )
    return issues


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _data_root_for(root_path: Path) -> Path:
    return root_path / "data" if (root_path / "data").exists() else root_path


def _records_from_yaml_rows(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    if isinstance(rows, dict):
        return [dict(rows)]
    raise TypeError("YAML catalog must be a mapping or list of mappings")


def _missing_manifest(manifest_name: str, expected_path: Path) -> ValidationIssue:
    return ValidationIssue(
        code="missing_manifest",
        message=f"Missing {manifest_name} manifest at {expected_path}",
        manifest=manifest_name,
    )


def _load_error(manifest_name: str, path: Path, exc: Exception) -> ValidationIssue:
    return ValidationIssue(
        code="manifest_load_error",
        message=f"Failed to load {manifest_name} from {path}: {type(exc).__name__}: {exc}",
        manifest=manifest_name,
    )


def _deduplicate_issues(issues: Sequence[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[ValidationIssue] = []
    for issue in issues:
        key = (
            issue.code,
            issue.message,
            issue.manifest,
            issue.record_index,
            issue.record_id,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
