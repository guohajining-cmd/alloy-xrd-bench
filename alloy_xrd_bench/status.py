"""Workspace status reporting for Alloy-XRD-Bench."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .mlip import adapter_for_engine
from .validate import load_benchmark_manifests, validate_benchmark_root


CORE_PACKAGES = (
    "chgnet",
    "ase",
    "pymatgen",
    "torch",
    "pandas",
    "pyarrow",
    "mace-torch",
    "matgl",
    "mdapy",
    "deepmd-kit",
    "lammps",
)

PER_POTENTIAL_STATUS_MANIFESTS = (
    "relaxations",
    "relaxed_structures",
    "xrd_samples",
    "xrd_averages",
    "provenance_edges",
)


def build_status_report(root: str | Path) -> dict[str, Any]:
    """Return a JSON-ready status report for a workspace or release root."""

    root_path = Path(root).resolve()
    data_root = root_path / "data" if (root_path / "data").exists() else root_path
    manifests, load_issues = load_benchmark_manifests(root_path)
    validation = validate_benchmark_root(root_path)

    registered_ids = sorted(
        str(row.get("potential_id"))
        for row in manifests["potentials_catalog"]
        if row.get("potential_id") not in {None, ""}
    )
    potential_directories = _potential_directories(data_root)

    return {
        "root": str(root_path),
        "validation": {
            "issue_count": validation["issue_count"],
            "manifest_counts": validation["manifest_counts"],
            "load_issue_count": len(load_issues),
        },
        "storage": {
            "potential_directories": potential_directories,
            "registered_potential_ids": registered_ids,
            "unregistered_potential_directories": sorted(set(potential_directories) - set(registered_ids)),
        },
        "structure_line": _structure_line_summary(manifests["initial_structures"]),
        "xrd_line": _xrd_line_summary(manifests["xrd_samples"]),
        "potentials": [
            _potential_status(root_path, row, potential_directories)
            for row in manifests["potentials_catalog"]
        ],
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": _package_versions(),
        },
        "git": _git_status(root_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report Alloy-XRD-Bench workspace status.")
    parser.add_argument("root", help="Benchmark workspace or release root")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--output", help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    report = build_status_report(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    if args.json:
        print(payload)
    else:
        print(f"Root: {report['root']}")
        print(f"Validation issues: {report['validation']['issue_count']}")
        print(f"Potentials: {len(report['potentials'])}")
        for potential in report["potentials"]:
            print(
                "- {potential_id}: engine={engine}, generated={generated_exists}, xrd={xrd_count}".format(
                    potential_id=potential["potential_id"],
                    engine=potential["engine"],
                    generated_exists=potential["generated_data_dir_exists"],
                    xrd_count=potential["manifest_counts"].get("xrd_samples", 0),
                )
            )
    return 0 if report["validation"]["issue_count"] == 0 else 1


def _potential_status(
    root_path: Path,
    row: Mapping[str, Any],
    potential_directories: Sequence[str],
) -> dict[str, Any]:
    potential_id = str(row.get("potential_id", ""))
    potential_dir = root_path / str(row.get("potential_dir", ""))
    generated_dir = root_path / str(row.get("generated_data_dir", ""))
    return {
        "potential_id": potential_id,
        "family": str(row.get("family", "")),
        "engine": str(row.get("engine", "")),
        "redistribution_policy": str(row.get("redistribution_policy", "")),
        "catalog_file_sha256": str(row.get("file_sha256", "")),
        "directory_registered": potential_id in set(potential_directories),
        "potential_dir": str(row.get("potential_dir", "")),
        "potential_dir_exists": potential_dir.exists(),
        "generated_data_dir": str(row.get("generated_data_dir", "")),
        "generated_data_dir_exists": generated_dir.exists(),
        "manifest_counts": _manifest_counts(generated_dir / "manifests"),
        "local_model_files": _known_local_model_files(row),
        "verification": _verification_status(potential_dir.parent / "audits"),
    }


def _manifest_counts(manifest_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest_name in PER_POTENTIAL_STATUS_MANIFESTS:
        path = manifest_dir / f"{manifest_name}.parquet"
        if not path.exists():
            counts[manifest_name] = 0
            continue
        counts[manifest_name] = int(len(pd.read_parquet(path)))
    return counts


def _verification_status(audit_dir: Path) -> dict[str, Any]:
    path = audit_dir / "potential_verification.json"
    if not path.exists():
        return {"exists": False, "status": None, "audit_path": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "exists": True,
            "status": "unreadable",
            "audit_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "exists": True,
        "status": payload.get("status"),
        "audit_path": str(path),
        "load_ok": bool(payload.get("load", {}).get("ok")),
    }


def _known_local_model_files(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        return adapter_for_engine(str(row.get("engine", ""))).model_files(Path.cwd(), row)
    except Exception as exc:  # pragma: no cover - depends on optional package state
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _potential_directories(data_root: Path) -> list[str]:
    potential_root = data_root / "potentials"
    if not potential_root.exists():
        return []
    return sorted(path.name for path in potential_root.iterdir() if path.is_dir())


def _structure_line_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "initial_structures": len(rows),
        "generation_methods": _unique_strings(rows, "generation_method"),
        "candidate_counts": sorted(
            {
                int(row["candidate_count"])
                for row in rows
                if row.get("candidate_count") not in {None, ""}
            }
        ),
    }


def _xrd_line_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    return {
        "xrd_samples": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in CORE_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_status(root_path: Path) -> dict[str, Any]:
    inside = _git_output(root_path, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip().lower() != "true":
        return {
            "is_repository": False,
            "head": None,
            "branch": None,
            "has_uncommitted_changes": None,
            "user_name_configured": False,
            "user_email_configured": False,
        }

    head = _git_output(root_path, "rev-parse", "--short", "HEAD")
    branch = _git_output(root_path, "branch", "--show-current")
    status = _git_output(root_path, "status", "--short")
    user_name = _git_output(root_path, "config", "--get", "user.name")
    user_email = _git_output(root_path, "config", "--get", "user.email")
    return {
        "is_repository": True,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "has_uncommitted_changes": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "user_name_configured": bool(user_name.stdout.strip()) if user_name.returncode == 0 else False,
        "user_email_configured": bool(user_email.stdout.strip()) if user_email.returncode == 0 else False,
    }


def _git_output(root_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _unique_strings(rows: Sequence[Mapping[str, Any]], field_name: str) -> list[str]:
    return sorted(
        {
            str(row.get(field_name))
            for row in rows
            if row.get(field_name) not in {None, ""}
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
