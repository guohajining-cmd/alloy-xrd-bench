"""Generate release audit CSVs for Alloy-XRD-Bench."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .contract import validate_split_isolation
from .validate import load_benchmark_manifests


NON_PACKAGED_POLICIES = {
    "metadata_only",
    "external_only",
    "not_redistributable",
    "no_redistribution",
}


def write_audits(
    *,
    root: str | Path,
    output_dir: str | Path,
    release_version: str = "working",
) -> dict[str, Any]:
    """Write license and leakage audit CSV files."""

    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifests, load_issues = load_benchmark_manifests(root_path)
    license_rows = [_license_audit_row(row) for row in manifests["potentials_catalog"]]
    leakage_issues = validate_split_isolation(manifests["xrd_samples"])
    leakage_rows = [
        {
            "release_version": release_version,
            "split_rule": "private_test_relaxed_structure_isolation",
            "check_name": "private_test_no_released_relaxed_structure_overlap",
            "status": "fail" if leakage_issues else "pass",
            "evidence_path": "manifests/xrd_samples.parquet",
            "notes": "; ".join(issue.message for issue in leakage_issues) if leakage_issues else "",
        }
    ]

    pd.DataFrame(
        license_rows,
        columns=[
            "potential_id",
            "license",
            "redistribution_policy",
            "release_action",
            "source_url",
            "doi",
            "file_sha256",
            "notes",
        ],
    ).to_csv(output_path / "license_audit.csv", index=False)
    pd.DataFrame(
        leakage_rows,
        columns=[
            "release_version",
            "split_rule",
            "check_name",
            "status",
            "evidence_path",
            "notes",
        ],
    ).to_csv(output_path / "leakage_audit.csv", index=False)

    return {
        "license_rows": len(license_rows),
        "leakage_status": leakage_rows[0]["status"],
        "load_issue_count": len(load_issues),
        "output_dir": str(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Alloy-XRD-Bench audit CSVs.")
    parser.add_argument("root", help="Benchmark workspace or release root")
    parser.add_argument("--output-dir", default="audits", help="Directory for audit CSV outputs")
    parser.add_argument("--release-version", default="working")
    args = parser.parse_args(argv)

    report = write_audits(
        root=args.root,
        output_dir=args.output_dir,
        release_version=args.release_version,
    )
    print(f"license_rows: {report['license_rows']}")
    print(f"leakage_status: {report['leakage_status']}")
    print(f"output_dir: {report['output_dir']}")
    return 0 if report["leakage_status"] == "pass" else 1


def _license_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    policy = str(row.get("redistribution_policy", ""))
    if policy in NON_PACKAGED_POLICIES:
        action = "metadata_only"
        notes = "Do not package potential file; provide metadata, source, and checksum."
    else:
        action = "package_file"
        notes = "Redistribution policy allows packaging; verify license text before release."
    return {
        "potential_id": row.get("potential_id", ""),
        "license": row.get("license", ""),
        "redistribution_policy": policy,
        "release_action": action,
        "source_url": row.get("source_url", ""),
        "doi": row.get("doi", ""),
        "file_sha256": row.get("file_sha256", ""),
        "notes": notes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
