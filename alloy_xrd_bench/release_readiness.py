"""Pre-publication readiness checks for Alloy-XRD-Bench releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .validate import load_benchmark_manifests, validate_benchmark_root


REQUIRED_RELEASE_FILES = (
    "release_summary.json",
    "DATASET_CARD.md",
    "CITATION.cff",
    "croissant.jsonld",
    "file_manifest.parquet",
    "checksums.sha256",
    "audits/license_audit.csv",
    "audits/leakage_audit.csv",
    "audits/quality_summary.json",
    "manifests/xrd_samples.parquet",
    "manifests/private_inputs.parquet",
)

SELF_REFERENTIAL_PACKAGE_FILES = {"file_manifest.parquet", "checksums.sha256"}


def check_release_readiness(
    *,
    release_root: str | Path,
    archive_path: str | Path | None = None,
    publication_record_path: str | Path | None = None,
    require_publication: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready release readiness report."""

    root = Path(release_root).resolve()
    archive = Path(archive_path).resolve() if archive_path else root.with_suffix(".zip")
    issues: list[dict[str, Any]] = []

    validation = validate_benchmark_root(root)
    if validation["issue_count"] != 0:
        issues.append(
            {
                "code": "validation_failed",
                "message": f"Release validator reported {validation['issue_count']} issues",
            }
        )

    for relative_path in REQUIRED_RELEASE_FILES:
        if not (root / relative_path).exists():
            issues.append(
                {
                    "code": "missing_release_file",
                    "message": f"Required release file is missing: {relative_path}",
                    "path": relative_path,
                }
            )

    quality = _quality_summary(root, issues)
    audits = _audit_summary(root, issues)
    manifest = _file_manifest_summary(root, issues)
    archive_summary = _archive_summary(archive, manifest.get("paths", []), issues)
    _check_metadata_only_potential_files_and_notes(root, manifest.get("paths", []), issues)
    publication = _publication_summary(
        publication_record_path=publication_record_path,
        archive=archive,
        require_publication=require_publication,
        issues=issues,
    )

    return {
        "release_root": str(root),
        "archive_path": str(archive),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "validation": {
            "issue_count": validation["issue_count"],
            "manifest_counts": validation["manifest_counts"],
        },
        "quality": quality,
        "audits": audits,
        "file_manifest": {
            "exists": manifest["exists"],
            "file_count": manifest["file_count"],
            "checksum_mismatches": manifest["checksum_mismatches"],
        },
        "archive": archive_summary,
        "publication": publication,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Alloy-XRD-Bench release readiness.")
    parser.add_argument("release_root", help="Release directory to check")
    parser.add_argument("--archive", help="Zip archive path; defaults to <release-root>.zip")
    parser.add_argument("--publication-record", help="Optional publication_record.yml path")
    parser.add_argument("--require-publication", action="store_true", help="Require filled publication metadata")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--output", help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    report = check_release_readiness(
        release_root=args.release_root,
        archive_path=args.archive,
        publication_record_path=args.publication_record,
        require_publication=args.require_publication,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    if args.json:
        print(payload)
    else:
        print(f"Status: {report['status']}")
        print(f"Validation issues: {report['validation']['issue_count']}")
        print(f"Quality issues: {report['quality'].get('issue_count')}")
        print(f"Archive entries: {report['archive'].get('zip_entries')}")
        for issue in report["issues"]:
            print(f"- {issue['code']}: {issue['message']}")

    return 0 if report["status"] == "pass" else 1


def _quality_summary(root: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "audits" / "quality_summary.json"
    if not path.exists():
        return {"exists": False, "issue_count": None, "check_count": None}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append({"code": "invalid_quality_summary", "message": str(exc)})
        return {"exists": True, "issue_count": None, "check_count": None}

    issue_count = int(summary.get("issue_count", -1))
    if issue_count != 0:
        issues.append(
            {
                "code": "quality_audit_failed",
                "message": f"Quality audit reported {issue_count} issues",
            }
        )
    return {
        "exists": True,
        "issue_count": issue_count,
        "check_count": int(summary.get("check_count", 0)),
    }


def _audit_summary(root: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    leakage_path = root / "audits" / "leakage_audit.csv"
    license_path = root / "audits" / "license_audit.csv"
    summary: dict[str, Any] = {
        "license_audit_exists": license_path.exists(),
        "leakage_audit_exists": leakage_path.exists(),
        "leakage_status": None,
        "license_rows": None,
    }

    if leakage_path.exists():
        leakage = pd.read_csv(leakage_path)
        statuses = sorted(str(value) for value in leakage.get("status", []))
        summary["leakage_status"] = "pass" if statuses and set(statuses) == {"pass"} else "fail"
        if summary["leakage_status"] != "pass":
            issues.append(
                {
                    "code": "leakage_audit_failed",
                    "message": "Leakage audit contains non-pass status",
                }
            )
    if license_path.exists():
        summary["license_rows"] = int(len(pd.read_csv(license_path)))
    return summary


def _file_manifest_summary(root: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = root / "file_manifest.parquet"
    checksums_path = root / "checksums.sha256"
    if not manifest_path.exists():
        return {"exists": False, "file_count": 0, "checksum_mismatches": [], "paths": []}

    manifest = pd.read_parquet(manifest_path)
    paths = [str(path) for path in manifest.get("path", [])]
    checksum_rows = _read_checksums(checksums_path) if checksums_path.exists() else {}
    mismatches: list[str] = []

    for row in manifest.to_dict(orient="records"):
        relative_path = str(row["path"])
        artifact = root / relative_path
        if not artifact.exists():
            issues.append(
                {
                    "code": "manifest_artifact_missing",
                    "message": f"Manifest path is missing: {relative_path}",
                    "path": relative_path,
                }
            )
            continue
        if relative_path in SELF_REFERENTIAL_PACKAGE_FILES:
            continue
        expected = str(row.get("sha256", "")).lower()
        actual = _sha256(artifact)
        checksum_file_value = checksum_rows.get(relative_path, "").lower()
        if expected != actual or checksum_file_value != actual:
            mismatches.append(relative_path)

    if mismatches:
        issues.append(
            {
                "code": "checksum_mismatch",
                "message": f"{len(mismatches)} manifest checksum rows do not match files",
            }
        )
    return {
        "exists": True,
        "file_count": int(len(manifest)),
        "checksum_mismatches": mismatches,
        "paths": paths,
    }


def _archive_summary(
    archive: Path,
    manifest_paths: Sequence[str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if not archive.exists():
        issues.append(
            {
                "code": "missing_archive",
                "message": f"Release archive is missing: {archive}",
            }
        )
        return {"exists": False, "zip_entries": 0, "path_mismatches": []}

    with zipfile.ZipFile(archive) as zip_file:
        names = sorted(zip_file.namelist())
    path_mismatches = sorted(set(manifest_paths) ^ set(names))
    if path_mismatches:
        issues.append(
            {
                "code": "archive_manifest_mismatch",
                "message": "Zip entries do not match file_manifest.parquet",
            }
        )
    return {
        "exists": True,
        "zip_entries": len(names),
        "path_mismatches": path_mismatches,
        "size_bytes": archive.stat().st_size,
    }


def _check_metadata_only_potential_files_and_notes(
    root: Path,
    manifest_paths: Sequence[str],
    issues: list[dict[str, Any]],
) -> None:
    manifests, load_issues = load_benchmark_manifests(root)
    if load_issues:
        return
    paths = {path.replace("\\", "/").strip("/") for path in manifest_paths}
    for potential in manifests["potentials_catalog"]:
        policy = str(potential.get("redistribution_policy", ""))
        if policy not in {"metadata_only", "external_only", "not_redistributable", "no_redistribution"}:
            continue
        potential_id = str(potential.get("potential_id", ""))
        forbidden_prefix = f"data/potentials/{potential_id}/potential/files/"
        install_path = f"data/potentials/{potential_id}/potential/INSTALL.md"
        if install_path not in paths:
            issues.append(
                {
                    "code": "missing_metadata_only_install_instructions",
                    "message": f"Metadata-only potential is missing install instructions: {potential_id}",
                    "path": install_path,
                }
            )
        if any(path.startswith(forbidden_prefix) for path in paths):
            issues.append(
                {
                    "code": "nonredistributable_potential_file_packaged",
                    "message": f"Metadata-only potential has packaged model files: {potential_id}",
                }
            )


def _publication_summary(
    *,
    publication_record_path: str | Path | None,
    archive: Path,
    require_publication: bool,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if publication_record_path is None:
        if require_publication:
            issues.append(
                {
                    "code": "missing_publication_record",
                    "message": "--require-publication needs --publication-record",
                }
            )
        return {
            "required": require_publication,
            "record_path": None,
            "exists": False,
            "archive": None,
            "doi": None,
            "dataset_url": None,
            "release_date": None,
            "code_repository_url": None,
            "archive_sha256_matches": None,
        }

    path = Path(publication_record_path).resolve()
    if not path.exists():
        issues.append(
            {
                "code": "missing_publication_record",
                "message": f"Publication record is missing: {path}",
            }
        )
        return {
            "required": require_publication,
            "record_path": str(path),
            "exists": False,
            "archive": None,
            "doi": None,
            "dataset_url": None,
            "release_date": None,
            "code_repository_url": None,
            "archive_sha256_matches": None,
        }

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        issues.append(
            {
                "code": "invalid_publication_record",
                "message": "Publication record must be a YAML mapping",
            }
        )
        payload = {}

    required_fields = (
        "archive",
        "doi",
        "dataset_url",
        "release_date",
        "code_repository_url",
        "release_archive_sha256",
    )
    if require_publication:
        for field_name in required_fields:
            if not str(payload.get(field_name, "")).strip():
                issues.append(
                    {
                        "code": "missing_publication_field",
                        "message": f"Publication record is missing {field_name}",
                        "field": field_name,
                    }
                )

    expected_archive_hash = str(payload.get("release_archive_sha256", "")).strip().lower()
    actual_archive_hash = _sha256(archive) if archive.exists() else ""
    archive_hash_matches = bool(expected_archive_hash and expected_archive_hash == actual_archive_hash)
    if require_publication and expected_archive_hash and not archive_hash_matches:
        issues.append(
            {
                "code": "publication_archive_hash_mismatch",
                "message": "Publication record release_archive_sha256 does not match archive",
            }
        )

    return {
        "required": require_publication,
        "record_path": str(path),
        "exists": True,
        "archive": payload.get("archive"),
        "doi": payload.get("doi"),
        "dataset_url": payload.get("dataset_url"),
        "release_date": payload.get("release_date"),
        "code_repository_url": payload.get("code_repository_url"),
        "archive_sha256_matches": archive_hash_matches,
    }


def _read_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            continue
        digest, relative_path = line.split("  ", 1)
        rows[relative_path.strip()] = digest.strip()
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
