"""Register and initialize per-potential Alloy-XRD-Bench workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import yaml

from .contract import potential_workspace_paths
from .mlip import audit_candidate_registry, download_candidate, verify_potential


REDISTRIBUTABLE_POLICIES = {
    "redistributable",
    "redistributable_with_attribution",
    "redistributable_with_license",
    "redistributable_public_domain",
}


def register_potential(
    *,
    root: str | Path,
    potential_id: str,
    family: str,
    engine: str,
    elements: Sequence[str],
    phases: Sequence[str],
    source_url: str,
    doi: str,
    license_name: str,
    redistribution_policy: str,
    citation_keys: Sequence[str],
    file_path: str | Path | None = None,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a potential workspace and upsert its catalog row."""

    root_path = Path(root).resolve()
    relative_paths = potential_workspace_paths(potential_id)
    workspace_paths = {name: root_path / relative for name, relative in relative_paths.items()}
    _create_workspace_dirs(workspace_paths)

    copied_file_relative: str | None = None
    checksum = file_sha256
    if file_path is not None:
        source = Path(file_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Potential file does not exist: {source}")
        checksum = _sha256_file(source)
        if redistribution_policy in REDISTRIBUTABLE_POLICIES:
            files_dir = workspace_paths["potential_dir"] / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            destination = files_dir / source.name
            if source != destination.resolve():
                shutil.copy2(source, destination)
            copied_file_relative = f"files/{source.name}"

    if not checksum:
        raise ValueError("--file-sha256 is required when --file is not provided")

    row = {
        "potential_id": potential_id,
        "family": family,
        "engine": engine,
        "elements": list(elements),
        "phases": list(phases),
        "source_url": source_url,
        "doi": doi,
        "license": license_name,
        "file_sha256": checksum,
        "redistribution_policy": redistribution_policy,
        "citation_keys": list(citation_keys),
        "potential_dir": relative_paths["potential_dir"],
        "generated_data_dir": relative_paths["generated_data_dir"],
    }
    _write_metadata(workspace_paths["potential_dir"], row, copied_file_relative)
    _write_checksum(workspace_paths["potential_dir"], checksum, copied_file_relative, source_url)
    _write_install_instructions(workspace_paths["potential_dir"], row, copied_file_relative)
    _upsert_catalog(root_path / "data" / "registry" / "potentials_catalog.yml", row)
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Alloy-XRD-Bench potential workspaces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Create or update a potential workspace")
    register.add_argument("--root", default=".", help="Benchmark workspace root")
    register.add_argument("--potential-id", required=True)
    register.add_argument("--family", required=True)
    register.add_argument("--engine", required=True)
    register.add_argument("--elements", nargs="+", required=True)
    register.add_argument("--phases", nargs="+", required=True)
    register.add_argument("--source-url", required=True)
    register.add_argument("--doi", required=True)
    register.add_argument("--license", dest="license_name", required=True)
    register.add_argument("--redistribution-policy", required=True)
    register.add_argument("--citation-key", action="append", dest="citation_keys", required=True)
    register.add_argument("--file")
    register.add_argument("--file-sha256")

    candidates = subparsers.add_parser("candidates", help="Manage MLIP candidate registry")
    candidate_subparsers = candidates.add_subparsers(dest="candidate_command", required=True)
    candidate_audit = candidate_subparsers.add_parser("audit", help="Audit MLIP candidates")
    candidate_audit.add_argument("--root", default=".", help="Benchmark workspace root")
    candidate_audit.add_argument("--json", action="store_true", help="Emit JSON")

    download = subparsers.add_parser("download", help="Report candidate download/install action")
    download.add_argument("--root", default=".", help="Benchmark workspace root")
    download.add_argument("--candidate-id", required=True)
    download.add_argument("--json", action="store_true", help="Emit JSON")

    verify = subparsers.add_parser("verify", help="Verify an accepted MLIP potential")
    verify.add_argument("--root", default=".", help="Benchmark workspace root")
    verify.add_argument("--potential-id", required=True)
    verify.add_argument("--reference-initial-id")
    verify.add_argument("--smoke-initial-id", action="append", dest="smoke_initial_ids")
    verify.add_argument("--json", action="store_true", help="Emit JSON")

    args = parser.parse_args(argv)
    if args.command == "register":
        try:
            row = register_potential(
                root=args.root,
                potential_id=args.potential_id,
                family=args.family,
                engine=args.engine,
                elements=args.elements,
                phases=args.phases,
                source_url=args.source_url,
                doi=args.doi,
                license_name=args.license_name,
                redistribution_policy=args.redistribution_policy,
                citation_keys=args.citation_keys,
                file_path=args.file,
                file_sha256=args.file_sha256,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(yaml.safe_dump(row, sort_keys=False).strip())
        return 0
    if args.command == "candidates" and args.candidate_command == "audit":
        report = audit_candidate_registry(args.root)
        _print_report(report, as_json=args.json)
        return 0 if report["issue_count"] == 0 else 1
    if args.command == "download":
        report = download_candidate(args.root, args.candidate_id)
        _print_report(report, as_json=args.json)
        return 0 if report["status"] != "fail" else 1
    if args.command == "verify":
        report = verify_potential(
            root=args.root,
            potential_id=args.potential_id,
            reference_initial_id=args.reference_initial_id,
            smoke_initial_ids=args.smoke_initial_ids or [],
        )
        _print_report(report, as_json=args.json)
        return 0 if report["status"] == "pass" else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_report(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(report, sort_keys=False).strip())


def _create_workspace_dirs(paths: dict[str, Path]) -> None:
    for key in (
        "potential_dir",
        "manifests_dir",
        "relaxation_logs_dir",
        "relaxed_structures_dir",
        "xrd_grid_dir",
        "xrd_peaks_dir",
        "audits_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)


def _write_metadata(potential_dir: Path, row: dict[str, Any], copied_file_relative: str | None) -> None:
    payload = dict(row)
    payload["packaged_file"] = copied_file_relative
    (potential_dir / "metadata.yml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_checksum(
    potential_dir: Path,
    checksum: str,
    copied_file_relative: str | None,
    source_url: str,
) -> None:
    target = copied_file_relative or f"external:{source_url}"
    (potential_dir / "checksums.sha256").write_text(
        f"{checksum}  {target}\n",
        encoding="utf-8",
    )


def _write_install_instructions(
    potential_dir: Path,
    row: dict[str, Any],
    copied_file_relative: str | None,
) -> None:
    install_path = potential_dir / "INSTALL.md"
    if copied_file_relative is not None:
        if install_path.exists():
            install_path.unlink()
        return

    install_path.write_text(
        "\n".join(
            [
                f"# {row['potential_id']} Installation",
                "",
                "This potential is registered as metadata-only. The benchmark release",
                "does not redistribute the model file.",
                "",
                f"- Source: {row['source_url']}",
                f"- DOI: {row['doi']}",
                f"- License: {row['license']}",
                f"- Expected SHA-256: {row['file_sha256']}",
                "",
                "Install the package or download the model through the upstream source,",
                "then verify that the local model file hash matches the expected SHA-256",
                "before regenerating relaxation data.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _upsert_catalog(catalog_path: Path, row: dict[str, Any]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if catalog_path.exists():
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        rows = payload.get("potentials", [])
    else:
        rows = []

    rows = [dict(existing) for existing in rows if existing.get("potential_id") != row["potential_id"]]
    rows.append(row)
    rows.sort(key=lambda item: item["potential_id"])
    catalog_path.write_text(
        yaml.safe_dump({"potentials": rows}, sort_keys=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
