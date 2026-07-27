"""MLIP dataset registry, download, and structure-first ingestion CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml
from pymatgen.core import Element, Lattice, Structure

from .contract import canonical_structure_hash


DATASET_REGISTRY_PATH = Path("data/registry/mlip_datasets.yml")
DATASET_FILE_COLUMNS = [
    "dataset_id",
    "source_file_id",
    "filename",
    "role",
    "source_url",
    "fairdata_path",
    "storage_tier",
    "path",
    "sha256",
    "bytes",
    "license",
    "redistribution_policy",
]
DATASET_STRUCTURE_COLUMNS = [
    "dataset_structure_id",
    "dataset_id",
    "source_file_id",
    "split",
    "structure_index",
    "elements",
    "natoms",
    "phase_hint",
    "artifact_path_json",
    "structure_sha256",
    "has_energy",
    "has_forces",
    "has_stress",
    "xrd_eligible",
    "quality_notes",
]
REQUIRED_DATASET_FIELDS = [
    "dataset_id",
    "source_url",
    "doi",
    "license",
    "redistribution_policy",
    "target_systems",
    "formats",
    "local_paths",
    "related_potential_ids",
    "status",
]


@dataclass
class DatasetParseError(Exception):
    """A structured parser failure that can be written to parse_report.json."""

    code: str
    message: str


def audit_dataset_registry(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    datasets, load_issues = _load_dataset_registry(root_path)
    issues: list[dict[str, Any]] = list(load_issues)
    seen: dict[str, int] = {}
    for index, row in enumerate(datasets):
        dataset_id = _string(row.get("dataset_id"))
        if dataset_id:
            if dataset_id in seen:
                issues.append(
                    {
                        "code": "duplicate_dataset_id",
                        "message": f"Duplicate dataset_id {dataset_id}",
                        "record_index": index,
                        "record_id": dataset_id,
                    }
                )
            seen[dataset_id] = index
        for field in REQUIRED_DATASET_FIELDS:
            if _missing_required_dataset_field(row, field):
                issues.append(
                    {
                        "code": "missing_required_field",
                        "message": f"Dataset {dataset_id or index} is missing required field {field}",
                        "record_index": index,
                        "record_id": dataset_id or None,
                        "field": field,
                    }
                )
        if dataset_id and not _is_redistributable(row):
            public_raw = root_path / "data" / "mlip_datasets" / dataset_id / "raw"
            if public_raw.exists() and any(public_raw.rglob("*")):
                issues.append(
                    {
                        "code": "license_storage_violation",
                        "message": f"Dataset {dataset_id} has non-redistributable raw files in data/mlip_datasets",
                        "record_index": index,
                        "record_id": dataset_id,
                    }
                )

    return {
        "status": "pass" if not issues else "fail",
        "dataset_count": len(datasets),
        "issue_count": len(issues),
        "issues": issues,
        "datasets": [
            {
                "dataset_id": row.get("dataset_id"),
                "license": row.get("license"),
                "redistribution_policy": row.get("redistribution_policy"),
                "status": row.get("status"),
                "storage_tier": _storage_tier(row),
            }
            for row in datasets
        ],
    }


def download_dataset(dataset_id: str, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    row = _require_dataset(root_path, dataset_id)
    raw_files = list(row.get("raw_files") or [])
    fairdata_inventory: list[dict[str, Any]] = []
    if not raw_files and _fairdata_dataset_id(row):
        fairdata_inventory = _fairdata_file_inventory(row)
        raw_files = _fairdata_selected_raw_files(row, fairdata_inventory)
    if not raw_files:
        return {
            "status": "blocked",
            "dataset_id": dataset_id,
            "reason": "no_downloadable_files",
            "storage_tier": _storage_tier(row),
            "files": [],
        }

    workspace = _dataset_workspace(root_path, row)
    raw_dir = workspace / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if fairdata_inventory:
        _write_fairdata_inventory(workspace, fairdata_inventory)
    files: list[dict[str, Any]] = []
    for index, spec in enumerate(raw_files):
        target_relative = _safe_relative_path(_string(spec.get("relative_path") or spec.get("filename") or spec.get("name") or f"file_{index:04d}"))
        target = raw_dir / target_relative
        source_url = _string(spec.get("url"))
        local_path = _string(spec.get("local_path"))
        if local_path:
            source_path = (root_path / local_path).resolve()
            if not source_path.exists():
                files.append(_download_failure(row, spec, target, "missing_local_path", f"{local_path} does not exist"))
                continue
            if source_path.resolve() != target.resolve():
                shutil.copy2(source_path, target)
        elif source_url:
            try:
                _download_url(source_url, target)
            except Exception as exc:  # pragma: no cover - network failures are recorded by design.
                files.append(_download_failure(row, spec, target, "download_failed", str(exc)))
                continue
        elif spec.get("fairdata_path"):
            try:
                authorized_url = _authorize_fairdata_download(row, _string(spec["fairdata_path"]))
                _download_url(authorized_url, target)
                time.sleep(0.1)
            except Exception as exc:  # pragma: no cover - network failures are recorded by design.
                files.append(_download_failure(row, spec, target, "fairdata_download_failed", str(exc)))
                continue
        else:
            files.append(_download_failure(row, spec, target, "missing_source", "raw_file has no url or local_path"))
            continue

        files.append(_file_manifest_row(root_path, row, target, spec))

    _write_dataset_metadata(root_path, row, workspace)
    _write_checksums(workspace / "checksums.sha256", files)
    _write_files_manifest(workspace / "manifests" / "files.parquet", files)
    audit = {
        "status": "pass" if all(file.get("status", "pass") == "pass" for file in files) else "partial",
        "dataset_id": dataset_id,
        "storage_tier": _storage_tier(row),
        "workspace": _relative_path(root_path, workspace),
        "files": files,
    }
    (workspace / "download_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def ingest_dataset(dataset_id: str, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    row = _require_dataset(root_path, dataset_id)
    workspace = _dataset_workspace(root_path, row)
    raw_dir = _raw_dir_for_ingest(root_path, row)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "manifests").mkdir(parents=True, exist_ok=True)
    (workspace / "structures" / "json").mkdir(parents=True, exist_ok=True)
    _write_dataset_metadata(root_path, row, workspace)

    raw_files = sorted(path for path in raw_dir.rglob("*") if path.is_file()) if raw_dir.exists() else []
    file_rows = [_file_manifest_row(root_path, row, path, _raw_spec_for_path(row, path, raw_dir)) for path in raw_files]
    _write_files_manifest(workspace / "manifests" / "files.parquet", file_rows)
    _write_checksums(workspace / "checksums.sha256", file_rows)

    structure_rows: list[dict[str, Any]] = []
    parse_files: list[dict[str, Any]] = []
    for file_row, path in zip(file_rows, raw_files, strict=False):
        try:
            parsed = _parse_structure_file(root_path, workspace, row, file_row, path)
        except DatasetParseError as exc:
            parse_files.append(
                {
                    "path": file_row["path"],
                    "status": "failed",
                    "code": exc.code,
                    "message": exc.message,
                    "structures": 0,
                }
            )
            continue
        structure_rows.extend(parsed)
        parse_files.append(
            {
                "path": file_row["path"],
                "status": "parsed" if parsed else "skipped",
                "code": "parsed" if parsed else _skip_code(path),
                "message": f"{len(parsed)} structures parsed" if parsed else "No supported structures parsed",
                "structures": len(parsed),
            }
        )

    _write_structures_manifest(workspace / "manifests" / "structures.parquet", structure_rows)
    status = _parse_status(parse_files, structure_rows)
    parse_report = {
        "dataset_id": dataset_id,
        "status": status,
        "raw_dir": _relative_path(root_path, raw_dir) if raw_dir.exists() else "",
        "file_count": len(file_rows),
        "structure_count": len(structure_rows),
        "files": parse_files,
    }
    (workspace / "manifests" / "parse_report.json").write_text(
        json.dumps(parse_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return parse_report


def dataset_status(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    datasets, load_issues = _load_dataset_registry(root_path)
    rows: list[dict[str, Any]] = []
    for row in datasets:
        dataset_id = _string(row.get("dataset_id"))
        public_workspace = root_path / "data" / "mlip_datasets" / dataset_id
        local_workspace = root_path / "local_dataset_cache" / dataset_id
        workspace = public_workspace if public_workspace.exists() else local_workspace
        files_path = workspace / "manifests" / "files.parquet"
        structures_path = workspace / "manifests" / "structures.parquet"
        rows.append(
            {
                "dataset_id": dataset_id,
                "license": row.get("license"),
                "redistribution_policy": row.get("redistribution_policy"),
                "storage_tier": _storage_tier(row),
                "workspace": _relative_path(root_path, workspace) if workspace.exists() else "",
                "raw_file_count": _count_manifest_rows(files_path),
                "structure_count": _count_manifest_rows(structures_path),
                "status": row.get("status"),
            }
        )
    return {
        "status": "pass" if not load_issues else "fail",
        "dataset_count": len(datasets),
        "issues": load_issues,
        "datasets": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit mlip_datasets.yml and safe storage policy")
    _add_common_args(audit_parser)

    download_parser = subparsers.add_parser("download", help="Download or copy raw dataset files")
    download_parser.add_argument("--dataset-id", required=True)
    _add_common_args(download_parser)

    ingest_parser = subparsers.add_parser("ingest", help="Build files and structures manifests")
    ingest_parser.add_argument("--dataset-id", required=True)
    _add_common_args(ingest_parser)

    status_parser = subparsers.add_parser("status", help="Report dataset corpus status")
    _add_common_args(status_parser)

    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_dataset_registry(args.root)
        elif args.command == "download":
            report = download_dataset(args.dataset_id, args.root)
        elif args.command == "ingest":
            report = ingest_dataset(args.dataset_id, args.root)
        elif args.command == "status":
            report = dataset_status(args.root)
        else:  # pragma: no cover - argparse prevents this.
            parser.error(f"unknown command {args.command}")
    except DatasetParseError as exc:
        report = {"status": "fail", "issue_count": 1, "issues": [{"code": exc.code, "message": exc.message}]}
    except Exception as exc:
        report = {"status": "fail", "issue_count": 1, "issues": [{"code": "unexpected_error", "message": str(exc)}]}

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.get("status") == "pass" else 1


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")


def _load_dataset_registry(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = root / DATASET_REGISTRY_PATH
    if not path.exists():
        return [], [{"code": "missing_dataset_registry", "message": f"Missing {DATASET_REGISTRY_PATH.as_posix()}"}]
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [], [{"code": "dataset_registry_load_error", "message": str(exc)}]
    rows = payload.get("datasets", payload)
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return [], [{"code": "invalid_dataset_registry", "message": "Registry must contain a datasets list"}]
    return [dict(row or {}) for row in rows], []


def _require_dataset(root: Path, dataset_id: str) -> dict[str, Any]:
    datasets, issues = _load_dataset_registry(root)
    if issues:
        raise DatasetParseError("dataset_registry_error", "; ".join(issue["message"] for issue in issues))
    for row in datasets:
        if row.get("dataset_id") == dataset_id:
            return row
    raise DatasetParseError("unknown_dataset_id", f"Dataset {dataset_id} is not declared in mlip_datasets.yml")


def _is_redistributable(row: Mapping[str, Any]) -> bool:
    license_text = _string(row.get("license")).lower()
    policy = _string(row.get("redistribution_policy")).lower()
    if any(token in policy for token in ("metadata_only", "request", "restricted", "unknown")):
        return False
    return any(token in license_text for token in ("cc-by", "cc0", "mit", "bsd", "apache", "public domain"))


def _storage_tier(row: Mapping[str, Any]) -> str:
    return "public_release_raw" if _is_redistributable(row) else "local_cache_only"


def _dataset_workspace(root: Path, row: Mapping[str, Any]) -> Path:
    dataset_id = _string(row.get("dataset_id"))
    if _is_redistributable(row):
        return root / "data" / "mlip_datasets" / dataset_id
    return root / "local_dataset_cache" / dataset_id


def _raw_dir_for_ingest(root: Path, row: Mapping[str, Any]) -> Path:
    dataset_id = _string(row.get("dataset_id"))
    public_raw = root / "data" / "mlip_datasets" / dataset_id / "raw"
    local_raw = root / "local_dataset_cache" / dataset_id / "raw"
    if public_raw.exists():
        return public_raw
    return local_raw


def _download_url(source_url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        shutil.copy2(source, target)
        return
    with urllib.request.urlopen(source_url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _fairdata_dataset_id(row: Mapping[str, Any]) -> str:
    if row.get("fairdata_dataset_id"):
        return _string(row["fairdata_dataset_id"])
    source_url = _string(row.get("source_url"))
    parsed = urllib.parse.urlparse(source_url)
    marker = "/dataset/"
    if marker in parsed.path:
        return parsed.path.rsplit(marker, 1)[-1].strip("/")
    return ""


def _fairdata_file_inventory(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_uuid = _fairdata_dataset_id(row)
    if not dataset_uuid:
        return []
    url = f"https://metax.fairdata.fi/v3/datasets/{dataset_uuid}/files?pagination=false"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return [dict(item) for item in payload]


def _fairdata_selected_raw_files(
    row: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in inventory:
        pathname = _string(item.get("pathname"))
        if not _is_fairdata_dataset_artifact(pathname):
            continue
        relative = _fairdata_relative_path(pathname)
        checksum = _string(item.get("checksum"))
        selected.append(
            {
                "filename": Path(relative).name,
                "relative_path": relative,
                "fairdata_path": pathname,
                "role": _infer_role(pathname),
                "expected_bytes": item.get("size"),
                "expected_sha256": checksum.removeprefix("sha256:") if checksum.startswith("sha256:") else checksum,
                "source_file_id": _fairdata_source_file_id(row, relative),
            }
        )
    return selected


def _is_fairdata_dataset_artifact(pathname: str) -> bool:
    lowered = pathname.lower()
    if "/tabgap_potential_files/" in lowered or "/train_test_errors/" in lowered:
        return False
    return "/data/" in lowered or "/input/" in lowered or lowered.endswith("/readme")


def _fairdata_relative_path(pathname: str) -> str:
    parts = [part for part in pathname.strip("/").split("/") if part]
    if len(parts) > 1:
        parts = parts[1:]
    return "/".join(parts)


def _fairdata_source_file_id(row: Mapping[str, Any], relative_path: str) -> str:
    safe = relative_path.replace("/", "_").replace(".", "_").replace("-", "_").replace("+", "p")
    return f"{row.get('dataset_id')}_{safe}"


def _write_fairdata_inventory(workspace: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    manifests = workspace / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": item.get("id"),
            "storage_identifier": item.get("storage_identifier"),
            "pathname": item.get("pathname"),
            "filename": item.get("filename"),
            "size": item.get("size"),
            "checksum": item.get("checksum"),
            "storage_service": item.get("storage_service"),
            "csc_project": item.get("csc_project"),
        }
        for item in inventory
    ]
    pd.DataFrame(rows).to_parquet(manifests / "fairdata_file_inventory.parquet")
    (manifests / "fairdata_file_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _authorize_fairdata_download(row: Mapping[str, Any], fairdata_path: str) -> str:
    dataset_uuid = _fairdata_dataset_id(row)
    payload = {"cr_id": dataset_uuid, "file": fairdata_path}
    request = urllib.request.Request(
        "https://etsin.fairdata.fi/api/v3/download/authorize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    if not data.get("url"):
        raise RuntimeError(f"Fairdata authorize did not return a URL for {fairdata_path}")
    return _string(data["url"])


def _download_failure(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    target: Path,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "code": code,
        "message": message,
        "dataset_id": row.get("dataset_id"),
        "source_file_id": _source_file_id(row, spec, target.name),
        "filename": target.name,
        "role": spec.get("role", ""),
        "source_url": spec.get("url", ""),
        "fairdata_path": spec.get("fairdata_path", ""),
        "storage_tier": _storage_tier(row),
        "path": "",
        "sha256": "",
        "bytes": 0,
        "license": row.get("license", ""),
        "redistribution_policy": row.get("redistribution_policy", ""),
    }


def _file_manifest_row(root: Path, row: Mapping[str, Any], path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    sha256 = _sha256_file(path)
    return {
        "status": "pass",
        "dataset_id": row.get("dataset_id"),
        "source_file_id": _source_file_id(row, spec, path.name),
        "filename": path.name,
        "role": spec.get("role", _infer_role(path.name)),
        "source_url": spec.get("url", ""),
        "fairdata_path": spec.get("fairdata_path", ""),
        "storage_tier": _storage_tier(row),
        "path": _relative_path(root, path),
        "sha256": sha256,
        "bytes": path.stat().st_size,
        "license": row.get("license", ""),
        "redistribution_policy": row.get("redistribution_policy", ""),
    }


def _write_dataset_metadata(root: Path, row: Mapping[str, Any], workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in row.items() if key != "raw_files"}
    metadata["storage_tier"] = _storage_tier(row)
    metadata["workspace"] = _relative_path(root, workspace)
    (workspace / "metadata.yml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def _write_checksums(path: Path, files: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in files:
        if row.get("sha256") and row.get("path"):
            lines.append(f"{row['sha256']}  {row['path']}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_files_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DATASET_FILE_COLUMNS).to_parquet(path)


def _write_structures_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DATASET_STRUCTURE_COLUMNS).to_parquet(path)


def _parse_structure_file(
    root: Path,
    workspace: Path,
    row: Mapping[str, Any],
    file_row: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar.bz2"):
        _write_archive_inventory(workspace, file_row, path)
        return []
    if path.suffix.lower() == ".zip":
        _write_archive_inventory(workspace, file_row, path)
        return []
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        return _parse_extxyz(root, workspace, row, file_row, path)
    if path.suffix.lower() == ".cfg":
        return _parse_mtp_cfg(root, workspace, row, file_row, path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _parse_lightpfp_h5(root, workspace, row, file_row, path)
    return []


def _parse_extxyz(
    root: Path,
    workspace: Path,
    row: Mapping[str, Any],
    file_row: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    try:
        from ase.io import read
    except ImportError as exc:
        raise DatasetParseError("missing_framework", "ASE is required to parse xyz/extxyz files") from exc
    try:
        images = read(str(path), index=":")
    except Exception as exc:
        raise DatasetParseError("parse_failed", f"ASE could not read {path.name}: {exc}") from exc
    if not isinstance(images, list):
        images = [images]
    rows: list[dict[str, Any]] = []
    for index, atoms in enumerate(images):
        cell = atoms.get_cell().array
        if abs(float(atoms.get_volume())) <= 0:
            structure = None
            doc = None
        else:
            structure = Structure(Lattice(cell), atoms.get_chemical_symbols(), atoms.get_positions(), coords_are_cartesian=True)
            doc = structure.as_dict()
        rows.append(
            _structure_row(
                root=root,
                workspace=workspace,
                row=row,
                file_row=file_row,
                structure=structure,
                structure_doc=doc,
                index=index,
                has_energy=_atoms_has_energy(atoms),
                has_forces=_atoms_has_forces(atoms),
                has_stress=_atoms_has_stress(atoms),
            )
        )
    return rows


def _parse_lightpfp_h5(
    root: Path,
    workspace: Path,
    row: Mapping[str, Any],
    file_row: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise DatasetParseError("missing_framework", "h5py is required to parse LightPFP HDF5 files") from exc
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        groups = _h5_structure_groups(handle)
        for index, group in enumerate(groups):
            if not all(key in group for key in ("cell", "positions", "numbers")):
                continue
            cell = group["cell"][()]
            positions = group["positions"][()]
            numbers = group["numbers"][()]
            species = [Element.from_Z(int(number)).symbol for number in numbers]
            structure = Structure(Lattice(cell), species, positions, coords_are_cartesian=True)
            rows.append(
                _structure_row(
                    root=root,
                    workspace=workspace,
                    row=row,
                    file_row=file_row,
                    structure=structure,
                    structure_doc=structure.as_dict(),
                    index=index,
                    has_energy="potential_energy" in group or "energy" in group,
                    has_forces="forces" in group,
                    has_stress="stress" in group,
                )
            )
    return rows


def _parse_mtp_cfg(
    root: Path,
    workspace: Path,
    row: Mapping[str, Any],
    file_row: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    type_map = row.get("type_map") or row.get("element_type_map")
    if not type_map:
        raise DatasetParseError("missing_type_map", f"{path.name} requires type_map in mlip_datasets.yml")
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = _cfg_blocks(text)
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        structure = _cfg_block_to_structure(block, type_map)
        rows.append(
            _structure_row(
                root=root,
                workspace=workspace,
                row=row,
                file_row=file_row,
                structure=structure,
                structure_doc=structure.as_dict(),
                index=index,
                has_energy=_cfg_block_has_section(block, "Energy"),
                has_forces=_cfg_block_has_forces(block),
                has_stress=_cfg_block_has_section(block, "PlusStress"),
            )
        )
    return rows


def _structure_row(
    *,
    root: Path,
    workspace: Path,
    row: Mapping[str, Any],
    file_row: Mapping[str, Any],
    structure: Structure | None,
    structure_doc: Mapping[str, Any] | None,
    index: int,
    has_energy: bool,
    has_forces: bool,
    has_stress: bool,
) -> dict[str, Any]:
    dataset_id = _string(row.get("dataset_id"))
    source_file_id = _string(file_row.get("source_file_id"))
    dataset_structure_id = f"{dataset_id}_{source_file_id}_s{index:06d}"
    artifact_relative = _relative_path(root, workspace / "structures" / "json" / f"{dataset_structure_id}.json")
    quality_notes: list[str] = []
    if structure is None or structure_doc is None:
        elements: list[str] = []
        natoms = 0
        structure_hash = ""
        xrd_eligible = False
        quality_notes.append("missing_lattice_or_coordinates")
    else:
        elements = sorted({str(element) for element in structure.composition.elements})
        natoms = len(structure)
        structure_hash = canonical_structure_hash(structure_doc)
        xrd_eligible, quality_notes = _xrd_eligibility(structure)
        artifact_path = root / artifact_relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(structure_doc, indent=2, sort_keys=True), encoding="utf-8")
    if not has_energy:
        quality_notes.append("missing_energy_label")
    if not has_forces:
        quality_notes.append("missing_forces_label")
    if not has_stress:
        quality_notes.append("missing_stress_label")
    return {
        "dataset_structure_id": dataset_structure_id,
        "dataset_id": dataset_id,
        "source_file_id": source_file_id,
        "split": _split_from_role(_string(file_row.get("role"))),
        "structure_index": index,
        "elements": elements,
        "natoms": natoms,
        "phase_hint": row.get("phase_hint", ""),
        "artifact_path_json": artifact_relative if structure_doc is not None else "",
        "structure_sha256": structure_hash,
        "has_energy": bool(has_energy),
        "has_forces": bool(has_forces),
        "has_stress": bool(has_stress),
        "xrd_eligible": bool(xrd_eligible),
        "quality_notes": ";".join(dict.fromkeys(quality_notes)),
    }


def _xrd_eligibility(structure: Structure) -> tuple[bool, list[str]]:
    notes: list[str] = ["raw_training_structure"]
    if len(structure) <= 0:
        notes.append("empty_structure")
    if structure.volume <= 0:
        notes.append("nonpositive_volume")
    for site in structure:
        try:
            Element(str(site.specie))
        except Exception:
            notes.append("unrecognized_element")
            break
    blocking = {"empty_structure", "nonpositive_volume", "unrecognized_element"}
    return not any(note in blocking for note in notes), notes


def _cfg_blocks(lines: Sequence[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "BEGIN_CFG":
            current = [line]
            in_block = True
            continue
        if in_block:
            current.append(line)
            if stripped == "END_CFG":
                blocks.append(current)
                in_block = False
    return blocks


def _cfg_block_to_structure(block: Sequence[str], type_map: Mapping[Any, Any]) -> Structure:
    lattice: list[list[float]] = []
    species: list[str] = []
    coords: list[list[float]] = []
    index = 0
    while index < len(block):
        token = block[index].strip()
        if token == "Supercell":
            lattice = [_float_tokens(block[index + offset])[:3] for offset in (1, 2, 3)]
            index += 4
            continue
        if token.startswith("AtomData:"):
            header = token.split(":", 1)[1].split()
            column = {name: pos for pos, name in enumerate(header)}
            index += 1
            while index < len(block) and block[index].strip() and not _cfg_section_header(block[index]):
                parts = block[index].split()
                atom_type = parts[column["type"]]
                symbol = type_map.get(str(atom_type), type_map.get(int(atom_type) if atom_type.isdigit() else atom_type))
                if not symbol:
                    raise DatasetParseError("unknown_type_map_entry", f"No symbol for cfg type {atom_type}")
                species.append(str(symbol))
                coords.append(
                    [
                        float(parts[column["cartes_x"]]),
                        float(parts[column["cartes_y"]]),
                        float(parts[column["cartes_z"]]),
                    ]
                )
                index += 1
            continue
        index += 1
    if not lattice or not coords:
        raise DatasetParseError("parse_failed", "CFG block is missing Supercell or AtomData")
    return Structure(Lattice(lattice), species, coords, coords_are_cartesian=True)


def _cfg_section_header(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (stripped[0].isalpha() or stripped == "END_CFG")


def _cfg_block_has_section(block: Sequence[str], section: str) -> bool:
    return any(
        (token := line.strip()) == section or token.startswith(f"{section}:")
        for line in block
    )


def _cfg_block_has_forces(block: Sequence[str]) -> bool:
    return any(" fx " in f" {line.strip()} " and " fy " in f" {line.strip()} " and " fz " in f" {line.strip()} " for line in block)


def _h5_structure_groups(handle: Any) -> list[Any]:
    groups = []
    for key in handle.keys():
        item = handle[key]
        if hasattr(item, "keys"):
            groups.append(item)
    if groups:
        return groups
    return [handle]


def _atoms_has_energy(atoms: Any) -> bool:
    if "energy" in getattr(atoms, "info", {}):
        return True
    try:
        atoms.get_potential_energy()
        return True
    except Exception:
        return False


def _atoms_has_forces(atoms: Any) -> bool:
    if "forces" in getattr(atoms, "arrays", {}):
        return True
    try:
        atoms.get_forces()
        return True
    except Exception:
        return False


def _atoms_has_stress(atoms: Any) -> bool:
    if "stress" in getattr(atoms, "info", {}):
        return True
    try:
        atoms.get_stress()
        return True
    except Exception:
        return False


def _write_archive_inventory(workspace: Path, file_row: Mapping[str, Any], path: Path) -> None:
    inventory_dir = workspace / "manifests" / "archive_inventories"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    output = inventory_dir / f"{file_row['source_file_id']}.json"
    entries: list[dict[str, Any]] = []
    suffixes = "".join(path.suffixes).lower()
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                entries = [{"name": item.filename, "bytes": item.file_size} for item in archive.infolist()]
        elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar.bz2"):
            with tarfile.open(path) as archive:
                entries = [
                    {"name": item.name, "bytes": item.size, "kind": "file" if item.isfile() else "directory"}
                    for item in archive.getmembers()
                ]
    except Exception as exc:
        entries = [{"name": "", "bytes": 0, "error": str(exc)}]
    output.write_text(json.dumps({"archive": file_row["path"], "entries": entries}, indent=2), encoding="utf-8")


def _source_file_id(row: Mapping[str, Any], spec: Mapping[str, Any], filename: str) -> str:
    if spec.get("source_file_id"):
        return _string(spec["source_file_id"])
    stem = filename.replace(".", "_").replace("-", "_").replace(" ", "_")
    return f"{row.get('dataset_id')}_{stem}"


def _raw_spec_for_path(row: Mapping[str, Any], path: Path, raw_dir: Path | None = None) -> Mapping[str, Any]:
    for spec in row.get("raw_files") or []:
        if _string(spec.get("filename") or spec.get("name")) == path.name:
            return spec
    if _fairdata_dataset_id(row) and raw_dir is not None:
        try:
            relative = path.relative_to(raw_dir).as_posix()
        except ValueError:
            relative = path.name
        fairdata_path = _fairdata_path_from_inventory(raw_dir.parent, relative)
        return {
            "filename": path.name,
            "relative_path": relative,
            "fairdata_path": fairdata_path,
            "role": _infer_role(relative),
            "source_file_id": _fairdata_source_file_id(row, relative),
        }
    return {"filename": path.name, "role": _infer_role(path.name)}


def _fairdata_path_from_inventory(workspace: Path, relative: str) -> str:
    inventory_path = workspace / "manifests" / "fairdata_file_inventory.parquet"
    if not inventory_path.exists():
        return ""
    try:
        rows = pd.read_parquet(inventory_path).to_dict(orient="records")
    except Exception:
        return ""
    suffix = "/" + relative.replace("\\", "/")
    for row in rows:
        pathname = _string(row.get("pathname"))
        if _fairdata_relative_path(pathname) == relative or pathname.endswith(suffix):
            return pathname
    return ""


def _infer_role(filename: str) -> str:
    lower = filename.lower()
    if "test" in lower or "valid" in lower:
        return "validation_or_test"
    if "train" in lower or "db_" in lower:
        return "training"
    return "raw_artifact"


def _split_from_role(role: str) -> str:
    lower = role.lower()
    if "valid" in lower:
        return "validation"
    if "test" in lower:
        return "test"
    if "train" in lower or "db" in lower:
        return "train"
    return "unspecified"


def _skip_code(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".zip" or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar.bz2"):
        return "archive_inventory_only"
    return "unsupported_format"


def _parse_status(parse_files: Sequence[Mapping[str, Any]], structures: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("status") == "failed" for row in parse_files):
        return "partial"
    if structures:
        return "pass"
    if parse_files:
        return "partial"
    return "blocked"


def _count_manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_filename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _safe_relative_path(value: str) -> Path:
    parts = [part for part in value.replace("\\", "/").split("/") if part and part not in {".", ".."}]
    safe_parts = [_safe_filename(part) for part in parts]
    return Path(*safe_parts) if safe_parts else Path("file")


def _float_tokens(line: str) -> list[float]:
    return [float(token) for token in line.split()]


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _missing_required_dataset_field(row: Mapping[str, Any], field: str) -> bool:
    if field not in row:
        return True
    value = row.get(field)
    if value is None or value == "":
        return True
    if field in {"target_systems", "formats"} and value == []:
        return True
    return False


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _print_human(report: Mapping[str, Any]) -> None:
    print(f"status: {report.get('status', 'unknown')}")
    if "dataset_count" in report:
        print(f"datasets: {report['dataset_count']}")
    if "structure_count" in report:
        print(f"structures: {report['structure_count']}")
    for issue in report.get("issues", []):
        print(f"- {issue.get('code')}: {issue.get('message')}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
