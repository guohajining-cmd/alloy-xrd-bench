"""MLIP candidate admission, verification, and smoke-audit helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import shlex
import shutil
import site
import ssl
import struct
import subprocess
import sysconfig
import urllib.parse
import urllib.request
import warnings
import zipfile
import zlib
import binascii
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .contract import canonical_structure_hash, potential_workspace_paths


CANDIDATE_REQUIRED_FIELDS = (
    "candidate_id",
    "target_role",
    "family",
    "engine",
    "model_name",
    "source_url",
    "doi",
    "license",
    "redistribution_policy",
    "elements",
    "phases",
    "target_systems",
    "install_method",
    "artifact_kind",
    "acceptance_status",
    "rejection_reason",
)


DEFAULT_SMOKE_INITIAL_IDS = (
    "init_CrCoNi_fcc_Co1Cr1Ni1_seed000007",
    "init_MoNbTaVW_bcc_Mo1Nb1Ta1V1W1_seed000017",
    "init_Ni-Al_l12_Al1Ni3_seed000023",
)


@dataclass
class LoadedInitial:
    """A shared initial structure row plus its JSON artifact."""

    initial_id: str
    row: dict[str, Any]
    document: dict[str, Any]
    structure: Any | None


def audit_candidate_registry(root: str | Path) -> dict[str, Any]:
    """Validate the MLIP candidate registry without creating potential workspaces."""

    root_path = Path(root).resolve()
    registry_path = root_path / "data" / "registry" / "mlip_candidates.yml"
    candidates, issues = _load_candidate_rows(registry_path)
    status_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for row in candidates:
        status = str(row.get("acceptance_status", ""))
        family = str(row.get("family", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "registry_path": str(registry_path),
        "candidate_count": len(candidates),
        "issue_count": len(issues),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


def download_candidate(root: str | Path, candidate_id: str) -> dict[str, Any]:
    """Download or report how to stage a candidate without accepting it.

    Candidate artifacts are staged only under ``local_potential_cache``. They do
    not create ``data/potentials/<id>`` workspaces until separately accepted
    into ``potentials_catalog.yml``.
    """

    root_path = Path(root).resolve()
    registry_path = root_path / "data" / "registry" / "mlip_candidates.yml"
    candidates, issues = _load_candidate_rows(registry_path)
    row = _find_row(candidates, "candidate_id", candidate_id)
    if row is None:
        return {
            "candidate_id": candidate_id,
            "status": "fail",
            "issues": issues
            + [
                {
                    "code": "unknown_candidate",
                    "message": f"Candidate {candidate_id} is not declared in mlip_candidates.yml",
                    "record_id": candidate_id,
                }
            ],
            "created_potential_directory": False,
        }

    artifact_kind = str(row.get("artifact_kind", ""))
    if artifact_kind == "package_pretrained":
        report = {
            "candidate_id": candidate_id,
            "status": "manual_or_package_managed",
            "engine": row.get("engine", ""),
            "install_method": row.get("install_method", ""),
            "artifact_kind": artifact_kind,
            "source_url": row.get("source_url", ""),
            "redistribution_policy": row.get("redistribution_policy", ""),
            "created_potential_directory": False,
            "notes": (
                "Package-managed candidates are installed through their Python "
                "package or framework cache; no data/potentials directory is created."
            ),
        }
        _write_candidate_download_audit(root_path, candidate_id, report)
        return report

    urls = _candidate_artifact_urls(row)
    fairdata_files = _candidate_fairdata_files(row)
    zip_members = _candidate_zip_members(row)
    dataverse_files = _candidate_dataverse_files(row)
    cache_dir = root_path / "local_potential_cache" / candidate_id
    downloads_dir = cache_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    if not urls and not fairdata_files and not zip_members and not dataverse_files:
        report = {
            "candidate_id": candidate_id,
            "status": "blocked",
            "code": "download_url_unresolved",
            "engine": row.get("engine", ""),
            "source_url": row.get("source_url", ""),
            "artifact_kind": artifact_kind,
            "created_potential_directory": False,
            "files": [],
            "notes": (
                "No direct downloadable artifact URL could be resolved. Keep this "
                "candidate out of potentials_catalog.yml until a model file URL, "
                "license, and engine path are confirmed."
            ),
        }
        _write_candidate_download_audit(root_path, candidate_id, report)
        return report

    files: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skipped_files: list[dict[str, Any]] = []
    used_destinations: set[Path] = set()
    for url in urls:
        try:
            destination = downloads_dir / _download_name_from_url(url)
            if destination in used_destinations:
                destination = downloads_dir / _download_name_from_url(url, include_parent=True)
            destination = _avoid_existing_destination_collision(destination, used_destinations)
            used_destinations.add(destination)
            expected_size = _url_content_length(url)
            if (
                destination.exists()
                and destination.stat().st_size > 0
                and expected_size is not None
                and destination.stat().st_size == expected_size
            ):
                digest = _sha256_file(destination)
                files.append(
                    {
                        "source": "existing_cache",
                        "url": url,
                        "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                        "sha256": digest,
                        "bytes": destination.stat().st_size,
                        "expected_bytes": expected_size,
                    }
                )
                continue
            _download_url(url, destination)
            digest = _sha256_file(destination)
            files.append(
                {
                    "url": url,
                    "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "url": url,
                    "code": "download_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    for spec in fairdata_files:
        if spec.get("auto_download") is False:
            skipped_files.append(
                {
                    "source": "fairdata",
                    "dataset_id": str(spec.get("dataset_id", "")),
                    "fairdata_path": str(spec.get("path", "")),
                    "filename": _fairdata_download_name(spec),
                    "expected_sha256": str(spec.get("sha256", "")).lower().replace("sha256:", ""),
                    "expected_bytes": spec.get("size", ""),
                    "reason": str(spec.get("skip_reason", "auto_download is false")),
                }
            )
            continue
        try:
            destination = downloads_dir / _fairdata_download_name(spec)
            destination = _avoid_existing_destination_collision(destination, used_destinations)
            used_destinations.add(destination)
            expected_sha256 = str(spec.get("sha256", "")).lower().replace("sha256:", "")
            if destination.exists() and expected_sha256:
                digest = _sha256_file(destination)
                if digest == expected_sha256:
                    files.append(
                        {
                            "source": "fairdata_existing_cache",
                            "dataset_id": str(spec.get("dataset_id", "")),
                            "fairdata_path": str(spec.get("path", "")),
                            "authorized_download_url": "",
                            "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                            "sha256": digest,
                            "expected_sha256": expected_sha256,
                            "hash_matches_expected": True,
                            "bytes": destination.stat().st_size,
                            "expected_bytes": spec.get("size", ""),
                        }
                    )
                    continue
            resolved_url = _download_fairdata_file(spec, destination)
            digest = _sha256_file(destination)
            files.append(
                {
                    "source": "fairdata",
                    "dataset_id": str(spec.get("dataset_id", "")),
                    "fairdata_path": str(spec.get("path", "")),
                    "authorized_download_url": _redact_url_token(resolved_url),
                    "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                    "sha256": digest,
                    "expected_sha256": expected_sha256,
                    "hash_matches_expected": bool(expected_sha256 and digest == expected_sha256),
                    "bytes": destination.stat().st_size,
                    "expected_bytes": spec.get("size", ""),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "url": str(spec.get("path", "")),
                    "code": "download_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    for spec in zip_members:
        if spec.get("auto_download") is False:
            skipped_files.append(
                {
                    "source": "zip_member",
                    "archive_url": str(spec.get("archive_url", "")),
                    "member_path": str(spec.get("member_path", "")),
                    "filename": _zip_member_download_name(spec),
                    "expected_sha256": str(spec.get("sha256", "")).lower().replace("sha256:", ""),
                    "expected_bytes": spec.get("size", ""),
                    "reason": str(spec.get("skip_reason", "auto_download is false")),
                }
            )
            continue
        try:
            destination = downloads_dir / _zip_member_download_name(spec)
            destination = _avoid_existing_destination_collision(destination, used_destinations)
            used_destinations.add(destination)
            expected_sha256 = str(spec.get("sha256", "")).lower().replace("sha256:", "")
            expected_size = _optional_int(spec.get("size"))
            if destination.exists() and destination.stat().st_size > 0:
                digest = _sha256_file(destination)
                size_matches = expected_size is not None and destination.stat().st_size == expected_size
                hash_matches = bool(expected_sha256 and digest == expected_sha256)
                if hash_matches or (not expected_sha256 and size_matches):
                    files.append(
                        {
                            "source": "zip_member_existing_cache",
                            "archive_url": str(spec.get("archive_url", "")),
                            "member_path": str(spec.get("member_path", "")),
                            "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                            "sha256": digest,
                            "expected_sha256": expected_sha256,
                            "hash_matches_expected": bool(expected_sha256 and digest == expected_sha256),
                            "bytes": destination.stat().st_size,
                            "expected_bytes": spec.get("size", ""),
                        }
                    )
                    continue
            member_info = _download_zip_member(spec, destination)
            digest = _sha256_file(destination)
            files.append(
                {
                    "source": "zip_member",
                    "archive_url": str(spec.get("archive_url", "")),
                    "member_path": str(spec.get("member_path", "")),
                    "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                    "sha256": digest,
                    "expected_sha256": expected_sha256,
                    "hash_matches_expected": bool(expected_sha256 and digest == expected_sha256),
                    "bytes": destination.stat().st_size,
                    "expected_bytes": spec.get("size", ""),
                    "zip_info": member_info,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "url": str(spec.get("archive_url", "")),
                    "member_path": str(spec.get("member_path", "")),
                    "code": "download_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    for spec in dataverse_files:
        if spec.get("auto_download") is False:
            skipped_files.append(
                {
                    "source": "dataverse",
                    "file_id": str(spec.get("file_id", "")),
                    "filename": _dataverse_download_name(spec),
                    "expected_md5": str(spec.get("md5", "")).lower().replace("md5:", ""),
                    "expected_bytes": spec.get("size", ""),
                    "reason": str(spec.get("skip_reason", "auto_download is false")),
                }
            )
            continue
        try:
            destination = downloads_dir / _dataverse_download_name(spec)
            destination = _avoid_existing_destination_collision(destination, used_destinations)
            used_destinations.add(destination)
            expected_md5 = str(spec.get("md5", "")).lower().replace("md5:", "")
            if destination.exists() and expected_md5:
                md5_digest = _md5_file(destination)
                if md5_digest == expected_md5:
                    files.append(
                        {
                            "source": "dataverse_existing_cache",
                            "file_id": str(spec.get("file_id", "")),
                            "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                            "sha256": _sha256_file(destination),
                            "md5": md5_digest,
                            "expected_md5": expected_md5,
                            "md5_matches_expected": True,
                            "bytes": destination.stat().st_size,
                            "expected_bytes": spec.get("size", ""),
                        }
                    )
                    continue
            resolved_url = _download_dataverse_file(spec, destination)
            md5_digest = _md5_file(destination)
            files.append(
                {
                    "source": "dataverse",
                    "file_id": str(spec.get("file_id", "")),
                    "download_url": resolved_url,
                    "path": str(destination.relative_to(root_path)).replace("\\", "/"),
                    "sha256": _sha256_file(destination),
                    "md5": md5_digest,
                    "expected_md5": expected_md5,
                    "md5_matches_expected": bool(expected_md5 and md5_digest == expected_md5),
                    "bytes": destination.stat().st_size,
                    "expected_bytes": spec.get("size", ""),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "url": str(spec.get("file_id", "")),
                    "code": "download_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    status = "downloaded" if files and not failures and not skipped_files else ("partial" if files else "blocked")
    report = {
        "candidate_id": candidate_id,
        "status": status,
        "engine": row.get("engine", ""),
        "install_method": row.get("install_method", ""),
        "artifact_kind": artifact_kind,
        "source_url": row.get("source_url", ""),
        "redistribution_policy": row.get("redistribution_policy", ""),
        "created_potential_directory": False,
        "cache_dir": str(cache_dir.relative_to(root_path)).replace("\\", "/"),
        "files": files,
        "skipped_files": skipped_files,
        "failures": failures,
    }
    _write_candidate_download_audit(root_path, candidate_id, report)
    return report


def verify_potential(
    *,
    root: str | Path,
    potential_id: str,
    reference_initial_id: str | None = None,
    smoke_initial_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify an accepted potential and write per-potential and summary audits."""

    root_path = Path(root).resolve()
    potential = _load_potential_row(root_path, potential_id)
    paths = potential_workspace_paths(potential_id)
    audit_dir = root_path / paths["audits_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    if potential is None:
        report = {
            "potential_id": potential_id,
            "status": "fail",
            "issues": [
                {
                    "code": "unknown_potential",
                    "message": f"Potential {potential_id} is not declared in potentials_catalog.yml",
                    "record_id": potential_id,
                }
            ],
            "load": {"ok": False},
            "model_files": [],
            "reference_inference": None,
            "smoke_relaxations": [],
        }
        _write_verification_report(root_path, potential_id, report)
        return report

    adapter = adapter_for_engine(str(potential.get("engine", "")))
    report: dict[str, Any] = {
        "potential_id": potential_id,
        "family": potential.get("family", ""),
        "engine": potential.get("engine", ""),
        "status": "pass",
        "package_versions": _package_versions(),
        "model_files": adapter.model_files(root_path, potential),
        "load": adapter.load(root_path, potential),
        "reference_inference": None,
        "smoke_relaxations": [],
        "issues": [],
    }

    load_ok = bool(report["load"].get("ok"))
    if not load_ok:
        report["status"] = "fail"
        report["issues"].append(
            {
                "code": report["load"].get("code", "load_failed"),
                "message": report["load"].get("message", "Failed to load MLIP adapter"),
                "record_id": potential_id,
            }
        )

    if reference_initial_id:
        report["reference_inference"] = _run_reference_inference(
            root_path=root_path,
            potential=potential,
            adapter=adapter,
            load_ok=load_ok,
            initial_id=reference_initial_id,
        )
        if report["reference_inference"]["status"] == "fail":
            report["status"] = "fail"

    for initial_id in smoke_initial_ids:
        smoke = smoke_mlip_initial_structure(
            root=root_path,
            potential_id=potential_id,
            initial_id=initial_id,
            write_audit=False,
            potential=potential,
            adapter=adapter,
            load_ok=load_ok,
        )
        report["smoke_relaxations"].append(smoke)
        if smoke["status"] == "fail":
            report["status"] = "fail"

    _write_verification_report(root_path, potential_id, report)
    return report


def smoke_mlip_initial_structure(
    *,
    root: str | Path,
    potential_id: str,
    initial_id: str,
    write_audit: bool = True,
    potential: Mapping[str, Any] | None = None,
    adapter: "MlipAdapter | None" = None,
    load_ok: bool | None = None,
) -> dict[str, Any]:
    """Run a non-manifest MLIP smoke check for one initial structure."""

    root_path = Path(root).resolve()
    potential_row = dict(potential or (_load_potential_row(root_path, potential_id) or {}))
    if not potential_row:
        result = {
            "potential_id": potential_id,
            "initial_structure_id": initial_id,
            "status": "fail",
            "code": "unknown_potential",
            "message": f"Potential {potential_id} is not declared in potentials_catalog.yml",
        }
        if write_audit:
            _write_smoke_report(root_path, potential_id, initial_id, result)
        return result

    adapter_obj = adapter or adapter_for_engine(str(potential_row.get("engine", "")))
    load_result = {"ok": True} if load_ok is True else adapter_obj.load(root_path, potential_row)
    loaded = _load_initial(root_path, initial_id)
    if loaded is None:
        result = {
            "potential_id": potential_id,
            "initial_structure_id": initial_id,
            "status": "fail",
            "code": "unknown_initial_structure",
            "message": f"Initial structure {initial_id} is not declared",
        }
    else:
        unsupported = _unsupported_elements(potential_row, loaded)
        if unsupported:
            result = {
                "potential_id": potential_id,
                "initial_structure_id": initial_id,
                "status": "skipped_unsupported_elements",
                "unsupported_elements": unsupported,
            }
        elif unsupported_phase := _unsupported_phase(potential_row, loaded):
            result = {
                "potential_id": potential_id,
                "initial_structure_id": initial_id,
                "status": "skipped_unsupported_phase",
                "unsupported_phase": unsupported_phase,
            }
        elif not bool(load_result.get("ok")):
            result = {
                "potential_id": potential_id,
                "initial_structure_id": initial_id,
                "status": "fail",
                "code": load_result.get("code", "load_failed"),
                "message": load_result.get("message", "Failed to load MLIP adapter"),
            }
        else:
            result = adapter_obj.smoke_relax(root_path, potential_row, loaded)

    if write_audit:
        _write_smoke_report(root_path, potential_id, initial_id, result)
    return result


class MlipAdapter:
    """Base adapter for package and executable-backed MLIP checks."""

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        potential_id = str(potential.get("potential_id", ""))
        files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
        expected_hash = str(potential.get("file_sha256", "")).lower()
        files: list[dict[str, Any]] = []
        if files_dir.exists():
            for path in sorted(item for item in files_dir.iterdir() if item.is_file()):
                digest = _sha256_file(path)
                files.append(
                    {
                        "path": str(path),
                        "sha256": digest,
                        "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                    }
                )
        return files

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        return {"ok": False, "code": "unsupported_engine", "message": "No adapter is available"}

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        return {"status": "fail", "code": "single_point_not_implemented"}

    def smoke_relax(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        single = self.single_point(root_path, potential, loaded)
        status = single.get("status", "fail")
        return {
            "potential_id": potential.get("potential_id", ""),
            "initial_structure_id": loaded.initial_id,
            "status": status,
            "runner": "single_point_smoke",
            "single_point": single,
            "structure_sha256": canonical_structure_hash(loaded.document),
        }


class FakeAdapter(MlipAdapter):
    """Deterministic adapter used by tests and example workspaces."""

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        return {"ok": True, "adapter": "fake"}

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        return {
            "status": "pass",
            "energy_eV": 0.0,
            "energy_per_atom_eV": 0.0,
            "max_force_eV_per_A": 0.0,
            "stress": {"xx": 0.0, "yy": 0.0, "zz": 0.0},
        }


class ChgnetAdapter(MlipAdapter):
    """CHGNet package-backed adapter."""

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        try:
            import chgnet  # type: ignore
        except Exception as exc:
            return [{"error": f"{type(exc).__name__}: {exc}"}]

        package_root = Path(chgnet.__file__).resolve().parent
        expected_hash = str(potential.get("file_sha256", "")).lower()
        files: list[dict[str, Any]] = []
        for path in sorted(package_root.glob("pretrained/**/*.pth.tar")):
            digest = _sha256_file(path)
            files.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                }
            )
        return files

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        try:
            from chgnet.model import CHGNet

            model_name = _model_name_for(potential, default="0.3.0")
            with redirect_stdout(io.StringIO()):
                model = CHGNet.load(model_name=model_name, verbose=False)
            setattr(self, "_model", model)
            return {"ok": True, "adapter": "chgnet", "model_name": model_name}
        except Exception as exc:
            return {
                "ok": False,
                "code": "load_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        if loaded.structure is None:
            return {
                "status": "fail",
                "code": "invalid_pymatgen_structure",
                "message": "CHGNet requires a valid pymatgen Structure artifact",
            }
        try:
            model = getattr(self, "_model")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = model.predict_structure(loaded.structure)
            forces = np.asarray(prediction.get("f", []), dtype=float)
            energy = float(np.asarray(prediction.get("e"), dtype=float).reshape(-1)[0])
            return {
                "status": "pass",
                "energy_eV": energy,
                "energy_per_atom_eV": energy / max(1, len(loaded.structure)),
                "max_force_eV_per_A": _max_force(forces),
                "stress": _stress_dict(prediction.get("s")),
            }
        except Exception as exc:
            return {
                "status": "fail",
                "code": "single_point_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }


class MaceAdapter(MlipAdapter):
    """MACE ASE-calculator adapter."""

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected_hash = str(potential.get("file_sha256", "")).lower()
        cache_dir = Path.home() / ".cache" / "mace"
        files: list[dict[str, Any]] = []
        if cache_dir.exists():
            for path in sorted(item for item in cache_dir.iterdir() if item.is_file()):
                digest = _sha256_file(path)
                files.append(
                    {
                        "path": str(path),
                        "sha256": digest,
                        "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                    }
                )
        return files or super().model_files(root_path, potential)

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        try:
            from mace.calculators import mace_mp

            model_name = _model_name_for(potential, default="medium")
            calc = mace_mp(model=model_name, device="cpu", default_dtype="float64")
            setattr(self, "_calculator", calc)
            return {"ok": True, "adapter": "mace", "model_name": model_name}
        except ModuleNotFoundError as exc:
            return {
                "ok": False,
                "code": "missing_framework",
                "message": f"mace-torch is not installed: {exc}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "code": "load_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        return _ase_single_point(getattr(self, "_calculator", None), loaded)


class MatglAdapter(MlipAdapter):
    """MatGL/M3GNet ASE-calculator adapter."""

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected_hash = str(potential.get("file_sha256", "")).lower()
        model_slug = "models--materialyze--M3GNet-PES-MatPES-PBE-2025.2"
        cache_dir = Path.home() / ".cache" / "matgl" / model_slug
        files: list[dict[str, Any]] = []
        if cache_dir.exists():
            for path in sorted(item for item in cache_dir.rglob("*") if item.is_file()):
                if path.name == "main" and path.parent.name == "refs":
                    continue
                digest = _sha256_file(path)
                files.append(
                    {
                        "path": str(path),
                        "sha256": digest,
                        "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                    }
                )
        return files or super().model_files(root_path, potential)

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        try:
            import matgl
            from matgl.ext.ase import M3GNetCalculator

            model_name = _model_name_for(
                potential,
                default="materialyze/M3GNet-PES-MatPES-PBE-2025.2",
            )
            model = matgl.load_model(model_name)
            calc = M3GNetCalculator(model)
            setattr(self, "_calculator", calc)
            return {"ok": True, "adapter": "matgl", "model_name": model_name}
        except ModuleNotFoundError as exc:
            return {
                "ok": False,
                "code": "missing_framework",
                "message": f"matgl is not installed: {exc}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "code": "load_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        return _ase_single_point(getattr(self, "_calculator", None), loaded)


class NepAdapter(MlipAdapter):
    """NEP adapter using mdapy.NEP4ASE when available, with GPUMD readiness fallback."""

    model_globs: tuple[str, ...] = ("*.txt", "*nep*", "*NEP*", "*UNEP*")

    def _model_candidates(self, files_dir: Path) -> list[Path]:
        matches = _matching_files(files_dir, self.model_globs)
        return sorted(path for path in matches if path.suffix.lower() != ".in")

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected_hash = str(potential.get("file_sha256", "")).lower()
        potential_id = str(potential.get("potential_id", ""))
        files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
        files: list[dict[str, Any]] = []
        for path in self._model_candidates(files_dir):
            digest = _sha256_file(path)
            files.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                }
            )
        return files or super().model_files(root_path, potential)

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        potential_id = str(potential.get("potential_id", ""))
        files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
        nep_files = self._model_candidates(files_dir)
        if not nep_files:
            return {
                "ok": False,
                "code": "missing_model",
                "message": (
                    f"No NEP model file matching {list(self.model_globs)} "
                    "found under potential/files; NEP input files such as nep.in are not model files"
                ),
            }
        model_file = nep_files[0]

        try:
            from mdapy.nep4ase import NEP4ASE

            calc = NEP4ASE(str(model_file))
            setattr(self, "_calculator", calc)
            setattr(self, "_runtime", "mdapy")
            setattr(self, "_model_file", str(model_file))
            return {
                "ok": True,
                "adapter": "mdapy_nep",
                "model_file": str(model_file),
                "package": "mdapy.nep4ase",
            }
        except ModuleNotFoundError as exc:
            mdapy_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return {
                "ok": False,
                "code": "load_failed",
                "message": f"{type(exc).__name__}: {exc}",
                "model_file": str(model_file),
                "package": "mdapy.nep4ase",
            }

        gpumd = shutil.which("gpumd")
        if gpumd is None:
            return {
                "ok": False,
                "code": "missing_framework",
                "message": f"Neither mdapy.NEP4ASE nor gpumd is available ({mdapy_error}; gpumd not on PATH)",
                "model_file": str(model_file),
            }

        setattr(self, "_runtime", "gpumd_readiness")
        setattr(self, "_model_file", str(model_file))
        return {
            "ok": True,
            "adapter": "gpumd_nep_readiness",
            "executable": gpumd,
            "model_file": str(model_file),
            "message": "GPUMD executable is present; Python single-point inference is not implemented for this fallback",
        }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        if getattr(self, "_runtime", "") == "mdapy":
            return _ase_single_point(getattr(self, "_calculator", None), loaded)
        return {
            "status": "fail",
            "code": "single_point_not_implemented",
            "message": "NEP single-point inference requires the mdapy.NEP4ASE runtime path",
        }


class MtpMlpAdapter(MlipAdapter):
    """MLIP-3 ``mlp`` executable adapter for MTP verification and smoke audits."""

    model_globs = ("*.mtp", "*mtp*")

    def model_files(self, root_path: Path, potential: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected_hash = str(potential.get("file_sha256", "")).lower()
        files = []
        for path in self._model_candidates(root_path, potential):
            digest = _sha256_file(path)
            files.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "matches_catalog_hash": bool(expected_hash and digest == expected_hash),
                }
            )
        return files or super().model_files(root_path, potential)

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        model_files = self._model_candidates(root_path, potential)
        if not model_files:
            return {
                "ok": False,
                "code": "missing_model",
                "message": "No MTP model file found under potential/files",
            }
        try:
            type_to_element, _ = _mtp_mlp_type_maps(potential)
        except ValueError as exc:
            return {"ok": False, "code": "missing_type_map", "message": str(exc)}
        command = _resolve_mtp_mlp_command(root_path, potential)
        if command is None:
            return {
                "ok": False,
                "code": "missing_engine",
                "message": "No MLIP-3 mlp executable found; set mlp_command or ALLOY_XRD_MLP_COMMAND_JSON",
            }
        model_file = model_files[0]
        expected_hash = str(potential.get("file_sha256", "")).lower()
        digest = _sha256_file(model_file)
        if expected_hash and digest != expected_hash:
            return {
                "ok": False,
                "code": "hash_mismatch",
                "message": f"MTP model hash mismatch: {digest} != {expected_hash}",
                "model_file": str(model_file),
            }
        setattr(self, "_model_file", model_file)
        setattr(self, "_command", command)
        setattr(self, "_type_to_element", type_to_element)
        return {
            "ok": True,
            "adapter": "mtp_mlp",
            "model_file": str(model_file),
            "mlp_command": command.get("display", command.get("argv", [])),
            "type_map": {str(key): value for key, value in type_to_element.items()},
        }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        if loaded.structure is None:
            return {
                "status": "fail",
                "code": "invalid_pymatgen_structure",
                "message": "MTP/MLIP-3 verification requires a valid pymatgen Structure artifact",
            }
        try:
            type_to_element, element_to_type = _mtp_mlp_type_maps(potential)
            work_dir = _mtp_mlp_audit_dir(root_path, potential, loaded.initial_id)
            input_cfg = work_dir / "single_point_input.cfg"
            output_cfg = work_dir / "single_point_output.cfg"
            input_cfg.write_text(
                _mtp_mlp_structure_to_cfg(loaded.structure, element_to_type=element_to_type),
                encoding="utf-8",
            )
            _run_mtp_mlp_command(
                root_path=root_path,
                command=getattr(self, "_command"),
                args=[
                    "calculate_efs",
                    _relative_path(root_path, getattr(self, "_model_file")),
                    _relative_path(root_path, input_cfg),
                    f"--output_filename={_relative_path(root_path, output_cfg)}",
                ],
            )
            parsed = _mtp_mlp_parse_cfg(output_cfg, type_to_element=type_to_element)
            energy = parsed.get("energy")
            return {
                "status": "pass",
                "energy_eV": energy,
                "energy_per_atom_eV": float(energy) / max(1, len(loaded.structure)) if energy is not None else None,
                "max_force_eV_per_A": parsed["max_force"],
                "stress": _stress_dict(parsed.get("stress")),
                "input_cfg": _relative_path(root_path, input_cfg),
                "output_cfg": _relative_path(root_path, output_cfg),
            }
        except Exception as exc:
            return {
                "status": "fail",
                "code": "single_point_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def smoke_relax(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        if loaded.structure is None:
            return {
                "potential_id": potential.get("potential_id", ""),
                "initial_structure_id": loaded.initial_id,
                "status": "fail",
                "code": "invalid_pymatgen_structure",
            }
        try:
            type_to_element, element_to_type = _mtp_mlp_type_maps(potential)
            work_dir = _mtp_mlp_audit_dir(root_path, potential, loaded.initial_id)
            input_cfg = work_dir / "relax_input.cfg"
            output_cfg = work_dir / "relax_output.cfg"
            settings = work_dir / "relax.ini"
            raw_log = work_dir / "relax.log"
            input_cfg.write_text(
                _mtp_mlp_structure_to_cfg(loaded.structure, element_to_type=element_to_type),
                encoding="utf-8",
            )
            settings.write_text(
                _mtp_mlp_relax_settings(raw_log_relative=_relative_path(root_path, raw_log)),
                encoding="utf-8",
            )
            _run_mtp_mlp_command(
                root_path=root_path,
                command=getattr(self, "_command"),
                args=[
                    "relax",
                    _relative_path(root_path, getattr(self, "_model_file")),
                    _relative_path(root_path, input_cfg),
                    _relative_path(root_path, output_cfg),
                    f"--relaxation_settings={_relative_path(root_path, settings)}",
                ],
            )
            parsed = _mtp_mlp_parse_cfg(output_cfg, type_to_element=type_to_element)
            return {
                "potential_id": potential.get("potential_id", ""),
                "initial_structure_id": loaded.initial_id,
                "status": "pass",
                "runner": "mtp_mlp",
                "energy_eV": parsed.get("energy"),
                "max_force_eV_per_A": parsed["max_force"],
                "stress": _stress_dict(parsed.get("stress")),
                "input_cfg": _relative_path(root_path, input_cfg),
                "output_cfg": _relative_path(root_path, output_cfg),
                "raw_log": _relative_path(root_path, raw_log),
                "structure_sha256": canonical_structure_hash(loaded.document),
            }
        except Exception as exc:
            return {
                "potential_id": potential.get("potential_id", ""),
                "initial_structure_id": loaded.initial_id,
                "status": "fail",
                "code": "smoke_relax_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

    def _model_candidates(self, root_path: Path, potential: Mapping[str, Any]) -> list[Path]:
        potential_id = str(potential.get("potential_id", ""))
        files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
        return _matching_files(files_dir, self.model_globs)


class LammpsFileAdapter(MlipAdapter):
    """Readiness checks for LAMMPS-backed external potential files."""

    executable_names: tuple[str, ...] = ("lmp", "lmp_serial", "lmp_mpi", "lammps")
    model_globs: tuple[str, ...] = ("*.*",)
    adapter_name = "lammps_file"

    def load(self, root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any]:
        executable = _first_executable(self.executable_names)
        if executable is None:
            return {
                "ok": False,
                "code": "missing_engine",
                "message": (
                    "LAMMPS executable was not found on PATH. Use WSL2/Linux "
                    "or a LAMMPS build with the required MLIP package."
                ),
                "required_executables": list(self.executable_names),
            }

        potential_id = str(potential.get("potential_id", ""))
        files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
        model_files = _matching_files(files_dir, self.model_globs)
        if not model_files:
            return {
                "ok": False,
                "code": "missing_model",
                "message": f"No model file matching {list(self.model_globs)} found under potential/files",
                "executable": executable,
            }
        return {
            "ok": True,
            "adapter": self.adapter_name,
            "executable": executable,
            "model_files": [str(path) for path in model_files],
        }

    def single_point(self, root_path: Path, potential: Mapping[str, Any], loaded: LoadedInitial) -> dict[str, Any]:
        return {
            "status": "fail",
            "code": "single_point_not_implemented",
            "message": (
                f"{self.adapter_name} currently performs admission readiness checks only; "
                "LAMMPS input generation is required before formal relaxation."
            ),
        }


class MtpLammpsAdapter(LammpsFileAdapter):
    """MTP through LAMMPS/user-mlip readiness adapter."""

    model_globs = ("*.mtp", "*.pot", "*mtp*")
    adapter_name = "mtp_lammps"


class TabgapLammpsAdapter(LammpsFileAdapter):
    """tabGAP through LAMMPS readiness adapter."""

    model_globs = ("*.h5", "*.hdf5", "*.xml", "*.gap", "*.tabgap", "*tabgap*", "*gap*")
    adapter_name = "tabgap_lammps"


class SnapLammpsAdapter(LammpsFileAdapter):
    """SNAP through LAMMPS readiness adapter."""

    model_globs = ("*.snapcoeff", "*.snapparam", "*snap*")
    adapter_name = "snap_lammps"


class PaceLammpsAdapter(LammpsFileAdapter):
    """ACE/PACE through LAMMPS readiness adapter."""

    model_globs = ("*.yace", "*.ace", "*pace*")
    adapter_name = "pace_lammps"


class DeepmdLammpsAdapter(LammpsFileAdapter):
    """DeepMD/DPA through LAMMPS readiness adapter."""

    model_globs = (
        "*.pb",
        "*.pth",
        "*.pt",
        "*.ckpt",
        "*.json",
        "*deepmd*",
        "*dpa*",
        "*dp-*",
    )
    adapter_name = "deepmd_lammps"


def adapter_for_engine(engine: str) -> MlipAdapter:
    normalized = engine.lower().replace("_", "-")
    if normalized == "fake":
        return FakeAdapter()
    if normalized == "chgnet":
        return ChgnetAdapter()
    if normalized in {"mace", "mace-torch"}:
        return MaceAdapter()
    if normalized in {"matgl", "m3gnet"}:
        return MatglAdapter()
    if normalized in {"nep", "gpumd"}:
        return NepAdapter()
    if normalized in {"mtp-mlp", "mlip3", "mlip-3"}:
        return MtpMlpAdapter()
    if normalized in {"mtp-lammps", "mlip-lammps"}:
        return MtpLammpsAdapter()
    if normalized in {"tabgap-lammps", "gap-lammps"}:
        return TabgapLammpsAdapter()
    if normalized in {"snap", "snap-lammps"}:
        return SnapLammpsAdapter()
    if normalized in {"pace", "pace-lammps", "ace", "ace-lammps"}:
        return PaceLammpsAdapter()
    if normalized in {"deepmd", "deepmd-lammps", "dpa", "dpa-lammps", "dp-lammps"}:
        return DeepmdLammpsAdapter()
    return MlipAdapter()


def _run_reference_inference(
    *,
    root_path: Path,
    potential: Mapping[str, Any],
    adapter: MlipAdapter,
    load_ok: bool,
    initial_id: str,
) -> dict[str, Any]:
    loaded = _load_initial(root_path, initial_id)
    if loaded is None:
        return {
            "initial_structure_id": initial_id,
            "status": "fail",
            "code": "unknown_initial_structure",
        }
    unsupported = _unsupported_elements(potential, loaded)
    if unsupported:
        return {
            "initial_structure_id": initial_id,
            "status": "skipped_unsupported_elements",
            "unsupported_elements": unsupported,
        }
    unsupported_phase = _unsupported_phase(potential, loaded)
    if unsupported_phase:
        return {
            "initial_structure_id": initial_id,
            "status": "skipped_unsupported_phase",
            "unsupported_phase": unsupported_phase,
        }
    if not load_ok:
        return {
            "initial_structure_id": initial_id,
            "status": "fail",
            "code": "load_failed",
        }
    result = adapter.single_point(root_path, potential, loaded)
    result["initial_structure_id"] = initial_id
    return result


def _load_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not path.exists():
        return [], [
            {
                "code": "missing_candidate_registry",
                "message": f"Missing MLIP candidate registry: {path}",
                "record_id": "",
            }
        ]
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = [dict(row) for row in payload.get("candidates", [])]
    except Exception as exc:
        return [], [
            {
                "code": "candidate_registry_load_error",
                "message": f"{type(exc).__name__}: {exc}",
                "record_id": "",
            }
        ]

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id", ""))
        for field in CANDIDATE_REQUIRED_FIELDS:
            if _is_empty_candidate_value(row.get(field)) and field != "rejection_reason":
                issues.append(
                    {
                        "code": "missing_required_field",
                        "message": f"Candidate {candidate_id or index} is missing {field}",
                        "record_id": candidate_id,
                    }
                )
        if candidate_id:
            if candidate_id in seen:
                issues.append(
                    {
                        "code": "duplicate_candidate_id",
                        "message": f"Duplicate candidate_id {candidate_id}",
                        "record_id": candidate_id,
                    }
                )
            else:
                seen[candidate_id] = index
    return rows, issues


def _is_empty_candidate_value(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0
    return False


def _load_potential_row(root_path: Path, potential_id: str) -> dict[str, Any] | None:
    catalog_path = root_path / "data" / "registry" / "potentials_catalog.yml"
    if not catalog_path.exists():
        return None
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    rows = payload.get("potentials", [])
    return _find_row(rows, "potential_id", potential_id)


def _find_row(rows: Sequence[Mapping[str, Any]], field: str, value: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get(field, "")) == value:
            return dict(row)
    return None


def _load_initial(root_path: Path, initial_id: str) -> LoadedInitial | None:
    manifest_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not manifest_path.exists():
        return None
    rows = pd.read_parquet(manifest_path)
    matches = rows[rows["initial_structure_id"] == initial_id]
    if matches.empty:
        return None
    row = matches.iloc[0].to_dict()
    artifact_path = root_path / str(row["artifact_path"])
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    structure = None
    try:
        from pymatgen.core import Structure

        structure = Structure.from_dict(document)
    except Exception:
        structure = None
    return LoadedInitial(initial_id=initial_id, row=row, document=document, structure=structure)


def _unsupported_elements(potential: Mapping[str, Any], loaded: LoadedInitial) -> list[str]:
    supported = {str(element) for element in potential.get("elements", [])}
    if not supported:
        return []
    elements = _initial_elements(loaded)
    return sorted(elements - supported)


def _unsupported_phase(potential: Mapping[str, Any], loaded: LoadedInitial) -> str:
    supported = {str(phase).lower() for phase in potential.get("phases", []) if str(phase)}
    if not supported:
        return ""
    phase = str(loaded.row.get("phase", "")).lower()
    return phase if phase and phase not in supported else ""


def _initial_elements(loaded: LoadedInitial) -> set[str]:
    composition = loaded.row.get("composition", {})
    if isinstance(composition, Mapping):
        return {
            str(element)
            for element, amount in composition.items()
            if amount not in {None, "", 0, 0.0}
        }
    if loaded.structure is not None:
        return {str(element) for element in loaded.structure.composition.elements}
    return set()


def _write_verification_report(root_path: Path, potential_id: str, report: Mapping[str, Any]) -> None:
    paths = potential_workspace_paths(potential_id)
    audit_dir = root_path / paths["audits_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "potential_verification.json"
    audit_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _upsert_verification_summary(root_path, potential_id, audit_path, report)


def _write_smoke_report(root_path: Path, potential_id: str, initial_id: str, report: Mapping[str, Any]) -> None:
    paths = potential_workspace_paths(potential_id)
    smoke_dir = root_path / paths["audits_dir"] / "smoke_relaxations"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / f"{initial_id}.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _upsert_verification_summary(
    root_path: Path,
    potential_id: str,
    audit_path: Path,
    report: Mapping[str, Any],
) -> None:
    summary_path = root_path / "audits" / "potential_verification_summary.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_rows = list(report.get("smoke_relaxations", []))
    row = {
        "potential_id": potential_id,
        "engine": report.get("engine", ""),
        "status": report.get("status", ""),
        "load_ok": bool(report.get("load", {}).get("ok")),
        "reference_status": (report.get("reference_inference") or {}).get("status", ""),
        "smoke_pass": sum(1 for item in smoke_rows if item.get("status") == "pass"),
        "smoke_skipped": sum(1 for item in smoke_rows if str(item.get("status", "")).startswith("skipped")),
        "smoke_failed": sum(1 for item in smoke_rows if item.get("status") == "fail"),
        "audit_path": str(audit_path.relative_to(root_path)).replace("\\", "/"),
    }
    if summary_path.exists():
        existing = pd.read_parquet(summary_path)
        existing = existing[existing["potential_id"] != potential_id]
        data = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        data = pd.DataFrame([row])
    data.sort_values("potential_id").to_parquet(summary_path)


def _package_versions() -> dict[str, str | None]:
    packages = ("chgnet", "mace-torch", "matgl", "ase", "pymatgen", "torch", "mdapy", "deepmd-kit", "lammps")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _ase_single_point(calculator: Any, loaded: LoadedInitial) -> dict[str, Any]:
    if calculator is None:
        return {"status": "fail", "code": "calculator_not_loaded"}
    if loaded.structure is None:
        return {
            "status": "fail",
            "code": "invalid_pymatgen_structure",
            "message": "ASE adapters require a valid pymatgen Structure artifact",
        }
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(loaded.structure)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        stress = atoms.get_stress()
        return {
            "status": "pass",
            "energy_eV": energy,
            "energy_per_atom_eV": energy / max(1, len(atoms)),
            "max_force_eV_per_A": _max_force(forces),
            "stress": _stress_dict(stress),
        }
    except Exception as exc:
        return {
            "status": "fail",
            "code": "single_point_failed",
            "message": f"{type(exc).__name__}: {exc}",
        }


def _model_name_for(potential: Mapping[str, Any], *, default: str) -> str:
    metadata_path = Path(str(potential.get("potential_dir", ""))) / "metadata.yml"
    if metadata_path.exists():
        payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if payload.get("model_name"):
            return str(payload["model_name"])
    potential_id = str(potential.get("potential_id", ""))
    if potential_id == "pot_matgl_m3gnet_matpes_pbe_2025_2":
        return "materialyze/M3GNet-PES-MatPES-PBE-2025.2"
    if potential_id == "pot_mace_mp_0b3_medium":
        return "medium"
    return default


def _max_force(forces: np.ndarray) -> float:
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces, axis=1).max())


def _stress_dict(stress: Any) -> dict[str, float]:
    if stress is None:
        return {}
    matrix_or_vector = np.asarray(stress, dtype=float)
    if matrix_or_vector.shape == (3, 3):
        return {
            "xx": float(matrix_or_vector[0, 0]),
            "yy": float(matrix_or_vector[1, 1]),
            "zz": float(matrix_or_vector[2, 2]),
            "yz": float(matrix_or_vector[1, 2]),
            "xz": float(matrix_or_vector[0, 2]),
            "xy": float(matrix_or_vector[0, 1]),
        }
    values = matrix_or_vector.reshape(-1)
    labels = ("xx", "yy", "zz", "yz", "xz", "xy")
    return {label: float(values[index]) for index, label in enumerate(labels[: len(values)])}


def _mtp_mlp_type_maps(potential: Mapping[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    raw = potential.get("type_map")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"Potential {potential.get('potential_id', '')} must declare type_map")
    type_to_element = {int(key): str(value) for key, value in raw.items()}
    element_to_type = {element: type_index for type_index, element in type_to_element.items()}
    missing = sorted(str(element) for element in potential.get("elements", []) if str(element) not in element_to_type)
    if missing:
        raise ValueError(f"type_map is missing declared elements: {missing}")
    return dict(sorted(type_to_element.items())), element_to_type


def _mtp_mlp_audit_dir(root_path: Path, potential: Mapping[str, Any], initial_id: str) -> Path:
    potential_id = str(potential.get("potential_id", ""))
    paths = potential_workspace_paths(potential_id)
    safe_initial = "".join(char if char.isalnum() or char in "-_." else "_" for char in initial_id)
    directory = root_path / paths["audits_dir"] / "mtp_mlp" / safe_initial
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _mtp_mlp_structure_to_cfg(structure: Any, *, element_to_type: Mapping[str, int]) -> str:
    lines = [
        "BEGIN_CFG",
        " Size",
        f"{len(structure):6d}",
        " Supercell",
    ]
    for vector in structure.lattice.matrix:
        lines.append("  " + " ".join(f"{float(value):16.10f}" for value in vector))
    lines.append(" AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz")
    for index, site in enumerate(structure, start=1):
        element = str(site.specie.symbol)
        type_index = element_to_type[element]
        x, y, z = (float(value) for value in site.coords)
        lines.append(
            f"{index:14d} {type_index:4d} {x:14.8f} {y:13.8f} {z:13.8f}"
            "     0.000000    0.000000    0.000000"
        )
    lines.extend(
        [
            " Energy",
            "       0.000000000000",
            " PlusStress:  xx          yy          zz          yz          xz          xy",
            "     0.00000    0.00000    0.00000    0.00000    0.00000    0.00000",
            " Feature   EFS_by\talloy-xrd-bench-mtp-mlp-audit",
            "END_CFG",
            "",
        ]
    )
    return "\n".join(lines)


def _mtp_mlp_parse_cfg(path: Path, *, type_to_element: Mapping[int, str]) -> dict[str, Any]:
    from pymatgen.core import Structure

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cell: list[list[float]] = []
    species: list[str] = []
    coords: list[list[float]] = []
    forces: list[list[float]] = []
    energy: float | None = None
    stress: list[float] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "Supercell":
            cell = [[float(value) for value in lines[index + offset].split()[:3]] for offset in (1, 2, 3)]
            index += 4
            continue
        if stripped.startswith("AtomData:"):
            index += 1
            while index < len(lines):
                atom_line = lines[index].strip()
                if not atom_line or atom_line.startswith(("Energy", "PlusStress", "Feature", "END_CFG")):
                    break
                parts = atom_line.split()
                if len(parts) >= 8:
                    type_index = int(parts[1])
                    species.append(type_to_element[type_index])
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
                index += 1
            continue
        if stripped == "Energy" and index + 1 < len(lines):
            energy = float(lines[index + 1].strip())
            index += 2
            continue
        if stripped.startswith("PlusStress") and index + 1 < len(lines):
            stress = [float(value) for value in lines[index + 1].split()[:6]]
            index += 2
            continue
        index += 1
    if not cell or not species:
        raise ValueError(f"Could not parse MTP cfg: {path}")
    return {
        "structure": Structure(cell, species, coords, coords_are_cartesian=True),
        "energy": energy,
        "stress": stress,
        "max_force": _max_force(np.asarray(forces, dtype=float)),
    }


def _mtp_mlp_relax_settings(*, raw_log_relative: str) -> str:
    return "\n".join(
        [
            "### RELAXATION OPTIONS ####",
            "relax:pressure                        0.0",
            "relax:iteration_limit                 200",
            "relax:mindist                         1.0",
            "relax:force_tolerance                 0.05",
            "relax:stress_tolerance                0.0",
            "relax:max_step                        0.2",
            "relax:min_step                        1.0e-8",
            "relax:bfgs_wolfe_c1                   1.0e-3",
            "relax:bfgs_wolfe_c2                   0.7",
            f"relax:log                             {raw_log_relative}",
            "relax:freeze_small_grads              FALSE",
            "relax:use_gd_method                   FALSE",
            "relax:correct_cell                    FALSE",
            "relax:init_mindist                    1.0",
            "relax:forces_elasticity_scale         10",
            "relax:stress_elasticity_scale         50",
            "",
        ]
    )


def _resolve_mtp_mlp_command(root_path: Path, potential: Mapping[str, Any]) -> dict[str, Any] | None:
    declared = potential.get("mlp_command")
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
        return {"kind": "direct", "argv": [str(part) for part in declared], "display": list(declared)}
    if isinstance(declared, str) and declared:
        parsed = shlex.split(declared, posix=os.name != "nt")
        return {"kind": "direct", "argv": parsed, "display": parsed}
    env_json = os.environ.get("ALLOY_XRD_MLP_COMMAND_JSON")
    if env_json:
        parsed = json.loads(env_json)
        return {"kind": "direct", "argv": [str(part) for part in parsed], "display": parsed}
    env_text = os.environ.get("ALLOY_XRD_MLP_COMMAND")
    if env_text:
        parsed = shlex.split(env_text, posix=os.name != "nt")
        return {"kind": "direct", "argv": parsed, "display": parsed}
    executable = _first_executable(("mlp", "mlp.exe"))
    if executable:
        return {"kind": "direct", "argv": [executable], "display": [executable]}
    wsl_build = root_path / "local_potential_cache" / "source_documents" / "mlip-3-wsl-build-round48" / "bin" / "mlp"
    if wsl_build.exists() and shutil.which("wsl"):
        return {
            "kind": "wsl",
            "argv": [str(wsl_build)],
            "display": ["wsl", "bash", "-lc", str(wsl_build)],
            "wsl_mlp": _windows_path_to_wsl(wsl_build),
        }
    return None


def _run_mtp_mlp_command(*, root_path: Path, command: Mapping[str, Any], args: Sequence[str]) -> None:
    if command.get("kind") == "wsl":
        wsl_cwd = _windows_path_to_wsl(root_path)
        shell_command = " ".join(shlex.quote(part) for part in [str(command["wsl_mlp"]), *args])
        completed = subprocess.run(
            ["wsl", "bash", "-lc", f"cd {shlex.quote(wsl_cwd)} && {shell_command}"],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    else:
        completed = subprocess.run(
            [*command["argv"], *args],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "mlp command failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(parts)


def _relative_path(root_path: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root_path)).replace("\\", "/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_artifact_urls(row: Mapping[str, Any]) -> list[str]:
    declared = row.get("artifact_urls", [])
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, Sequence) and not isinstance(declared, (bytes, bytearray)):
        urls = [str(item) for item in declared if str(item)]
        if urls:
            return urls
    if _candidate_zip_members(row):
        return []

    source_url = str(row.get("source_url", ""))
    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme == "file":
        return [source_url]
    if parsed.scheme == "" and source_url:
        source_path = Path(source_url)
        if source_path.exists():
            return [source_path.resolve().as_uri()]
    if parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            return [f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"]
    if parsed.netloc.lower() == "zenodo.org" and "/records/" in parsed.path:
        record_id = parsed.path.rstrip("/").split("/")[-1]
        return _zenodo_file_urls(record_id)
    return []


def _candidate_fairdata_files(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared = row.get("fairdata_files", [])
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes, bytearray)):
        return []
    specs: list[dict[str, Any]] = []
    for item in declared:
        if isinstance(item, Mapping) and item.get("dataset_id") and item.get("path"):
            specs.append(dict(item))
    return specs


def _candidate_zip_members(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared = row.get("zip_members", [])
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes, bytearray)):
        return []
    specs: list[dict[str, Any]] = []
    for item in declared:
        if isinstance(item, Mapping) and item.get("archive_url") and item.get("member_path"):
            specs.append(dict(item))
    return specs


def _candidate_dataverse_files(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared = row.get("dataverse_files", [])
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes, bytearray)):
        return []
    specs: list[dict[str, Any]] = []
    for item in declared:
        if isinstance(item, Mapping) and item.get("file_id"):
            specs.append(dict(item))
    return specs


def _zenodo_file_urls(record_id: str) -> list[str]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    urls: list[str] = []
    for file_row in payload.get("files", []):
        links = file_row.get("links", {})
        download = links.get("self") or links.get("download")
        if download:
            urls.append(str(download))
    return urls


def _download_name_from_url(url: str, *, include_parent: bool = False) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        name = Path(urllib.request.url2pathname(parsed.path)).name
    else:
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[-1] == "content":
            name = path_parts[-2]
        elif include_parent and len(path_parts) >= 2:
            name = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            name = Path(parsed.path).name
    if not name:
        name = "artifact.bin"
    return name


def _avoid_existing_destination_collision(destination: Path, used_destinations: set[Path]) -> Path:
    if destination not in used_destinations:
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if candidate not in used_destinations:
            return candidate
        counter += 1


def _fairdata_download_name(spec: Mapping[str, Any]) -> str:
    name = str(spec.get("filename") or "")
    if not name:
        name = Path(str(spec.get("path", ""))).name
    return name or "fairdata_artifact.bin"


def _zip_member_download_name(spec: Mapping[str, Any]) -> str:
    name = str(spec.get("filename") or "")
    if not name:
        name = Path(str(spec.get("member_path", ""))).name
    return name or "zip_member_artifact.bin"


def _dataverse_download_name(spec: Mapping[str, Any]) -> str:
    name = str(spec.get("filename") or "")
    return name or f"dataverse_file_{spec.get('file_id', 'artifact')}.bin"


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        source_path = Path(urllib.request.url2pathname(parsed.path))
        shutil.copy2(source_path, destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "alloy-xrd-bench/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        expected_size = _response_content_length(response)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    _assert_download_size(destination, expected_size)


def _url_content_length(url: str) -> int | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        source_path = Path(urllib.request.url2pathname(parsed.path))
        return source_path.stat().st_size if source_path.exists() else None
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "alloy-xrd-bench/0.1"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return _response_content_length(response)
    except Exception:
        return None


def _download_fairdata_file(spec: Mapping[str, Any], destination: Path) -> str:
    dataset_id = str(spec.get("dataset_id", ""))
    file_path = str(spec.get("path", ""))
    base_url = str(spec.get("download_base_url") or "https://etsin.fairdata.fi").rstrip("/")
    if not dataset_id or not file_path:
        raise ValueError("Fairdata download spec requires dataset_id and path")

    payload = json.dumps({"cr_id": dataset_id, "file": file_path}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/v3/download/authorize",
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "alloy-xrd-bench/0.1",
        },
        method="POST",
    )
    ssl_context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
        authorization = json.loads(response.read().decode("utf-8"))
    download_url = str(authorization.get("url", ""))
    if not download_url:
        raise RuntimeError("Fairdata authorize response did not include a download URL")

    destination.parent.mkdir(parents=True, exist_ok=True)
    download_request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "alloy-xrd-bench/0.1"},
    )
    with urllib.request.urlopen(download_request, timeout=300, context=ssl_context) as response:
        expected_size = _response_content_length(response)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    _assert_download_size(destination, expected_size)
    return download_url


def _download_dataverse_file(spec: Mapping[str, Any], destination: Path) -> str:
    file_id = str(spec.get("file_id", ""))
    if not file_id:
        raise ValueError("Dataverse download spec requires file_id")
    base_url = str(spec.get("base_url") or "https://darus.uni-stuttgart.de").rstrip("/")
    download_url = f"{base_url}/api/access/datafile/{file_id}"
    request = urllib.request.Request(download_url, headers={"User-Agent": "alloy-xrd-bench/0.1"})
    ssl_context = ssl._create_unverified_context()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300, context=ssl_context) as response:
        expected_size = _response_content_length(response)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    declared_size = _optional_int(spec.get("size"))
    _assert_download_size(destination, expected_size if expected_size is not None else declared_size)
    return download_url


def _download_zip_member(spec: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    archive_url = str(spec.get("archive_url", ""))
    member_path = str(spec.get("member_path", "")).lstrip("/")
    if not archive_url or not member_path:
        raise ValueError("ZIP member download spec requires archive_url and member_path")

    parsed = urllib.parse.urlparse(archive_url)
    if parsed.scheme == "file":
        archive_path = Path(urllib.request.url2pathname(parsed.path))
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(member_path)
            data = archive.read(info)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {
            "archive_kind": "local_zip",
            "member_path": member_path,
            "compression_method": info.compress_type,
            "compressed_bytes": info.compress_size,
            "uncompressed_bytes": info.file_size,
            "crc32": f"{info.CRC:08x}",
        }

    entry = _remote_zip_member_entry(archive_url, member_path)
    compressed = _http_get_range(
        archive_url,
        int(entry["data_offset"]),
        int(entry["data_offset"]) + int(entry["compressed_size"]) - 1,
    )
    method = int(entry["compression_method"])
    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise NotImplementedError(f"Unsupported ZIP compression method {method}")

    expected_size = int(entry["uncompressed_size"])
    if len(data) != expected_size:
        raise IOError(f"Extracted {len(data)} bytes, expected {expected_size}")
    expected_crc = int(entry["crc32"])
    actual_crc = binascii.crc32(data) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise IOError(f"Extracted CRC32 {actual_crc:08x}, expected {expected_crc:08x}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "archive_kind": "remote_zip_http_range",
        "member_path": member_path,
        "compression_method": method,
        "compressed_bytes": int(entry["compressed_size"]),
        "uncompressed_bytes": expected_size,
        "crc32": f"{expected_crc:08x}",
    }


def _remote_zip_member_entry(archive_url: str, member_path: str) -> dict[str, Any]:
    archive_size = _url_content_length(archive_url)
    if archive_size is None:
        raise IOError("Could not determine remote ZIP Content-Length")

    tail_size = min(archive_size, 1024 * 1024)
    tail_start = archive_size - tail_size
    tail = _http_get_range(archive_url, tail_start, archive_size - 1)
    eocd_index = tail.rfind(b"PK\x05\x06")
    if eocd_index < 0:
        raise zipfile.BadZipFile("End-of-central-directory record was not found")
    eocd = tail[eocd_index : eocd_index + 22]
    (
        signature,
        _disk_number,
        _central_directory_disk,
        _disk_entries,
        _total_entries,
        central_directory_size,
        central_directory_offset,
        _comment_length,
    ) = struct.unpack("<IHHHHIIH", eocd)
    if signature != 0x06054B50:
        raise zipfile.BadZipFile("Invalid end-of-central-directory signature")
    if central_directory_size == 0xFFFFFFFF or central_directory_offset == 0xFFFFFFFF:
        raise NotImplementedError("ZIP64 central directories are not supported by the range downloader")

    central_directory = _http_get_range(
        archive_url,
        central_directory_offset,
        central_directory_offset + central_directory_size - 1,
    )
    offset = 0
    wanted = member_path.replace("\\", "/").lstrip("/")
    while offset + 46 <= len(central_directory):
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", central_directory, offset)
        signature = fields[0]
        if signature != 0x02014B50:
            raise zipfile.BadZipFile("Invalid central-directory file header signature")
        flags = fields[3]
        compression_method = fields[4]
        crc32_value = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        local_header_offset = fields[16]
        filename_bytes = central_directory[offset + 46 : offset + 46 + filename_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        filename = filename_bytes.decode(encoding).replace("\\", "/")
        if filename == wanted:
            data_offset = _remote_zip_data_offset(archive_url, local_header_offset)
            return {
                "member_path": filename,
                "compression_method": compression_method,
                "crc32": crc32_value,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "data_offset": data_offset,
            }
        offset += 46 + filename_length + extra_length + comment_length

    raise FileNotFoundError(f"ZIP member {member_path} was not found")


def _remote_zip_data_offset(archive_url: str, local_header_offset: int) -> int:
    local_header = _http_get_range(archive_url, local_header_offset, local_header_offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", local_header)
    if fields[0] != 0x04034B50:
        raise zipfile.BadZipFile("Invalid local file header signature")
    filename_length = fields[9]
    extra_length = fields[10]
    return local_header_offset + 30 + filename_length + extra_length


def _http_get_range(url: str, start: int, end: int) -> bytes:
    if end < start:
        return b""
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "alloy-xrd-bench/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise IOError(f"Remote server did not honor Range request; status={status}")
        data = response.read()
    expected_size = end - start + 1
    if len(data) != expected_size:
        raise IOError(f"Received {len(data)} bytes for range {start}-{end}, expected {expected_size}")
    return data


def _response_content_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _assert_download_size(destination: Path, expected_size: int | None) -> None:
    if expected_size is None:
        return
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Downloaded {actual_size} bytes, expected {expected_size}")


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _redact_url_token(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.query:
        return url
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted = [(key, "<redacted>" if key.lower() == "token" else value) for key, value in query]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(redacted)))


def _write_candidate_download_audit(root_path: Path, candidate_id: str, report: Mapping[str, Any]) -> None:
    cache_dir = root_path / "local_potential_cache" / candidate_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "download_audit.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = list(report.get("files", []))
    if files:
        (cache_dir / "checksums.sha256").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in files),
            encoding="utf-8",
        )


def _python_script_dirs() -> list[Path]:
    dirs: list[Path] = []
    for value in (
        sysconfig.get_path("scripts"),
        str(Path(site.USER_BASE) / ("Scripts" if os.name == "nt" else "bin")),
    ):
        if value:
            dirs.append(Path(value))
    try:
        user_site = Path(site.getusersitepackages())
        dirs.append(user_site.parent / ("Scripts" if os.name == "nt" else "bin"))
    except Exception:
        pass
    try:
        import lammps  # type: ignore[import-not-found]

        lammps_dir = Path(lammps.__file__).resolve().parent
        dirs.extend([lammps_dir, lammps_dir / "bin"])
    except Exception:
        pass
    return list(dict.fromkeys(dirs))


def _executable_path_names(name: str) -> list[str]:
    path = Path(name)
    names = [path.name]
    if os.name == "nt" and not path.suffix:
        pathext = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD")
        names.extend(path.name + ext.lower() for ext in pathext.split(";") if ext)
    return list(dict.fromkeys(names))


def _first_executable(names: Sequence[str]) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    for directory in _python_script_dirs():
        for name in names:
            for filename in _executable_path_names(name):
                path = directory / filename
                if path.is_file():
                    return str(path)
    return None


def _matching_files(files_dir: Path, globs: Sequence[str]) -> list[Path]:
    if not files_dir.exists():
        return []
    matches: list[Path] = []
    for pattern in globs:
        matches.extend(path for path in files_dir.glob(pattern) if path.is_file())
    return sorted(set(matches))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
