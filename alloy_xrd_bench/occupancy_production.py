"""Resumable, run-scoped production of occupancy-aware XRD shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .occupancy_pipeline import (
    XRD_COLUMNS,
    _phase_peak_table,
    _split_map,
    _structure_from_record,
    audit_dataset,
)
from .xrd import _broadened_grid, _xrd_pattern


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
REQUIRED_CONFIG_KEYS = {
    "xrd_config_id", "radiation", "two_theta_min", "two_theta_max", "step",
    "broadening", "intensity_normalization",
}
RUN_PURPOSES = {"xrd-artifact", "benchmark-release"}


def plan_production_run(
    *, dataset_dir: str | Path, run_id: str, subset: str, config_path: str | Path,
    shard_size: int = 256, max_structures: int | None = None,
    purpose: str = "xrd-artifact",
) -> dict[str, Any]:
    """Freeze selected structures, source payloads, configuration, and shard assignment."""

    dataset = Path(dataset_dir).resolve()
    _validate_run_id(run_id)
    if purpose not in RUN_PURPOSES:
        return {"status": "fail", "reason": "invalid_run_purpose", "purpose": purpose}
    if shard_size < 1 or (max_structures is not None and max_structures < 1):
        return {"status": "fail", "reason": "shard_size_and_max_structures_must_be_positive"}
    dataset_audit = audit_dataset(dataset_dir=dataset)
    if dataset_audit["status"] != "pass":
        return {
            "status": "fail", "reason": "dataset_engineering_gate_failed",
            "purpose": purpose, "issues": dataset_audit["issues"],
        }
    if purpose == "benchmark-release" and not dataset_audit["benchmark_ready"]:
        return {
            "status": "fail", "reason": "benchmark_release_gate_failed",
            "purpose": purpose, "benchmark_ready": False,
            "manual_review_complete": dataset_audit["manual_review_complete"],
            "manual_review_pass_rate": dataset_audit["manual_review_pass_rate"],
        }
    config = _load_config(Path(config_path))
    points = _config_points(config)
    if points != 4001:
        return {"status": "fail", "reason": "production_config_must_have_4001_points", "points": points}

    structures = pd.read_parquet(dataset / "manifests" / "structures.parquet")
    selected = _select_structures(dataset, structures, subset)
    if max_structures is not None:
        selected = selected.head(max_structures)
    if selected.empty:
        return {"status": "fail", "reason": "empty_subset", "subset": subset}

    summary = _read_json(dataset / "reports" / "inventory_summary.json")
    source = _resolve_source_csv(dataset, summary)
    source_sha256 = _sha256_file(source)
    expected_source_sha256 = str(summary.get("source_sha256", ""))
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        return {"status": "fail", "reason": "source_checksum_mismatch"}

    run_root = _run_root(dataset, run_id)
    plan_path = run_root / "run_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        requested_ids = selected["dataset_structure_id"].astype(str).tolist()
        existing_config = _read_json(run_root / "xrd_config.json")
        compatible = (
            existing.get("subset") == subset
            and existing.get("purpose", "benchmark-release") == purpose
            and int(existing.get("shard_size", 0)) == shard_size
            and existing.get("structure_ids") == requested_ids
            and existing.get("source_sha256") == source_sha256
            and _mapping_sha256(existing_config) == _mapping_sha256(config)
        )
        if not compatible:
            return {"status": "fail", "reason": "run_plan_incompatible", "run_id": run_id}
        return {
            "status": "pass", "action": "existing", "run_id": run_id,
            "structure_count": existing["structure_count"], "shard_count": existing["shard_count"],
            "plan_sha256": existing["plan_sha256"], "purpose": purpose,
        }
    if run_root.exists() and any(run_root.iterdir()):
        return {"status": "fail", "reason": "run_directory_exists_without_plan", "run_id": run_id}

    for name in ("grids", "peaks", "manifests", "completed"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    config_path_out = run_root / "xrd_config.json"
    _atomic_write_json(config_path_out, config)
    source_cache = run_root / "structure_payloads.parquet"
    _write_structure_payload_cache(source, selected, source_cache)

    structure_ids = selected["dataset_structure_id"].astype(str).tolist()
    structure_hashes = selected["structure_sha256"].astype(str).tolist()
    source_rows = selected["source_row"].astype(int).tolist()
    shard_count = math.ceil(len(selected) / shard_size)
    shards = []
    for index in range(shard_count):
        start = index * shard_size
        stop = min(start + shard_size, len(selected))
        shards.append({
            "index": index, "start": start, "stop": stop, "count": stop - start,
            "structure_ids": structure_ids[start:stop],
            "structure_hashes": structure_hashes[start:stop],
            "source_rows": source_rows[start:stop],
        })
    plan: dict[str, Any] = {
        "schema_version": 2, "run_id": run_id, "subset": subset,
        "purpose": purpose,
        "label_binding": "deferred" if purpose == "xrd-artifact" else "frozen",
        "clean_only": True, "structure_count": len(selected), "shard_size": shard_size,
        "shard_count": shard_count, "points_per_pattern": points,
        "source_csv_name": source.name, "source_sha256": source_sha256,
        "source_payload_sha256": _sha256_file(source_cache),
        "config_sha256": _sha256_file(config_path_out),
        "code_sha256": _code_sha256(Path(__file__).resolve().parents[1]),
        "structure_ids": structure_ids, "structure_hashes": structure_hashes,
        "shards": shards,
    }
    plan["plan_sha256"] = _mapping_sha256(plan)
    _atomic_write_json(plan_path, plan)
    return {
        "status": "pass", "action": "created", "run_id": run_id,
        "structure_count": len(selected), "shard_count": shard_count,
        "points_per_pattern": points, "plan_sha256": plan["plan_sha256"],
        "purpose": purpose,
    }


def simulate_production_shard(
    *, dataset_dir: str | Path, run_id: str, shard_index: int,
    resume: bool = False, overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one deterministic shard and publish its completion marker last."""

    dataset = Path(dataset_dir).resolve()
    run_root = _run_root(dataset, run_id)
    plan = _load_and_validate_plan(run_root)
    if shard_index < 0 or shard_index >= int(plan["shard_count"]):
        return {"status": "fail", "reason": "shard_index_out_of_range", "shard_index": shard_index}
    marker = run_root / "completed" / f"shard_{shard_index:04d}.json"
    if marker.exists() and resume:
        valid, issues = _validate_shard(run_root, plan, shard_index)
        if valid:
            return {"status": "pass", "action": "skipped", "run_id": run_id, "shard_index": shard_index}
        if not overwrite:
            return {"status": "fail", "reason": "existing_shard_invalid", "issues": issues, "shard_index": shard_index}
    if marker.exists() and not overwrite:
        return {"status": "fail", "reason": "shard_already_exists", "shard_index": shard_index}

    shard = plan["shards"][shard_index]
    structures = pd.read_parquet(dataset / "manifests" / "structures.parquet")
    structures = structures.set_index("dataset_structure_id").loc[shard["structure_ids"]].reset_index()
    if structures["structure_sha256"].astype(str).tolist() != shard["structure_hashes"]:
        return {"status": "fail", "reason": "structure_manifest_changed", "shard_index": shard_index}
    payloads = pd.read_parquet(
        run_root / "structure_payloads.parquet",
        filters=[("source_row", "in", [int(v) for v in shard["source_rows"]])],
    )
    records = {int(row["source_row"]): row for row in payloads.to_dict(orient="records")}
    if set(records) != set(int(v) for v in shard["source_rows"]):
        return {"status": "fail", "reason": "source_payload_rows_missing", "shard_index": shard_index}
    config = _read_json(run_root / "xrd_config.json")
    split_map = _split_map(dataset / "manifests" / "splits.parquet")
    purpose = str(plan.get("purpose", "benchmark-release"))
    defer_labels = purpose == "xrd-artifact"

    grids: list[np.ndarray] = []
    sample_ids: list[str] = []
    peak_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    two_theta: np.ndarray | None = None
    for row in structures.to_dict(orient="records"):
        structure = _structure_from_record(records[int(row["source_row"])])
        pattern = _xrd_pattern(structure, config)
        grid = _broadened_grid(pattern, config)
        if two_theta is None:
            two_theta = grid["two_theta"].to_numpy(dtype=np.float32)
        intensity = grid["intensity"].to_numpy(dtype=np.float32)
        xrd_id = f"xrd_{row['dataset_structure_id']}_clean"
        local_peaks = _phase_peak_table(pattern, row)
        if defer_labels:
            for peak in local_peaks:
                peak["reflection_class"] = "not_assigned"
                peak["is_superlattice_candidate"] = False
        for peak in local_peaks:
            peak_rows.append({"xrd_sample_id": xrd_id, **peak})
        grid_index = len(grids)
        grids.append(intensity)
        sample_ids.append(xrd_id)
        super_count = sum(peak["reflection_class"] == "superlattice" for peak in local_peaks)
        weak_count = sum(float(peak["relative_intensity"]) < 1.0 for peak in local_peaks)
        relative_root = f"xrd/runs/{run_id}"
        sample_rows.append({
            "xrd_sample_id": xrd_id, "dataset_structure_id": row["dataset_structure_id"],
            "fid": row["fid"], "formula": row["formula"], "chemical_system": row["chemical_system"],
            "structure_family": "unassigned" if defer_labels else row["structure_family"],
            "parent_lattice": "unassigned" if defer_labels else row["parent_lattice"],
            "benchmark_subset": "unassigned" if defer_labels else row["benchmark_subset"],
            "variant": "clean",
            "xrd_config_id": config["xrd_config_id"], "two_theta_min": config["two_theta_min"],
            "two_theta_max": config["two_theta_max"], "step": config["step"],
            "wavelength": config["radiation"],
            "broadening": f"{config['broadening']['kind']}_fwhm_{config['broadening']['fwhm_deg']}",
            "grid_shard_path": f"{relative_root}/grids/grids_{shard_index:04d}.h5",
            "grid_row_index": grid_index,
            "peak_shard_path": f"{relative_root}/peaks/peaks_{shard_index:04d}.parquet",
            "split": (
                "unassigned" if defer_labels
                else split_map.get(str(row["dataset_structure_id"]), "unassigned")
            ),
            "superlattice_peak_count": super_count, "weak_peak_count": weak_count,
            "peak_count": len(local_peaks), "perturbation_json": "{}",
        })

    grid_path = run_root / "grids" / f"grids_{shard_index:04d}.h5"
    peak_path = run_root / "peaks" / f"peaks_{shard_index:04d}.parquet"
    sample_path = run_root / "manifests" / f"samples_{shard_index:04d}.parquet"
    if not marker.exists() and not overwrite and any(
        path.exists() for path in (grid_path, peak_path, sample_path)
    ):
        return {
            "status": "fail", "reason": "unmarked_shard_artifacts_exist",
            "run_id": run_id, "shard_index": shard_index,
        }
    _atomic_write_h5(grid_path, two_theta, grids, sample_ids)
    _atomic_write_parquet(peak_path, pd.DataFrame(peak_rows))
    _atomic_write_parquet(sample_path, pd.DataFrame(sample_rows, columns=XRD_COLUMNS))
    marker_payload = {
        "schema_version": 1, "run_id": run_id, "shard_index": shard_index,
        "plan_sha256": plan["plan_sha256"], "sample_count": len(sample_rows),
        "grid_sha256": _sha256_file(grid_path), "peaks_sha256": _sha256_file(peak_path),
        "samples_sha256": _sha256_file(sample_path),
    }
    _atomic_write_json(marker, marker_payload)
    return {
        "status": "pass", "action": "generated", "run_id": run_id,
        "shard_index": shard_index, "xrd_sample_count": len(sample_rows),
    }


def finalize_production_run(*, dataset_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Validate every planned shard and publish the merged immutable run manifest."""

    dataset = Path(dataset_dir).resolve()
    run_root = _run_root(dataset, run_id)
    plan = _load_and_validate_plan(run_root)
    missing: list[int] = []
    invalid: list[int] = []
    issues: dict[str, list[str]] = {}
    fragments: list[pd.DataFrame] = []
    for shard_index in range(int(plan["shard_count"])):
        marker = run_root / "completed" / f"shard_{shard_index:04d}.json"
        if not marker.exists():
            missing.append(shard_index)
            continue
        valid, shard_issues = _validate_shard(run_root, plan, shard_index)
        if not valid:
            invalid.append(shard_index)
            issues[str(shard_index)] = shard_issues
            continue
        fragments.append(pd.read_parquet(run_root / "manifests" / f"samples_{shard_index:04d}.parquet"))
    if missing or invalid:
        return {
            "status": "fail", "reason": "run_incomplete_or_invalid", "run_id": run_id,
            "missing_shards": missing, "invalid_shards": invalid, "issues": issues,
        }

    samples = pd.concat(fragments, ignore_index=True).sort_values("dataset_structure_id", kind="stable")
    actual_ids = samples["dataset_structure_id"].astype(str).tolist()
    if len(samples) != int(plan["structure_count"]) or len(set(actual_ids)) != len(actual_ids):
        return {"status": "fail", "reason": "merged_sample_count_or_uniqueness_invalid"}
    if actual_ids != sorted(plan["structure_ids"]):
        return {"status": "fail", "reason": "merged_structure_ids_do_not_match_plan"}
    merged_path = run_root / "xrd_samples.parquet"
    _atomic_write_parquet(merged_path, samples)

    artifact_paths = _run_artifact_paths(run_root, int(plan["shard_count"]), merged_path)
    checksum_lines = [f"{_sha256_file(path)}  {path.relative_to(run_root).as_posix()}" for path in artifact_paths]
    _atomic_write_text(run_root / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    report = {
        "status": "pass", "run_id": run_id, "structure_count": int(plan["structure_count"]),
        "xrd_sample_count": len(samples), "shard_count": int(plan["shard_count"]),
        "points_per_pattern": int(plan["points_per_pattern"]), "clean_only": True,
        "plan_sha256": plan["plan_sha256"],
        "purpose": str(plan.get("purpose", "benchmark-release")),
        "label_binding": str(plan.get("label_binding", "frozen")),
    }
    _atomic_write_json(run_root / "run_summary.json", report)
    return report


def audit_production_run(*, dataset_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Audit a finalized run without changing its artifacts."""

    dataset = Path(dataset_dir).resolve()
    run_root = _run_root(dataset, run_id)
    try:
        plan = _load_and_validate_plan(run_root, require_code_match=False)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "fail", "run_integrity": False, "issues": [str(exc)]}
    issues: list[str] = []
    for shard_index in range(int(plan["shard_count"])):
        valid, shard_issues = _validate_shard(run_root, plan, shard_index)
        if not valid:
            issues.extend(f"shard_{shard_index:04d}:{issue}" for issue in shard_issues)
    merged_path = run_root / "xrd_samples.parquet"
    if not merged_path.exists():
        issues.append("missing_merged_manifest")
        xrd_rows = 0
    else:
        samples = pd.read_parquet(merged_path)
        xrd_rows = len(samples)
        if xrd_rows != int(plan["structure_count"]):
            issues.append("merged_manifest_count_mismatch")
        if samples["dataset_structure_id"].duplicated().any():
            issues.append("duplicate_structure_ids")
    checksum_path = run_root / "checksums.sha256"
    if not checksum_path.exists():
        issues.append("missing_checksums")
    else:
        issues.extend(_verify_checksum_file(run_root, checksum_path, int(plan["shard_count"])))
    current_code_sha256 = _code_sha256(Path(__file__).resolve().parents[1])
    code_matches_production = current_code_sha256 == plan["code_sha256"]
    report = {
        "status": "pass" if not issues else "fail", "run_integrity": not issues,
        "run_id": run_id, "xrd_rows": xrd_rows,
        "structure_count": int(plan["structure_count"]),
        "shard_count": int(plan["shard_count"]),
        "points_per_pattern": int(plan["points_per_pattern"]), "issues": issues,
        "purpose": str(plan.get("purpose", "benchmark-release")),
        "label_binding": str(plan.get("label_binding", "frozen")),
        "code_reproducibility": {
            "status": "pass" if code_matches_production else "version_mismatch",
            "production_code_sha256": plan["code_sha256"],
            "current_code_sha256": current_code_sha256,
        },
    }
    _atomic_write_json(dataset / "reports" / f"xrd_run_{run_id}_audit.json", report)
    return report


def _select_structures(dataset: Path, structures: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "pilot":
        pilot_ids = set(pd.read_parquet(dataset / "manifests" / "pilot.parquet")["dataset_structure_id"])
        selected = structures[structures["dataset_structure_id"].isin(pilot_ids)]
    elif subset == "all":
        selected = structures[structures["xrd_eligible"]]
    else:
        selected = structures[(structures["benchmark_subset"] == subset) & structures["xrd_eligible"]]
    return selected.sort_values("dataset_structure_id", kind="stable").reset_index(drop=True)


def _write_structure_payload_cache(source: Path, selected: pd.DataFrame, output: Path) -> None:
    wanted = set(int(value) for value in selected["source_row"])
    writer: pq.ParquetWriter | None = None
    source_row = 0
    found = 0
    temp = _temp_path(output)
    try:
        for chunk in pd.read_csv(source, usecols=["structure_as_dict"], chunksize=500):
            rows = []
            for payload in chunk["structure_as_dict"].tolist():
                if source_row in wanted:
                    rows.append({"source_row": source_row, "structure_as_dict": payload})
                    found += 1
                source_row += 1
            if rows:
                table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        if found != len(wanted):
            raise ValueError(f"Expected {len(wanted)} source rows, found {found}")
        os.replace(temp, output)
    finally:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)


def _validate_shard(run_root: Path, plan: Mapping[str, Any], shard_index: int) -> tuple[bool, list[str]]:
    issues: list[str] = []
    marker_path = run_root / "completed" / f"shard_{shard_index:04d}.json"
    grid_path = run_root / "grids" / f"grids_{shard_index:04d}.h5"
    peak_path = run_root / "peaks" / f"peaks_{shard_index:04d}.parquet"
    sample_path = run_root / "manifests" / f"samples_{shard_index:04d}.parquet"
    if not marker_path.exists():
        return False, ["missing_completion_marker"]
    marker = _read_json(marker_path)
    if marker.get("plan_sha256") != plan["plan_sha256"]:
        issues.append("plan_checksum_mismatch")
    for path, key in ((grid_path, "grid_sha256"), (peak_path, "peaks_sha256"), (sample_path, "samples_sha256")):
        if not path.exists():
            issues.append(f"missing_{path.name}")
        elif _sha256_file(path) != marker.get(key):
            issues.append(f"checksum_mismatch_{path.name}")
    if issues:
        return False, issues
    shard = plan["shards"][shard_index]
    samples = pd.read_parquet(sample_path)
    if len(samples) != int(shard["count"]):
        issues.append("sample_count_mismatch")
    if samples["dataset_structure_id"].astype(str).tolist() != shard["structure_ids"]:
        issues.append("structure_id_order_mismatch")
    try:
        with h5py.File(grid_path, "r") as handle:
            values = handle["intensity"][:]
            ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["xrd_sample_id"][:]]
            if values.shape != (int(shard["count"]), int(plan["points_per_pattern"])):
                issues.append("grid_shape_mismatch")
            elif not np.isfinite(values).all() or float(values.min()) < 0:
                issues.append("invalid_grid_values")
            elif not np.allclose(values.max(axis=1), 100.0, atol=1e-4):
                issues.append("invalid_grid_normalization")
            if ids != samples["xrd_sample_id"].astype(str).tolist():
                issues.append("hdf5_sample_ids_mismatch")
            if len(handle["two_theta"]) != int(plan["points_per_pattern"]):
                issues.append("two_theta_width_mismatch")
    except (OSError, KeyError, ValueError) as exc:
        issues.append(f"invalid_hdf5:{type(exc).__name__}")
    return not issues, issues


def _load_and_validate_plan(run_root: Path, *, require_code_match: bool = True) -> dict[str, Any]:
    plan = _read_json(run_root / "run_plan.json")
    expected = plan.pop("plan_sha256", None)
    actual = _mapping_sha256(plan)
    plan["plan_sha256"] = expected
    if expected != actual:
        raise ValueError("run_plan_checksum_mismatch")
    config_path = run_root / "xrd_config.json"
    payload_path = run_root / "structure_payloads.parquet"
    if _sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("xrd_config_checksum_mismatch")
    if _sha256_file(payload_path) != plan["source_payload_sha256"]:
        raise ValueError("source_payload_checksum_mismatch")
    if require_code_match and _code_sha256(Path(__file__).resolve().parents[1]) != plan["code_sha256"]:
        raise ValueError("production_code_checksum_mismatch")
    return plan


def _run_artifact_paths(run_root: Path, shard_count: int, merged_path: Path) -> list[Path]:
    paths = [
        run_root / "run_plan.json", run_root / "xrd_config.json",
        run_root / "structure_payloads.parquet", merged_path,
    ]
    for shard_index in range(shard_count):
        paths.extend([
            run_root / "grids" / f"grids_{shard_index:04d}.h5",
            run_root / "peaks" / f"peaks_{shard_index:04d}.parquet",
            run_root / "manifests" / f"samples_{shard_index:04d}.parquet",
            run_root / "completed" / f"shard_{shard_index:04d}.json",
        ])
    return paths


def _verify_checksum_file(run_root: Path, checksum_path: Path, shard_count: int) -> list[str]:
    issues: list[str] = []
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            issues.append("invalid_checksum_line")
            continue
        entries[parts[1]] = parts[0]
    expected_paths = _run_artifact_paths(run_root, shard_count, run_root / "xrd_samples.parquet")
    expected_names = {path.relative_to(run_root).as_posix() for path in expected_paths}
    if set(entries) != expected_names:
        issues.append("checksum_entry_set_mismatch")
    for name, expected_digest in entries.items():
        path = (run_root / name).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError:
            issues.append(f"checksum_path_outside_run:{name}")
            continue
        if not path.exists():
            issues.append(f"checksum_path_missing:{name}")
        elif _sha256_file(path) != expected_digest:
            issues.append(f"checksum_mismatch:{name}")
    return issues


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.resolve())
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing XRD config keys: {missing}")
    if config["broadening"].get("kind") != "gaussian":
        raise ValueError("Only gaussian broadening is supported")
    if config["intensity_normalization"].get("method") != "max":
        raise ValueError("Only max intensity normalization is supported")
    return config


def _config_points(config: Mapping[str, Any]) -> int:
    start = float(config["two_theta_min"])
    stop = float(config["two_theta_max"])
    step = float(config["step"])
    if step <= 0 or stop <= start:
        raise ValueError("Invalid two-theta range or step")
    intervals = (stop - start) / step
    if not math.isclose(intervals, round(intervals), abs_tol=1e-8):
        raise ValueError("XRD range must be divisible by step")
    return int(round(intervals)) + 1


def _resolve_source_csv(dataset: Path, summary: Mapping[str, Any]) -> Path:
    source_value = str(summary.get("source_csv", ""))
    source_name = _source_csv_name(summary)
    candidates = [
        Path(str(summary.get("source_csv_local_path", ""))),
        Path(source_value), dataset / "source" / source_name, dataset / source_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Source CSV not found; expected {source_name} under dataset/source")


def _source_csv_name(summary: Mapping[str, Any]) -> str:
    explicit = str(summary.get("source_csv_name", "")).strip()
    if explicit:
        return explicit
    source_value = str(summary.get("source_csv", ""))
    return PureWindowsPath(source_value).name if "\\" in source_value else Path(source_value).name


def _run_root(dataset: Path, run_id: str) -> Path:
    _validate_run_id(run_id)
    return dataset / "xrd" / "runs" / run_id


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen")


def _code_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "alloy_xrd_bench").glob("*.py"))
    paths.extend(path for path in [project_root / "pyproject.toml"] if path.exists())
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _atomic_write_h5(path: Path, two_theta: np.ndarray | None, grids: list[np.ndarray], ids: list[str]) -> None:
    if two_theta is None or not grids:
        raise ValueError("Cannot write an empty XRD shard")
    temp = _temp_path(path)
    try:
        with h5py.File(temp, "w") as handle:
            handle.create_dataset("two_theta", data=two_theta, compression="gzip")
            handle.create_dataset("intensity", data=np.vstack(grids), compression="gzip", chunks=True)
            handle.create_dataset("xrd_sample_id", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = _temp_path(path)
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(path)
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
