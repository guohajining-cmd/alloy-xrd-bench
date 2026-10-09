"""Staged processing for occupancy-aware HEA XRD benchmark datasets."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .contract import DEFAULT_XRD_CONFIG, canonical_structure_hash
from .xrd import _broadened_grid, _peak_table, _xrd_pattern


INVENTORY_COLUMNS = [
    "dataset_structure_id", "fid", "source_row", "formula", "chemical_system",
    "elements_json", "nelements", "natoms", "source_lattice", "source_space_group_number",
    "volume_per_atom", "e_per_atom", "formation_energy_per_atom", "e_above_hull",
    "structure_sha256", "duplicate_of", "quality_tier", "quality_notes",
]
STRUCTURE_COLUMNS = INVENTORY_COLUMNS + [
    "structure_family", "parent_lattice", "classification_confidence",
    "template_mapping_rms", "standardized_space_group_number", "primitive_structure_sha256",
    "conventional_structure_sha256", "occupancy_entropy", "occupancy_fingerprint",
    "l12_order_parameter", "b2_order_parameter", "corner_element_distribution_json",
    "face_element_distribution_json", "b2_a_distribution_json", "b2_b_distribution_json",
    "sro_shell1_json", "sro_shell2_json", "benchmark_subset", "xrd_eligible",
]
SITE_COLUMNS = [
    "dataset_structure_id", "fid", "site_index", "element", "frac_x", "frac_y", "frac_z",
    "equivalent_site_group", "parent_lattice_site", "coordination_number", "local_environment",
]
GROUP_COLUMNS = [
    "dataset_structure_id", "site_group_id", "group_kind", "multiplicity",
    "element_distribution_json", "occupancy_entropy", "dominant_element",
    "sro_shell1_json", "sro_shell2_json",
]
XRD_COLUMNS = [
    "xrd_sample_id", "dataset_structure_id", "fid", "formula", "chemical_system",
    "structure_family", "parent_lattice", "benchmark_subset", "variant", "xrd_config_id",
    "two_theta_min", "two_theta_max", "step", "wavelength", "broadening",
    "grid_shard_path", "grid_row_index", "peak_shard_path", "split",
    "superlattice_peak_count", "weak_peak_count", "peak_count", "perturbation_json",
]


def inventory_dataset(
    *, csv_path: str | Path, output_dir: str | Path, limit: int | None = None,
    min_neighbor_distance: float = 1.5, min_volume_per_atom: float = 5.0,
    max_volume_per_atom: float = 30.0,
) -> dict[str, Any]:
    """Stream the source CSV into an accounted structure inventory."""

    source = Path(csv_path).resolve()
    output = Path(output_dir).resolve()
    manifests = output / "manifests"
    reports = output / "reports"
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    source_row = 0

    for chunk in pd.read_csv(source, chunksize=500):
        for record in chunk.to_dict(orient="records"):
            if limit is not None and source_row >= limit:
                break
            row = _inventory_row(
                record, source_row, seen_hashes, min_neighbor_distance,
                min_volume_per_atom, max_volume_per_atom,
            )
            rows.append(row)
            if row["quality_tier"] != "pass":
                rejected.append(row)
            source_row += 1
        if limit is not None and source_row >= limit:
            break

    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    inventory.to_parquet(manifests / "structure_inventory.parquet", index=False)
    inventory.to_parquet(manifests / "structure_quality.parquet", index=False)
    pd.DataFrame(rejected, columns=INVENTORY_COLUMNS).to_parquet(
        manifests / "rejected_structures.parquet", index=False
    )
    summary = {
        "status": "pass" if rows else "fail",
        "source_csv": source.name,
        "source_csv_name": source.name,
        "source_csv_local_path": str(source),
        "source_sha256": _sha256_file(source),
        "source_row_count": len(rows),
        "accounted_row_count": len(rows),
        "accepted_count": len(rows) - len(rejected),
        "rejected_count": len(rejected),
        "duplicate_count": sum(bool(row["duplicate_of"]) for row in rows),
        "limited": limit is not None,
        "thresholds": {
            "min_neighbor_distance": min_neighbor_distance,
            "min_volume_per_atom": min_volume_per_atom,
            "max_volume_per_atom": max_volume_per_atom,
        },
    }
    _write_json(reports / "inventory_summary.json", summary)
    return summary


def label_dataset(
    *, dataset_dir: str | Path, pilot_size: int = 1000, symprec: float = 0.05,
) -> dict[str, Any]:
    """Generate parent-lattice, occupancy, site-group, and pilot labels."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    reports = dataset / "reports"
    inventory = pd.read_parquet(manifests / "structure_inventory.parquet")
    inventory_summary = _read_json(reports / "inventory_summary.json")
    source = _resolve_inventory_source(dataset, inventory_summary)
    inventory_by_row = {
        int(row["source_row"]): row for row in inventory.to_dict(orient="records")
        if row["quality_tier"] == "pass"
    }
    structure_rows: list[dict[str, Any]] = []
    site_writer: pq.ParquetWriter | None = None
    group_writer: pq.ParquetWriter | None = None
    source_row = 0

    try:
        for chunk in pd.read_csv(source, chunksize=250):
            site_batch: list[dict[str, Any]] = []
            group_batch: list[dict[str, Any]] = []
            for record in chunk.to_dict(orient="records"):
                base = inventory_by_row.get(source_row)
                source_row += 1
                if base is None:
                    continue
                structure = _structure_from_record(record)
                labeled, sites, groups = _label_structure(base, structure, symprec)
                structure_rows.append(labeled)
                site_batch.extend(sites)
                group_batch.extend(groups)
            if site_batch:
                site_writer = _append_parquet_batch(
                    manifests / "occupancy_sites.parquet", site_batch, SITE_COLUMNS, site_writer
                )
            if group_batch:
                group_writer = _append_parquet_batch(
                    manifests / "occupancy_site_groups.parquet", group_batch, GROUP_COLUMNS, group_writer
                )
    finally:
        if site_writer is not None:
            site_writer.close()
        if group_writer is not None:
            group_writer.close()

    structures = pd.DataFrame(structure_rows, columns=STRUCTURE_COLUMNS)
    structures.to_parquet(manifests / "structures.parquet", index=False)
    pilot = _balanced_pilot(structures, pilot_size)
    pilot.to_parquet(manifests / "pilot.parquet", index=False)
    review = pilot[[
        "dataset_structure_id", "fid", "formula", "chemical_system", "structure_family",
        "parent_lattice", "classification_confidence", "benchmark_subset", "pilot_bucket",
    ]].copy()
    review["review_status"] = "pending"
    review["reviewer_label"] = ""
    review["review_notes"] = ""
    review.to_parquet(manifests / "pilot_review_queue.parquet", index=False)

    family_counts = structures["structure_family"].value_counts().to_dict()
    subset_counts = structures["benchmark_subset"].value_counts().to_dict()
    summary = {
        "status": "pass" if len(structures) else "fail",
        "labeled_count": len(structures),
        "pilot_count": len(pilot),
        "family_counts": {str(k): int(v) for k, v in family_counts.items()},
        "subset_counts": {str(k): int(v) for k, v in subset_counts.items()},
        "manual_review_status": "pending",
        "symprec": symprec,
    }
    _write_json(reports / "label_summary.json", summary)
    return summary


def simulate_dataset(
    *, dataset_dir: str | Path, subset: str = "pilot", shard_size: int = 256,
    split: str = "unassigned", augmentations: int = 0,
) -> dict[str, Any]:
    """Generate clean and optional augmented XRD into HDF5/Parquet shards."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    reports = dataset / "reports"
    source = _resolve_inventory_source(dataset, _read_json(reports / "inventory_summary.json"))
    structures = pd.read_parquet(manifests / "structures.parquet")
    if subset == "pilot":
        selected_ids = set(pd.read_parquet(manifests / "pilot.parquet")["dataset_structure_id"])
        selected = structures[structures["dataset_structure_id"].isin(selected_ids)].copy()
    elif subset == "all":
        selected = structures[structures["xrd_eligible"]].copy()
    else:
        selected = structures[
            (structures["benchmark_subset"] == subset) & structures["xrd_eligible"]
        ].copy()
    selected = selected.sort_values("dataset_structure_id").reset_index(drop=True)
    if selected.empty:
        report = {"status": "fail", "reason": "empty_subset", "subset": subset, "xrd_sample_count": 0}
        _write_json(reports / "xrd_simulation_summary.json", report)
        return report

    records_by_row = _load_source_records(source, set(int(v) for v in selected["source_row"]))
    xrd_root = dataset / "xrd"
    xrd_root.mkdir(parents=True, exist_ok=True)
    xrd_rows: list[dict[str, Any]] = []
    shard_count = math.ceil(len(selected) / shard_size)
    split_map = _split_map(manifests / "splits.parquet")

    for shard_index in range(shard_count):
        shard = selected.iloc[shard_index * shard_size:(shard_index + 1) * shard_size]
        grids: list[np.ndarray] = []
        ids: list[str] = []
        peak_rows: list[dict[str, Any]] = []
        local_rows: list[dict[str, Any]] = []
        two_theta: np.ndarray | None = None
        for _, row in shard.iterrows():
            structure = _structure_from_record(records_by_row[int(row["source_row"])])
            pattern = _xrd_pattern(structure)
            grid = _broadened_grid(pattern)
            if two_theta is None:
                two_theta = grid["two_theta"].to_numpy(dtype=np.float32)
            clean = grid["intensity"].to_numpy(dtype=np.float32)
            variants = [("clean", clean, {})]
            for aug_index in range(augmentations):
                params = _augmentation_parameters(str(row["structure_sha256"]), aug_index)
                variants.append((f"aug_{aug_index:02d}", _augment_grid(clean, params), params))
            base_peaks = _phase_peak_table(pattern, row.to_dict())
            for variant, intensity, params in variants:
                xrd_id = f"xrd_{row['dataset_structure_id']}_{variant}"
                grid_row = len(grids)
                grids.append(intensity)
                ids.append(xrd_id)
                if variant == "clean":
                    for peak in base_peaks:
                        peak_rows.append({"xrd_sample_id": xrd_id, **peak})
                peak_count = len(base_peaks)
                super_count = sum(p["reflection_class"] == "superlattice" for p in base_peaks)
                weak_count = sum(float(p["relative_intensity"]) < 1.0 for p in base_peaks)
                local_rows.append({
                    "xrd_sample_id": xrd_id,
                    "dataset_structure_id": row["dataset_structure_id"],
                    "fid": row["fid"], "formula": row["formula"],
                    "chemical_system": row["chemical_system"],
                    "structure_family": row["structure_family"],
                    "parent_lattice": row["parent_lattice"],
                    "benchmark_subset": row["benchmark_subset"], "variant": variant,
                    "xrd_config_id": DEFAULT_XRD_CONFIG["xrd_config_id"],
                    "two_theta_min": DEFAULT_XRD_CONFIG["two_theta_min"],
                    "two_theta_max": DEFAULT_XRD_CONFIG["two_theta_max"],
                    "step": DEFAULT_XRD_CONFIG["step"], "wavelength": DEFAULT_XRD_CONFIG["radiation"],
                    "broadening": "gaussian_fwhm_0p1",
                    "grid_shard_path": f"xrd/grids_{shard_index:04d}.h5",
                    "grid_row_index": grid_row,
                    "peak_shard_path": f"xrd/peaks_{shard_index:04d}.parquet",
                    "split": split_map.get(str(row["dataset_structure_id"]), split),
                    "superlattice_peak_count": super_count, "weak_peak_count": weak_count,
                    "peak_count": peak_count, "perturbation_json": json.dumps(params, sort_keys=True),
                })
        h5_path = xrd_root / f"grids_{shard_index:04d}.h5"
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("two_theta", data=two_theta, compression="gzip")
            handle.create_dataset("intensity", data=np.vstack(grids), compression="gzip", chunks=True)
            handle.create_dataset("xrd_sample_id", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
        pd.DataFrame(peak_rows).to_parquet(xrd_root / f"peaks_{shard_index:04d}.parquet", index=False)
        xrd_rows.extend(local_rows)

    pd.DataFrame(xrd_rows, columns=XRD_COLUMNS).to_parquet(manifests / "xrd_samples.parquet", index=False)
    report = {
        "status": "pass", "subset": subset, "structure_count": len(selected),
        "xrd_sample_count": len(xrd_rows), "shard_count": shard_count,
        "points_per_pattern": 4001, "augmentations_per_structure": augmentations,
    }
    _write_json(reports / "xrd_simulation_summary.json", report)
    return report


def export_review_batches(
    *, dataset_dir: str | Path, output_dir: str | Path,
    limit: int | None = None, batch_size: int = 50,
) -> dict[str, Any]:
    """Export deterministic pilot-review batches with CIF and XRD evidence."""

    dataset = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    manifests = dataset / "manifests"
    if batch_size < 1 or (limit is not None and limit < 1):
        return {"status": "fail", "reason": "limit_and_batch_size_must_be_positive"}

    review = pd.read_parquet(manifests / "pilot_review_queue.parquet")
    structures = pd.read_parquet(manifests / "structures.parquet")
    detail_columns = [
        "dataset_structure_id", "source_row", "natoms", "source_space_group_number",
        "standardized_space_group_number", "template_mapping_rms", "occupancy_entropy",
        "l12_order_parameter", "b2_order_parameter", "corner_element_distribution_json",
        "face_element_distribution_json", "b2_a_distribution_json", "b2_b_distribution_json",
    ]
    selected = review.merge(structures[detail_columns], on="dataset_structure_id", how="left")
    selected["review_priority"] = selected.apply(_review_priority, axis=1)
    selected["priority_reason"] = selected.apply(_review_priority_reason, axis=1)
    selected = selected.sort_values(
        ["review_priority", "classification_confidence", "dataset_structure_id"],
        kind="stable",
    )
    if limit is not None:
        selected = selected.head(limit)
    selected = selected.reset_index(drop=True)

    source = _resolve_inventory_source(
        dataset, _read_json(dataset / "reports" / "inventory_summary.json")
    )
    records = _load_source_records(source, set(int(value) for value in selected["source_row"]))
    xrd_path = manifests / "xrd_samples.parquet"
    clean_xrd: dict[str, dict[str, Any]] = {}
    if xrd_path.exists():
        samples = pd.read_parquet(xrd_path)
        samples = samples[samples["variant"].eq("clean")]
        clean_xrd = {
            str(row["dataset_structure_id"]): row
            for row in samples.to_dict(orient="records")
        }

    output.mkdir(parents=True, exist_ok=True)
    batch_count = math.ceil(len(selected) / batch_size)
    for batch_index in range(batch_count):
        batch = selected.iloc[batch_index * batch_size:(batch_index + 1) * batch_size].copy()
        batch_dir = output / f"batch_{batch_index:03d}"
        cif_dir = batch_dir / "cif"
        preview_dir = batch_dir / "xrd"
        cif_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        cif_paths: list[str] = []
        preview_paths: list[str] = []
        for _, row in batch.iterrows():
            structure_id = str(row["dataset_structure_id"])
            structure = _structure_from_record(records[int(row["source_row"])])
            cif_path = cif_dir / f"{structure_id}.cif"
            structure.to(filename=str(cif_path))
            cif_paths.append(cif_path.relative_to(output).as_posix())

            sample = clean_xrd.get(structure_id)
            preview_path = preview_dir / f"{structure_id}.png"
            if sample is not None:
                _write_xrd_preview(dataset, sample, preview_path, row.to_dict())
                preview_paths.append(preview_path.relative_to(output).as_posix())
            else:
                preview_paths.append("")
        batch["cif_path"] = cif_paths
        batch["xrd_preview_path"] = preview_paths
        batch.to_csv(batch_dir / "review.csv", index=False)
        _write_review_html(batch_dir, batch, output)

    report = {
        "status": "pass" if len(selected) else "fail",
        "selected_count": len(selected),
        "batch_count": batch_count,
        "pending_count": int(selected["review_status"].eq("pending").sum()),
        "output_dir": str(output),
    }
    _write_json(output / "review_export_summary.json", report)
    return report


def apply_review_results(
    *, dataset_dir: str | Path, input_path: str | Path,
) -> dict[str, Any]:
    """Validate and merge edited review CSV rows into the canonical pilot queue."""

    dataset = Path(dataset_dir).resolve()
    source = Path(input_path).resolve()
    queue_path = dataset / "manifests" / "pilot_review_queue.parquet"
    queue = pd.read_parquet(queue_path)
    updates = pd.read_csv(source, keep_default_na=False)
    required = {"dataset_structure_id", "review_status", "reviewer_label", "review_notes"}
    missing = sorted(required - set(updates.columns))
    if missing:
        return {"status": "fail", "missing_columns": missing, "updated_count": 0}
    duplicate_ids = sorted(
        updates.loc[updates["dataset_structure_id"].duplicated(), "dataset_structure_id"].astype(str).unique()
    )
    known_ids = set(queue["dataset_structure_id"].astype(str))
    unknown_ids = sorted(set(updates["dataset_structure_id"].astype(str)) - known_ids)
    allowed_statuses = {"pending", "pass", "fail"}
    invalid_statuses = sorted(set(updates["review_status"].astype(str)) - allowed_statuses)
    if duplicate_ids or unknown_ids or invalid_statuses:
        return {
            "status": "fail", "updated_count": 0,
            "duplicate_structure_ids": duplicate_ids,
            "unknown_structure_ids": unknown_ids,
            "invalid_review_statuses": invalid_statuses,
        }

    indexed = queue.set_index("dataset_structure_id")
    incoming = updates.set_index("dataset_structure_id")
    for column in ["review_status", "reviewer_label", "review_notes"]:
        indexed.loc[incoming.index, column] = incoming[column].astype(str)
    merged = indexed.reset_index()
    merged.to_parquet(queue_path, index=False)
    report = {
        "status": "pass", "updated_count": len(incoming),
        "pass_count": int(merged["review_status"].eq("pass").sum()),
        "fail_count": int(merged["review_status"].eq("fail").sum()),
        "pending_count": int(merged["review_status"].eq("pending").sum()),
    }
    _write_json(dataset / "reports" / "review_apply_summary.json", report)
    return report


def split_dataset(*, dataset_dir: str | Path) -> dict[str, Any]:
    """Create deterministic leakage-resistant benchmark split schemes."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    structures = pd.read_parquet(manifests / "structures.parquet")
    phase_split_map = _group_holdout_map(structures["structure_family"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for row in structures.to_dict(orient="records"):
        structure_id = str(row["dataset_structure_id"])
        rows.append({
            "dataset_structure_id": structure_id,
            "structure_sha256": row["structure_sha256"],
            "formula": row["formula"], "chemical_system": row["chemical_system"],
            "structure_family": row["structure_family"],
            "occupancy_fingerprint": row["occupancy_fingerprint"],
            "id_split": _hash_split(row["structure_sha256"]),
            "composition_ood_split": _hash_split(str(row["formula"])),
            "chemistry_ood_split": _hash_split(str(row["chemical_system"])),
            "occupancy_ood_split": _hash_split(str(row["occupancy_fingerprint"])),
            "phase_ood_split": phase_split_map[str(row["structure_family"])],
        })
    splits = pd.DataFrame(rows)
    splits.to_parquet(manifests / "splits.parquet", index=False)
    report = {"status": "pass" if len(splits) else "fail", "split_row_count": len(splits)}
    _write_json(dataset / "reports" / "split_summary.json", report)
    return report


def audit_dataset(*, dataset_dir: str | Path) -> dict[str, Any]:
    """Audit accounting, XRD integrity, split leakage, and readiness gates."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    reports = dataset / "reports"
    inventory = pd.read_parquet(manifests / "structure_inventory.parquet")
    rejected = pd.read_parquet(manifests / "rejected_structures.parquet")
    structures = pd.read_parquet(manifests / "structures.parquet") if (manifests / "structures.parquet").exists() else pd.DataFrame()
    summary = _read_json(reports / "inventory_summary.json")
    issues: list[dict[str, str]] = []
    if len(inventory) != int(summary["source_row_count"]):
        issues.append({"code": "inventory_count_mismatch", "message": "inventory rows do not match source count"})
    if len(structures) + len(rejected) != len(inventory):
        issues.append({"code": "accounting_mismatch", "message": "accepted plus rejected rows do not match inventory"})
    splits_path = manifests / "splits.parquet"
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        for scheme, group in [
            ("composition_ood_split", "formula"), ("chemistry_ood_split", "chemical_system"),
            ("occupancy_ood_split", "occupancy_fingerprint"), ("phase_ood_split", "structure_family"),
        ]:
            leakage = splits.groupby(group)[scheme].nunique().gt(1).sum()
            if leakage:
                issues.append({"code": f"{scheme}_leakage", "message": f"{leakage} groups cross splits"})
    xrd_path = manifests / "xrd_samples.parquet"
    xrd_rows = 0
    if xrd_path.exists():
        xrd = pd.read_parquet(xrd_path)
        xrd_rows = len(xrd)
        for shard_path, shard_rows in xrd.groupby("grid_shard_path"):
            with h5py.File(dataset / shard_path, "r") as handle:
                if handle["intensity"].shape[1] != 4001:
                    issues.append({"code": "invalid_xrd_width", "message": str(shard_path)})
                    continue
                values = handle["intensity"][:]
                if not np.isfinite(values).all() or float(values.min()) < 0:
                    issues.append({"code": "invalid_xrd_values", "message": str(shard_path)})
                maxima = values.max(axis=1)
                if not np.allclose(maxima, 100.0, atol=1e-4):
                    issues.append({"code": "invalid_xrd_normalization", "message": str(shard_path)})
                if len(values) != len(shard_rows):
                    issues.append({"code": "xrd_shard_count_mismatch", "message": str(shard_path)})
    review_path = manifests / "pilot_review_queue.parquet"
    manual_review_pass_rate = 0.0
    manual_review_complete = False
    manual_reviewed_count = 0
    manual_review_pending_count = 0
    if review_path.exists():
        review = pd.read_parquet(review_path)
        pending = review["review_status"].eq("pending")
        manual_review_pending_count = int(pending.sum())
        manual_reviewed_count = int((~pending).sum())
        manual_review_complete = len(review) > 0 and manual_review_pending_count == 0
        if manual_reviewed_count:
            manual_review_pass_rate = float(
                review.loc[~pending, "review_status"].eq("pass").mean()
            )
    engineering_pass = not issues
    benchmark_ready = engineering_pass and manual_review_complete and manual_review_pass_rate >= 0.95 and xrd_rows > 0
    report = {
        "status": "pass" if engineering_pass else "fail",
        "benchmark_ready": benchmark_ready,
        "inventory_rows": len(inventory), "accepted_rows": len(structures),
        "rejected_rows": len(rejected), "xrd_rows": xrd_rows,
        "manual_review_complete": manual_review_complete,
        "manual_reviewed_count": manual_reviewed_count,
        "manual_review_pending_count": manual_review_pending_count,
        "manual_review_pass_rate": manual_review_pass_rate, "issues": issues,
    }
    _write_json(reports / "dataset_summary.json", report)
    _write_readiness(reports / "benchmark_readiness.md", report)
    return report


def _inventory_row(
    record: Mapping[str, Any], source_row: int, seen_hashes: dict[str, str],
    min_distance: float, min_vpa: float, max_vpa: float,
) -> dict[str, Any]:
    fid = str(record.get("fid", ""))
    structure_id = f"occstr_{_safe_id(fid) or source_row}"
    notes: list[str] = []
    structure_hash = ""
    duplicate_of = ""
    elements: list[str] = []
    natoms = int(_number(record.get("NIONS"), 0))
    try:
        structure = _structure_from_record(record)
        structure_hash = canonical_structure_hash(structure.as_dict())
        elements = sorted(str(element) for element in structure.composition.elements)
        if len(structure) != natoms:
            notes.append("natoms_mismatch")
        if not np.isfinite(structure.volume) or structure.volume <= 0:
            notes.append("invalid_volume")
        vpa = structure.volume / max(len(structure), 1)
        if not min_vpa <= vpa <= max_vpa:
            notes.append("volume_per_atom_out_of_range")
        if _min_distance(structure) < min_distance:
            notes.append("min_neighbor_distance_below_threshold")
        if any(abs(float(site.species.num_atoms) - 1.0) > 1e-6 for site in structure):
            notes.append("invalid_site_occupancy")
        duplicate_of = seen_hashes.get(structure_hash, "")
        seen_hashes.setdefault(structure_hash, structure_id)
    except Exception as exc:
        notes.append(f"parse_failed:{type(exc).__name__}")
        vpa = _number(record.get("volume_per_atom"), float("nan"))
    return {
        "dataset_structure_id": structure_id, "fid": fid, "source_row": source_row,
        "formula": str(record.get("reduced_formula", "")),
        "chemical_system": str(record.get("chemical_system", "")),
        "elements_json": json.dumps(elements), "nelements": int(_number(record.get("nelements"), len(elements))),
        "natoms": natoms, "source_lattice": _clean_text(record.get("lattice")),
        "source_space_group_number": int(_number(record.get("space_group_number"), 0)),
        "volume_per_atom": _number(record.get("volume_per_atom"), vpa),
        "e_per_atom": _number(record.get("e_per_atom"), float("nan")),
        "formation_energy_per_atom": _number(record.get("Ef_per_atom"), float("nan")),
        "e_above_hull": _number(record.get("e_above_hull"), float("nan")),
        "structure_sha256": structure_hash, "duplicate_of": duplicate_of,
        "quality_tier": "pass" if not notes else "fail", "quality_notes": ";".join(notes),
    }


def _label_structure(
    base: Mapping[str, Any], structure: Structure, symprec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    family, parent, confidence, mapping_rms, special = _classify_structure(base, structure)
    distributions = _special_distributions(structure, special)
    overall = _element_distribution(structure)
    occupancy_entropy = _entropy(overall)
    sro1, sro2 = _sro(structure, parent)
    equivalent_groups, standardized = _equivalent_groups(
        structure, family, special, symprec, int(base["source_space_group_number"])
    )
    coordination = _coordination_numbers(structure, parent)
    sites: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    group_elements: dict[str, list[str]] = defaultdict(list)
    for index, site in enumerate(structure):
        element = _site_element(site)
        group_id = equivalent_groups.get(index, "all_sites")
        group_elements[group_id].append(element)
        frac = np.mod(site.frac_coords, 1.0)
        sites.append({
            "dataset_structure_id": base["dataset_structure_id"], "fid": base["fid"],
            "site_index": index, "element": element, "frac_x": float(frac[0]),
            "frac_y": float(frac[1]), "frac_z": float(frac[2]),
            "equivalent_site_group": group_id, "parent_lattice_site": group_id,
            "coordination_number": int(coordination[index]),
            "local_environment": f"{parent or 'unknown'}_cn{int(coordination[index])}",
        })
    for group_id, elems in sorted(group_elements.items()):
        dist = _distribution(elems)
        groups.append({
            "dataset_structure_id": base["dataset_structure_id"], "site_group_id": group_id,
            "group_kind": "sublattice" if family in {"l12_like", "b2_like"} else "equivalent_sites",
            "multiplicity": len(elems), "element_distribution_json": json.dumps(dist, sort_keys=True),
            "occupancy_entropy": _entropy(dist), "dominant_element": max(dist, key=dist.get),
            "sro_shell1_json": json.dumps(sro1, sort_keys=True),
            "sro_shell2_json": json.dumps(sro2, sort_keys=True),
        })
    fingerprint_payload = [(g["site_group_id"], g["element_distribution_json"]) for g in groups]
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    subset = _benchmark_subset(base, family, parent)
    l12_order = _order_parameter(distributions.get("corner", {}), distributions.get("face", {})) if family == "l12_like" else 0.0
    b2_order = _binary_order_parameter(distributions.get("b2_a", {}), distributions.get("b2_b", {})) if family == "b2_like" else 0.0
    labeled = {**base,
        "structure_family": family, "parent_lattice": parent,
        "classification_confidence": confidence, "template_mapping_rms": mapping_rms,
        "standardized_space_group_number": standardized["space_group_number"],
        "primitive_structure_sha256": standardized["primitive_hash"],
        "conventional_structure_sha256": standardized["conventional_hash"],
        "occupancy_entropy": occupancy_entropy, "occupancy_fingerprint": fingerprint,
        "l12_order_parameter": l12_order, "b2_order_parameter": b2_order,
        "corner_element_distribution_json": json.dumps(distributions.get("corner", {}), sort_keys=True),
        "face_element_distribution_json": json.dumps(distributions.get("face", {}), sort_keys=True),
        "b2_a_distribution_json": json.dumps(distributions.get("b2_a", {}), sort_keys=True),
        "b2_b_distribution_json": json.dumps(distributions.get("b2_b", {}), sort_keys=True),
        "sro_shell1_json": json.dumps(sro1, sort_keys=True),
        "sro_shell2_json": json.dumps(sro2, sort_keys=True),
        "benchmark_subset": subset, "xrd_eligible": True,
    }
    return labeled, sites, groups


def _classify_structure(
    base: Mapping[str, Any], structure: Structure,
) -> tuple[str, str, float, float, str]:
    sg = int(base["source_space_group_number"])
    source_lattice = str(base.get("source_lattice", "")).lower()
    special = _special_ordered_type(structure, sg)
    if special == "l12":
        return "l12_like", "fcc", 1.0, 0.0, special
    if special == "b2":
        return "b2_like", "bcc", 1.0, 0.0, special
    if source_lattice in {"fcc", "bcc"}:
        other_lattice = "bcc" if source_lattice == "fcc" else "fcc"
        source_rms = _parent_mapping_rms(structure, source_lattice)
        other_rms = _parent_mapping_rms(structure, other_lattice)
        parent = other_lattice if other_rms + 0.08 < source_rms else source_lattice
        rms = other_rms if parent == other_lattice else source_rms
        confidence = max(0.55, 0.98 - rms)
        if parent != source_lattice:
            confidence = min(confidence, 0.9)
        return f"{parent}_like", parent, confidence, rms, ""
    if sg in {225, 226, 227, 228}:
        return "fcc_like", "fcc", 0.9, 0.0, ""
    if sg == 229:
        return "bcc_like", "bcc", 0.9, 0.0, ""
    if 168 <= sg <= 194:
        return "hcp_like", "hcp", 0.8, 0.0, ""
    if len(structure) > 1:
        return "complex", "", 0.7, float("nan"), ""
    return "unknown", "", 0.2, float("nan"), ""


def _special_ordered_type(structure: Structure, sg: int) -> str:
    if sg != 221 or not _is_cubic(structure):
        return ""
    elements = [_site_element(site) for site in structure]
    fracs = [np.mod(site.frac_coords, 1.0) for site in structure]
    if len(structure) == 4 and sorted(Counter(elements).values()) == [1, 3]:
        labels = [_cubic_site_label(frac) for frac in fracs]
        if labels.count("corner") == 1 and labels.count("face") == 3:
            return "l12"
    if len(structure) == 2 and len(set(elements)) == 2:
        labels = {_cubic_site_label(frac) for frac in fracs}
        if labels == {"corner", "body"}:
            return "b2"
    return ""


def _special_distributions(structure: Structure, special: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    if special:
        labels = _special_site_labels(structure, special)
        for index, site in enumerate(structure):
            label = labels[index]
            grouped[label].append(_site_element(site))
    return {key: _distribution(values) for key, values in grouped.items()}


def _equivalent_groups(
    structure: Structure, family: str, special: str, symprec: float, source_space_group: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    if special:
        groups = _special_site_labels(structure, special)
    elif family in {"fcc_like", "bcc_like"}:
        groups = {index: "parent_lattice_all_sites" for index in range(len(structure))}
    else:
        groups = {}
        try:
            equivalent = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetrized_structure().equivalent_indices
            for group_index, indices in enumerate(equivalent):
                for index in indices:
                    groups[int(index)] = f"eq_{group_index:03d}"
        except Exception:
            groups = {index: "all_sites" for index in range(len(structure))}
    original_hash = canonical_structure_hash(structure.as_dict())
    standardized = {
        "space_group_number": source_space_group,
        "primitive_hash": original_hash,
        "conventional_hash": original_hash,
    }
    if family not in {"fcc_like", "bcc_like"} and len(structure) <= 64:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
            primitive = analyzer.get_primitive_standard_structure()
            conventional = analyzer.get_conventional_standard_structure()
            standardized = {
                "space_group_number": analyzer.get_space_group_number(),
                "primitive_hash": canonical_structure_hash(primitive.as_dict()),
                "conventional_hash": canonical_structure_hash(conventional.as_dict()),
            }
        except Exception:
            pass
    return groups, standardized


def _special_site_labels(structure: Structure, special: str) -> dict[int, str]:
    if special == "b2":
        labels: dict[int, str] = {}
        for index, site in enumerate(structure):
            cubic_label = _cubic_site_label(np.mod(site.frac_coords, 1.0))
            labels[index] = "b2_a" if cubic_label == "corner" else "b2_b"
        return labels

    candidates: list[tuple[float, tuple[float, float, float], dict[int, str]]] = []
    for origin_site in structure:
        shift = -np.asarray(origin_site.frac_coords, dtype=float)
        labels = {
            index: _cubic_site_label(np.mod(site.frac_coords + shift, 1.0))
            for index, site in enumerate(structure)
        }
        if list(labels.values()).count("corner") != 1 or list(labels.values()).count("face") != 3:
            continue
        grouped: dict[str, list[str]] = defaultdict(list)
        for index, label in labels.items():
            grouped[label].append(_site_element(structure[index]))
        score = _order_parameter(
            _distribution(grouped["corner"]), _distribution(grouped["face"])
        )
        shift_key = tuple(float(value) for value in np.round(np.mod(shift, 1.0), 8))
        candidates.append((score, shift_key, labels))
    if not candidates:
        return {index: "other" for index in range(len(structure))}
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _coordination_numbers(structure: Structure, parent: str) -> list[int]:
    if parent not in {"fcc", "bcc"}:
        return [0] * len(structure)
    vpa = structure.volume / len(structure)
    if parent == "fcc":
        d1 = (4.0 * vpa) ** (1.0 / 3.0) / math.sqrt(2.0)
        d2 = d1 * math.sqrt(2.0)
    else:
        d1 = math.sqrt(3.0) * (2.0 * vpa) ** (1.0 / 3.0) / 2.0
        d2 = (2.0 * vpa) ** (1.0 / 3.0)
    try:
        cutoff = (d1 + d2) / 2.0
        return [len(neighbors) for neighbors in structure.get_all_neighbors(cutoff)]
    except Exception:
        return [0] * len(structure)


def _parent_mapping_rms(structure: Structure, parent: str) -> float:
    expected = 12 if parent == "fcc" else 8
    coordination = np.asarray(_coordination_numbers(structure, parent), dtype=float)
    return float(np.sqrt(np.mean(((coordination - expected) / expected) ** 2)))


def _sro(structure: Structure, parent: str) -> tuple[dict[str, float], dict[str, float]]:
    if parent not in {"fcc", "bcc"} or len(structure.composition.elements) < 2:
        return {}, {}
    vpa = structure.volume / len(structure)
    if parent == "fcc":
        d1 = (4.0 * vpa) ** (1.0 / 3.0) / math.sqrt(2.0)
        d2 = d1 * math.sqrt(2.0)
    else:
        d1 = math.sqrt(3.0) * (2.0 * vpa) ** (1.0 / 3.0) / 2.0
        d2 = (2.0 * vpa) ** (1.0 / 3.0)
    shell1 = _warren_cowley(structure, 0.0, (d1 + d2) / 2.0)
    shell2 = _warren_cowley(structure, (d1 + d2) / 2.0, d2 * 1.15)
    return shell1, shell2


def _warren_cowley(structure: Structure, rmin: float, rmax: float) -> dict[str, float]:
    concentrations = {str(el): float(amount) / len(structure) for el, amount in structure.composition.items()}
    pair_counts: Counter[tuple[str, str]] = Counter()
    center_counts: Counter[str] = Counter()
    for index, site in enumerate(structure):
        center = _site_element(site)
        neighbors = structure.get_neighbors(site, rmax)
        for neighbor in neighbors:
            if float(neighbor.nn_distance) <= rmin + 1e-8:
                continue
            other = _site_element(neighbor)
            pair_counts[(center, other)] += 1
            center_counts[center] += 1
    values = {}
    # Zero observed pairs are valid alpha=1 labels; empty central shells are undefined.
    for center, total in center_counts.items():
        for other, concentration in concentrations.items():
            probability = pair_counts[(center, other)] / total
            values[f"{center}->{other}"] = 1.0 - probability / concentration
    return values


def _phase_peak_table(pattern: Any, structure_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    peaks = _peak_table(pattern, f"xrd_{structure_row['dataset_structure_id']}")
    rows: list[dict[str, Any]] = []
    family = str(structure_row["structure_family"])
    for peak in peaks.to_dict(orient="records"):
        reflection_class = _reflection_class(str(peak["hkls_json"]), family)
        rows.append({
            "dataset_structure_id": structure_row["dataset_structure_id"],
            "peak_index": int(peak["peak_index"]), "two_theta": float(peak["two_theta"]),
            "intensity": float(peak["intensity"]), "relative_intensity": float(peak["intensity"]),
            "hkls_json": peak["hkls_json"], "reflection_class": reflection_class,
            "is_superlattice_candidate": reflection_class == "superlattice",
        })
    return rows


def _reflection_class(hkls_json: str, family: str) -> str:
    if family not in {"l12_like", "b2_like"}:
        return "not_applicable"
    hkls = json.loads(hkls_json)
    classes = []
    for item in hkls:
        h, k, l = (int(round(v)) for v in item.get("hkl", [0, 0, 0]))
        if family == "l12_like":
            classes.append("fundamental" if h % 2 == k % 2 == l % 2 else "superlattice")
        else:
            classes.append("fundamental" if (h + k + l) % 2 == 0 else "superlattice")
    return "superlattice" if "superlattice" in classes else "fundamental"


def _balanced_pilot(structures: pd.DataFrame, pilot_size: int) -> pd.DataFrame:
    if len(structures) <= pilot_size:
        result = structures.copy()
        result["pilot_bucket"] = "all"
        return result
    quotas = {
        "ordered_prototypes": int(round(pilot_size * 0.2)),
        "fcc_hea": int(round(pilot_size * 0.3)),
        "bcc_hea": int(round(pilot_size * 0.3)),
        "complex": pilot_size - int(round(pilot_size * 0.8)),
    }
    candidates = {
        "ordered_prototypes": structures[structures["benchmark_subset"] == "ordered_prototypes"],
        "fcc_hea": structures[
            (structures["benchmark_subset"] == "hea_bcc_fcc_sqs")
            & (structures["structure_family"] == "fcc_like") & (structures["nelements"] >= 5)
        ],
        "bcc_hea": structures[
            (structures["benchmark_subset"] == "hea_bcc_fcc_sqs")
            & (structures["structure_family"] == "bcc_like") & (structures["nelements"] >= 5)
        ],
        "complex": structures[structures["benchmark_subset"] == "complex_structures"],
    }
    selected: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    for name, quota in quotas.items():
        frame = candidates[name].copy()
        if name == "ordered_prototypes":
            frame["_special_rank"] = ~frame["structure_family"].isin(["l12_like", "b2_like"])
        else:
            frame["_special_rank"] = False
        frame["_rank"] = frame["structure_sha256"].map(_stable_rank)
        frame = frame.sort_values(["_special_rank", "_rank"])
        frame = frame[~frame["dataset_structure_id"].isin(selected_ids)].head(quota)
        frame = frame.drop(columns=["_special_rank", "_rank"])
        frame["pilot_bucket"] = name
        selected.append(frame)
        selected_ids.update(frame["dataset_structure_id"])
    result = pd.concat(selected, ignore_index=True) if selected else structures.head(0)
    if len(result) < pilot_size:
        remainder = structures[~structures["dataset_structure_id"].isin(selected_ids)].copy()
        remainder["_rank"] = remainder["structure_sha256"].map(_stable_rank)
        remainder = remainder.sort_values("_rank").head(pilot_size - len(result)).drop(columns="_rank")
        remainder["pilot_bucket"] = "fill"
        result = pd.concat([result, remainder])
    return result.reset_index(drop=True)


def _benchmark_subset(base: Mapping[str, Any], family: str, parent: str) -> str:
    if family in {"l12_like", "b2_like"} or (int(base["natoms"]) <= 8 and int(base["nelements"]) >= 2):
        return "ordered_prototypes"
    if parent in {"bcc", "fcc"} and int(base["nelements"]) >= 3:
        return "hea_bcc_fcc_sqs"
    return "complex_structures"


def _split_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return dict(zip(frame["dataset_structure_id"].astype(str), frame["id_split"].astype(str)))


def _hash_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _group_holdout_map(values: Iterable[str]) -> dict[str, str]:
    groups = sorted({str(value) for value in values}, key=_stable_rank)
    result = {group: "train" for group in groups}
    if len(groups) >= 2:
        result[groups[-1]] = "test"
    if len(groups) >= 3:
        result[groups[-2]] = "validation"
    return result


def _stable_rank(value: str) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)


def _augmentation_parameters(structure_hash: str, index: int) -> dict[str, float]:
    seed = int(hashlib.sha256(f"{structure_hash}:{index}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return {
        "noise_fraction": float(rng.uniform(0.002, 0.02)),
        "background_fraction": float(rng.uniform(0.0, 0.05)),
        "zero_shift_points": int(rng.integers(-2, 3)), "seed": seed,
    }


def _augment_grid(clean: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(params["seed"]))
    shifted = np.roll(clean, int(params["zero_shift_points"])).astype(float)
    background = float(params["background_fraction"]) * 100.0 * np.linspace(1.0, 0.3, len(clean))
    noise = rng.normal(0.0, float(params["noise_fraction"]) * 100.0, len(clean))
    values = np.clip(shifted + background + noise, 0.0, None)
    if values.max() > 0:
        values = values / values.max() * 100.0
    return values.astype(np.float32)


def _review_priority(row: Mapping[str, Any]) -> int:
    family = str(row.get("structure_family", ""))
    confidence = _number(row.get("classification_confidence"), 0.0)
    if family in {"l12_like", "b2_like"}:
        return 0
    if confidence < 0.9:
        return 1
    if family in {"complex", "hcp_like"}:
        return 2
    return 3


def _review_priority_reason(row: Mapping[str, Any]) -> str:
    priority = _review_priority(row)
    return {
        0: "ordered_prototype",
        1: "low_classification_confidence",
        2: "complex_or_hcp_structure",
        3: "stratified_pilot",
    }[priority]


def _write_xrd_preview(
    dataset: Path, sample: Mapping[str, Any], output_path: Path, structure: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with h5py.File(dataset / str(sample["grid_shard_path"]), "r") as handle:
        two_theta = handle["two_theta"][:]
        intensity = handle["intensity"][int(sample["grid_row_index"])]
    figure, axis = plt.subplots(figsize=(9.6, 3.2))
    axis.plot(two_theta, intensity, color="#1769aa", linewidth=0.8)
    axis.set_xlim(float(two_theta.min()), float(two_theta.max()))
    axis.set_ylim(0.0, 105.0)
    axis.set_xlabel("2 theta (deg), Cu K-alpha")
    axis.set_ylabel("Relative intensity")
    axis.set_title(
        f"{structure.get('formula', '')} | {structure.get('structure_family', '')} | "
        f"confidence={_number(structure.get('classification_confidence'), 0.0):.3f}"
    )
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _write_review_html(batch_dir: Path, batch: pd.DataFrame, output_root: Path) -> None:
    rows: list[str] = []
    for _, row in batch.iterrows():
        cif = Path(str(row["cif_path"])).relative_to(batch_dir.relative_to(output_root)).as_posix()
        preview_value = str(row["xrd_preview_path"])
        preview = ""
        if preview_value:
            relative = Path(preview_value).relative_to(batch_dir.relative_to(output_root)).as_posix()
            preview = f'<img src="{escape(relative)}" alt="XRD preview" width="720">'
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['dataset_structure_id']))}</td>"
            f"<td>{escape(str(row['formula']))}</td>"
            f"<td>{escape(str(row['structure_family']))}</td>"
            f"<td>{_number(row['classification_confidence'], 0.0):.3f}</td>"
            f"<td>{escape(str(row['priority_reason']))}</td>"
            f"<td><a href=\"{escape(cif)}\">CIF</a></td>"
            f"<td>{preview}</td>"
            "</tr>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Occupancy XRD pilot review</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; color: #202124; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #d0d4d8; padding: 6px; text-align: left; vertical-align: top; }
th { background: #f1f3f4; position: sticky; top: 0; }
img { max-width: 100%; height: auto; }
</style></head><body><table><thead><tr>
<th>Structure ID</th><th>Formula</th><th>Family</th><th>Confidence</th>
<th>Priority</th><th>Structure</th><th>XRD</th>
</tr></thead><tbody>""" + "\n".join(rows) + """</tbody></table></body></html>"""
    (batch_dir / "index.html").write_text(document, encoding="utf-8")


def _load_source_records(source: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    source_row = 0
    for chunk in pd.read_csv(source, chunksize=500):
        for record in chunk.to_dict(orient="records"):
            if source_row in wanted:
                records[source_row] = record
            source_row += 1
        if len(records) == len(wanted):
            break
    return records


def _append_parquet_batch(
    path: Path, rows: list[dict[str, Any]], columns: Sequence[str], writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(pd.DataFrame(rows, columns=columns), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _structure_from_record(record: Mapping[str, Any]) -> Structure:
    payload = record["structure_as_dict"]
    document = ast.literal_eval(str(payload)) if isinstance(payload, str) else payload
    return Structure.from_dict(document)


def _element_distribution(structure: Structure) -> dict[str, float]:
    return {str(el): float(amount) / len(structure) for el, amount in structure.composition.items()}


def _distribution(elements: Iterable[str]) -> dict[str, float]:
    counts = Counter(elements)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())}


def _entropy(distribution: Mapping[str, float]) -> float:
    if len(distribution) <= 1:
        return 0.0
    value = -sum(float(p) * math.log(float(p)) for p in distribution.values() if p > 0)
    return value / math.log(len(distribution))


def _order_parameter(corner: Mapping[str, float], face: Mapping[str, float]) -> float:
    if not corner or not face:
        return 0.0
    corner_element = max(corner, key=corner.get)
    face_element = max(face, key=face.get)
    if corner_element == face_element:
        return 0.0
    return float(corner[corner_element] * face[face_element])


def _binary_order_parameter(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - sum(min(a.get(el, 0.0), b.get(el, 0.0)) for el in set(a) | set(b)))


def _is_cubic(structure: Structure, tolerance: float = 0.03) -> bool:
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    angles = np.asarray(structure.lattice.angles, dtype=float)
    return float(lengths.max() / lengths.min() - 1.0) <= tolerance and np.allclose(angles, 90.0, atol=2.0)


def _cubic_site_label(frac: np.ndarray, tolerance: float = 0.08) -> str:
    zero = [_periodic_close(value, 0.0, tolerance) for value in frac]
    half = [abs(float(value) - 0.5) <= tolerance for value in frac]
    if all(zero):
        return "corner"
    if all(half):
        return "body"
    if sum(half) == 2 and sum(zero) == 1:
        return "face"
    return "other"


def _periodic_close(value: float, target: float, tolerance: float) -> bool:
    delta = abs(float(value) - target)
    return min(delta, abs(delta - 1.0)) <= tolerance


def _site_element(site: Any) -> str:
    if len(site.species) == 1:
        return str(next(iter(site.species.elements)))
    return str(site.species_string)


def _min_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=float)
    distances[distances <= 1e-12] = np.inf
    return float(distances.min())


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_inventory_source(dataset: Path, summary: Mapping[str, Any]) -> Path:
    source_value = str(summary.get("source_csv", ""))
    source_name = str(summary.get("source_csv_name", "")).strip()
    if not source_name:
        source_name = (
            PureWindowsPath(source_value).name if "\\" in source_value else Path(source_value).name
        )
    candidates = [
        Path(str(summary.get("source_csv_local_path", ""))),
        dataset / "source" / source_name,
        dataset / source_name,
        Path(str(summary.get("source_csv", ""))),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Source CSV not found; place {source_name} under {dataset / 'source'}"
    )


def _write_readiness(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Occupancy XRD Benchmark Readiness", "",
        f"- Engineering audit: {report['status']}",
        f"- Benchmark ready: {str(report['benchmark_ready']).lower()}",
        f"- Inventory rows: {report['inventory_rows']}",
        f"- Accepted rows: {report['accepted_rows']}",
        f"- Rejected rows: {report['rejected_rows']}",
        f"- XRD rows: {report['xrd_rows']}",
        f"- Manual review complete: {str(report['manual_review_complete']).lower()}",
        f"- Manual reviewed rows: {report['manual_reviewed_count']}",
        f"- Manual pending rows: {report['manual_review_pending_count']}",
        f"- Manual review pass rate: {report['manual_review_pass_rate']:.3f}", "",
        "## Issues", "",
    ]
    lines.extend(f"- {issue['code']}: {issue['message']}" for issue in report["issues"])
    if not report["issues"]:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
