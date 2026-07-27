"""Source ingestion and parent-reflection indexing for the Ni-Al benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from .contract import canonical_structure_hash
from .datasets import (
    _cfg_block_has_forces,
    _cfg_block_has_section,
    _cfg_block_to_structure,
    _cfg_blocks,
)
from .gamma_overlap_physics import canonical_cubic_hkl, parent_hkl_from_supercell


DARUS_DATASET_ID = "darus_4480_ni3al_yield_anomaly_2024"
DARUS_DOI = "10.18419/darus-4480"
DARUS_LICENSE = "CC BY 4.0"


def classify_darus_structure(structure: Structure) -> dict[str, Any]:
    """Classify a DaRUS configuration for the v1 parent-Bragg benchmark."""

    natoms = len(structure)
    amounts = structure.composition.get_el_amt_dict()
    composition = {element: int(round(amount)) for element, amount in sorted(amounts.items())}
    elements_ok = set(composition) <= {"Al", "Ni"} and bool(composition)
    finite = bool(
        np.isfinite(structure.lattice.matrix).all()
        and np.isfinite(np.asarray(structure.cart_coords)).all()
        and structure.volume > 0
    )
    volume_per_atom = float(structure.volume / natoms) if natoms else float("nan")
    minimum_distance = _minimum_distance(structure) if natoms > 1 and finite else float("nan")
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    angles = np.asarray(structure.lattice.angles, dtype=float)
    near_cubic = bool(
        finite
        and float(lengths.max() / lengths.min() - 1.0) <= 0.03
        and np.allclose(angles, 90.0, atol=2.0)
    )
    geometry_ok = bool(
        finite
        and 5.0 <= volume_per_atom <= 30.0
        and minimum_distance >= 1.5
    )
    core_compositions = {
        (27, 81),
        (26, 81),
        (27, 80),
    }
    composition_key = (composition.get("Al", 0), composition.get("Ni", 0))

    if natoms in {96, 144} and elements_ok and geometry_ok:
        eligibility = "ood_quarantine"
        reason = "complex_or_noncore_supercell"
    elif natoms in {107, 108} and composition_key in core_compositions and near_cubic and geometry_ok:
        eligibility = "accepted_core"
        reason = ""
    else:
        eligibility = "rejected"
        reasons = []
        if natoms not in {107, 108}:
            reasons.append("unsupported_atom_count")
        if not elements_ok:
            reasons.append("non_al_ni_chemistry")
        if composition_key not in core_compositions:
            reasons.append("unsupported_core_composition")
        if not near_cubic:
            reasons.append("not_3x3x3_cubic_parent")
        if not geometry_ok:
            reasons.append("invalid_geometry")
        reason = ";".join(reasons)
    return {
        "eligibility": eligibility,
        "exclusion_reason": reason,
        "natoms": natoms,
        "composition": composition,
        "minimum_neighbor_distance_angstrom": minimum_distance,
        "volume_per_atom_angstrom3": volume_per_atom,
        "near_cubic_3x3x3": near_cubic,
    }


def extract_parent_reflections(
    structure: Structure,
    *,
    phase: str,
    supercell: Sequence[int] = (3, 3, 3),
    wavelength_angstrom: float = 1.5406,
    two_theta_range: tuple[float, float] = (20.0, 100.0),
) -> dict[str, list[dict[str, Any]]]:
    """Separate integer parent-Bragg peaks from artificial supercell satellites."""

    if phase not in {"gamma", "gamma_prime"}:
        raise ValueError("phase must be gamma or gamma_prime")
    cell_count = int(np.prod(np.asarray(supercell, dtype=int)))
    if cell_count <= 0:
        raise ValueError("supercell repeats must be positive")
    pattern = XRDCalculator(wavelength=float(wavelength_angstrom)).get_pattern(
        structure,
        scaled=False,
        two_theta_range=two_theta_range,
    )
    parent_rows: list[dict[str, Any]] = []
    satellite_rows: list[dict[str, Any]] = []
    for peak_index, (center, intensity, hkl_entries) in enumerate(
        zip(pattern.x, pattern.y, pattern.hkls)
    ):
        mapped: dict[tuple[int, int, int], int] = {}
        serialized_entries = []
        for entry in hkl_entries:
            super_hkl = tuple(int(value) for value in entry["hkl"])
            multiplicity = int(entry.get("multiplicity", 1))
            serialized_entries.append({"hkl": list(super_hkl), "multiplicity": multiplicity})
            parent_hkl = parent_hkl_from_supercell(super_hkl, supercell)
            if parent_hkl is None or not _reflection_allowed(phase, parent_hkl):
                continue
            mapped[parent_hkl] = mapped.get(parent_hkl, 0) + multiplicity
        corrected_intensity = float(intensity) / float(cell_count**2)
        if mapped:
            share = corrected_intensity / len(mapped)
            for parent_hkl, multiplicity in sorted(mapped.items()):
                parent_rows.append(
                    {
                        "reflection_index": len(parent_rows),
                        "phase": phase,
                        "h": parent_hkl[0],
                        "k": parent_hkl[1],
                        "l": parent_hkl[2],
                        "center_deg": float(center),
                        "raw_integrated_intensity": share,
                        "multiplicity": multiplicity,
                        "reflection_kind": "parent_bragg",
                        "scored": True,
                        "source_peak_index": peak_index,
                        "supercell_hkls_json": json.dumps(serialized_entries, sort_keys=True),
                    }
                )
        else:
            satellite_rows.append(
                {
                    "reflection_index": len(satellite_rows),
                    "phase": phase,
                    "h": None,
                    "k": None,
                    "l": None,
                    "center_deg": float(center),
                    "raw_integrated_intensity": corrected_intensity,
                    "multiplicity": sum(int(entry.get("multiplicity", 1)) for entry in hkl_entries),
                    "reflection_kind": "satellite_unscored",
                    "scored": False,
                    "source_peak_index": peak_index,
                    "supercell_hkls_json": json.dumps(serialized_entries, sort_keys=True),
                }
            )
    return {
        "parent_reflections": _merge_duplicate_parent_rows(parent_rows),
        "satellite_reflections": satellite_rows,
    }


def ingest_darus_cfg(
    *,
    cfg_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Ingest all DaRUS CFG records with explicit eligibility and provenance."""

    source = Path(cfg_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output = Path(output_dir).resolve()
    structures_dir = output / "structures" / "gamma_prime"
    structures_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    blocks = _cfg_blocks(source.read_text(encoding="utf-8", errors="replace").splitlines())
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    previous_signature: tuple[Any, ...] | None = None
    run_index = -1
    for index, block in enumerate(blocks):
        structure = _cfg_block_to_structure(block, {"0": "Al", "1": "Ni"})
        classification = classify_darus_structure(structure)
        signature = _source_run_signature(structure)
        if signature != previous_signature:
            run_index += 1
            previous_signature = signature
        structure_doc = structure.as_dict()
        structure_hash = canonical_structure_hash(structure_doc)
        eligibility = str(classification["eligibility"])
        exclusion_reason = str(classification["exclusion_reason"])
        if structure_hash in seen_hashes and eligibility == "accepted_core":
            eligibility = "rejected"
            exclusion_reason = "duplicate_structure_hash"
        seen_hashes.add(structure_hash)
        labels_ok = bool(
            _cfg_block_has_section(block, "Energy")
            and _cfg_block_has_forces(block)
            and _cfg_block_has_section(block, "PlusStress")
        )
        if not labels_ok and eligibility == "accepted_core":
            eligibility = "rejected"
            exclusion_reason = "missing_energy_force_stress_labels"
        parent_id = f"gamma_prime_darus4480_{index:06d}"
        artifact_path = ""
        artifact_sha256 = ""
        if eligibility == "accepted_core":
            artifact = structures_dir / f"{parent_id}.json"
            payload = json.dumps(structure_doc, indent=2, sort_keys=True).encode("utf-8")
            artifact.write_bytes(payload)
            artifact_path = artifact.relative_to(output).as_posix()
            artifact_sha256 = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "parent_structure_id": parent_id,
                "phase_role": "gamma_prime",
                "source_dataset_id": DARUS_DATASET_ID,
                "source_record_id": f"cfg_{index:06d}",
                "source_index": index,
                "source_run_id": f"darus4480_run_{run_index:04d}",
                "source_doi": DARUS_DOI,
                "license": DARUS_LICENSE,
                "source_cfg_sha256": source_sha256,
                "structure_sha256": structure_hash,
                "artifact_file_sha256": artifact_sha256,
                "artifact_path": artifact_path,
                "has_energy": _cfg_block_has_section(block, "Energy"),
                "has_forces": _cfg_block_has_forces(block),
                "has_stress": _cfg_block_has_section(block, "PlusStress"),
                **classification,
                "eligibility": eligibility,
                "exclusion_reason": exclusion_reason,
            }
        )
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "darus_parent_structures.parquet", index=False)
    frame[frame["eligibility"] != "accepted_core"].to_parquet(
        output / "rejected_structures.parquet", index=False
    )
    counts = frame["eligibility"].value_counts().to_dict()
    report = {
        "source_configuration_count": len(frame),
        "accepted_core_count": int(counts.get("accepted_core", 0)),
        "ood_quarantine_count": int(counts.get("ood_quarantine", 0)),
        "rejected_count": int(counts.get("rejected", 0)),
        "source_cfg_sha256": source_sha256,
    }
    (output / "ingest_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _reflection_allowed(phase: str, hkl: Sequence[int]) -> bool:
    if phase == "gamma_prime":
        return True
    parity = {abs(int(value)) % 2 for value in hkl}
    return len(parity) == 1


def _merge_duplicate_parent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[float, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            round(float(row["center_deg"]), 8),
            int(row["h"]),
            int(row["k"]),
            int(row["l"]),
        )
        if key not in merged:
            merged[key] = dict(row)
        else:
            merged[key]["raw_integrated_intensity"] += float(row["raw_integrated_intensity"])
            merged[key]["multiplicity"] += int(row["multiplicity"])
    result = sorted(merged.values(), key=lambda row: (row["center_deg"], row["h"], row["k"], row["l"]))
    for index, row in enumerate(result):
        row["reflection_index"] = index
    return result


def _source_run_signature(structure: Structure) -> tuple[Any, ...]:
    composition = tuple(
        sorted((element, int(round(amount))) for element, amount in structure.composition.get_el_amt_dict().items())
    )
    lattice = tuple(round(float(value), 4) for value in np.asarray(structure.lattice.matrix).ravel())
    return len(structure), composition, lattice


def _minimum_distance(structure: Structure) -> float:
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    matrix[matrix <= 1e-8] = np.inf
    return float(matrix.min())

