"""Pilot physical audit for nickel gamma/gamma-prime structure sources."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from .contract import canonical_structure_hash
from .datasets import (
    _cfg_block_has_forces,
    _cfg_block_has_section,
    _cfg_block_to_structure,
    _cfg_blocks,
)


MIN_NEIGHBOR_DISTANCE = 1.5
MIN_VOLUME_PER_ATOM = 5.0
MAX_VOLUME_PER_ATOM = 30.0


def run_nickel_pilot_audit(
    *,
    root: str | Path,
    output_dir: str | Path,
    darus_cfg_path: str | Path | None = None,
    darus_sample_size: int = 32,
    include_relaxed: bool = True,
) -> dict[str, Any]:
    """Audit the current nickel parent pairs and a small external DFT sample."""

    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    checks: list[dict[str, Any]] = []

    initial_summary = _audit_initial_pairs(root_path, checks)
    relaxed_summary = (
        _audit_relaxed_pairs(root_path, checks)
        if include_relaxed
        else {"potential_count": 0, "pair_count": 0, "misfit_min": None, "misfit_max": None}
    )
    darus_summary = _audit_darus_sample(
        Path(darus_cfg_path).resolve() if darus_cfg_path else None,
        sample_size=darus_sample_size,
        checks=checks,
    )
    _record_unimplemented_constraints(checks)

    counts = Counter(str(row["status"]) for row in checks)
    check_summary = {
        key: int(counts.get(key, 0))
        for key in ("pass", "warn", "fail", "not_evaluated")
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root_path),
        "status": "fail" if check_summary["fail"] else ("review" if check_summary["warn"] else "pass"),
        "initial_pairs": initial_summary,
        "relaxed_pairs": relaxed_summary,
        "darus_sample": darus_summary,
        "checks": check_summary,
        "limitations": [
            "Configured compositions and initial lattice parameters are benchmark design values, not experimental phase-equilibrium measurements.",
            "Current gamma-prime seeds are ideal ordered L12 structures and do not span antisite disorder.",
            "The pilot does not establish thermodynamic, phonon, or mechanical stability.",
            "No gamma/gamma-prime component-level overlapping XRD samples currently enter this audit.",
        ],
    }

    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output_path / "checks.csv", index=False)
    (output_path / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_path / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _audit_initial_pairs(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_dir = root / "data" / "shared" / "manifests"
    initial_path = manifest_dir / "initial_structures.parquet"
    pair_path = manifest_dir / "gamma_prime_pairs.parquet"
    if not initial_path.exists() or not pair_path.exists():
        raise FileNotFoundError("Initial-structure and gamma-prime pair manifests are required")

    initial_rows = pd.read_parquet(initial_path)
    pair_rows = pd.read_parquet(pair_path)
    by_id = {
        str(row["initial_structure_id"]): row
        for row in initial_rows.to_dict(orient="records")
    }
    gamma_hashes: list[str] = []
    gamma_prime_hashes: list[str] = []
    misfits: list[float] = []

    for pair in pair_rows.to_dict(orient="records"):
        pair_id = str(pair["pair_id"])
        gamma_row = by_id.get(str(pair["gamma_initial_structure_id"]))
        gamma_prime_row = by_id.get(str(pair["gamma_prime_initial_structure_id"]))
        _check(checks, "initial_pair", pair_id, "pair_complete", gamma_row is not None and gamma_prime_row is not None)
        if gamma_row is None or gamma_prime_row is None:
            continue
        gamma = _load_structure(root, gamma_row, "artifact_path")
        gamma_prime = _load_structure(root, gamma_prime_row, "artifact_path")
        gamma_hashes.append(str(gamma_row["structure_sha256"]))
        gamma_prime_hashes.append(str(gamma_prime_row["structure_sha256"]))

        _check(checks, "initial_pair", pair_id, "gamma_phase", str(gamma_row["phase"]).lower() == "fcc", value=gamma_row["phase"], threshold="fcc")
        _check(checks, "initial_pair", pair_id, "gamma_prime_phase", str(gamma_prime_row["phase"]).lower() == "l12", value=gamma_prime_row["phase"], threshold="l12")
        same_system = str(gamma_row["nominal_alloy_system"]) == str(gamma_prime_row["nominal_alloy_system"])
        _check(checks, "initial_pair", pair_id, "nominal_alloy_system_match", same_system)
        _check_structure_geometry(checks, "initial_gamma", str(gamma_row["initial_structure_id"]), gamma)
        _check_structure_geometry(checks, "initial_gamma_prime", str(gamma_prime_row["initial_structure_id"]), gamma_prime)

        supercell = _supercell_tuple(str(gamma_prime_row["supercell"]))
        l12_ok, l12_note = _is_ideal_l12_assignment(gamma_prime, supercell)
        _check(
            checks,
            "initial_gamma_prime",
            str(gamma_prime_row["initial_structure_id"]),
            "ideal_l12_sublattice_assignment",
            l12_ok,
            notes=l12_note,
        )
        counts = gamma_prime.composition.get_el_amt_dict()
        ni_count = float(counts.get("Ni", 0.0))
        minority_count = float(sum(value for element, value in counts.items() if element != "Ni"))
        stoich_ok = minority_count > 0 and math.isclose(ni_count / minority_count, 3.0, abs_tol=1e-8)
        _check(
            checks,
            "initial_gamma_prime",
            str(gamma_prime_row["initial_structure_id"]),
            "ni3x_stoichiometry",
            stoich_ok,
            value=ni_count / minority_count if minority_count else None,
            threshold="3.0",
        )
        gamma_a = _equivalent_fcc_lattice_parameter(gamma)
        gamma_prime_a = _equivalent_fcc_lattice_parameter(gamma_prime)
        misfit = (gamma_prime_a - gamma_a) / gamma_a
        misfits.append(misfit)
        _row(
            checks,
            scope="initial_pair",
            record_id=pair_id,
            check="configured_lattice_misfit",
            status="pass",
            value=misfit,
            threshold="reported_not_filtered",
            notes="Initial configured value; not an experimental acceptance check.",
        )

    duplicate_gamma = len(gamma_hashes) - len(set(gamma_hashes))
    duplicate_gamma_prime = len(gamma_prime_hashes) - len(set(gamma_prime_hashes))
    if duplicate_gamma:
        _row(checks, scope="initial_corpus", record_id="gamma", check="duplicate_structure_hashes", status="warn", value=duplicate_gamma, threshold=0)
    if duplicate_gamma_prime:
        _row(checks, scope="initial_corpus", record_id="gamma_prime", check="duplicate_structure_hashes", status="warn", value=duplicate_gamma_prime, threshold=0)

    return {
        "pair_count": int(len(pair_rows)),
        "unique_gamma_structures": len(set(gamma_hashes)),
        "unique_gamma_prime_structures": len(set(gamma_prime_hashes)),
        "duplicate_gamma_count": duplicate_gamma,
        "duplicate_gamma_prime_count": duplicate_gamma_prime,
        "configured_misfit_min": min(misfits) if misfits else None,
        "configured_misfit_max": max(misfits) if misfits else None,
    }


def _audit_relaxed_pairs(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    potential_dirs = sorted((root / "data" / "potentials").glob("*/generated/manifests"))
    misfits: list[float] = []
    pair_count = 0
    potential_count = 0
    by_potential: dict[str, int] = {}
    for manifest_dir in potential_dirs:
        pair_path = manifest_dir / "gamma_prime_pairs.parquet"
        structure_path = manifest_dir / "relaxed_structures.parquet"
        if not pair_path.exists() or not structure_path.exists():
            continue
        potential_id = manifest_dir.parts[-3]
        pairs = pd.read_parquet(pair_path)
        structures = pd.read_parquet(structure_path)
        by_id = {
            str(row["relaxed_structure_id"]): row
            for row in structures.to_dict(orient="records")
        }
        valid_here = 0
        for pair in pairs.to_dict(orient="records"):
            gamma_id = str(pair.get("gamma_relaxed_structure_id", ""))
            gamma_prime_id = str(pair.get("gamma_prime_relaxed_structure_id", ""))
            if not gamma_id or not gamma_prime_id:
                continue
            pair_id = f"{potential_id}:{pair['pair_id']}"
            gamma_row = by_id.get(gamma_id)
            gamma_prime_row = by_id.get(gamma_prime_id)
            _check(checks, "relaxed_pair", pair_id, "pair_complete", gamma_row is not None and gamma_prime_row is not None)
            if gamma_row is None or gamma_prime_row is None:
                continue
            gamma = _load_structure(root, gamma_row, "artifact_path_json")
            gamma_prime = _load_structure(root, gamma_prime_row, "artifact_path_json")
            _check(checks, "relaxed_pair", pair_id, "gamma_quality_tier", gamma_row.get("quality_tier") == "pass", value=gamma_row.get("quality_tier"), threshold="pass")
            _check(checks, "relaxed_pair", pair_id, "gamma_prime_quality_tier", gamma_prime_row.get("quality_tier") == "pass", value=gamma_prime_row.get("quality_tier"), threshold="pass")
            _check_structure_geometry(checks, "relaxed_gamma", gamma_id, gamma)
            _check_structure_geometry(checks, "relaxed_gamma_prime", gamma_prime_id, gamma_prime)
            gamma_a = _equivalent_fcc_lattice_parameter(gamma)
            gamma_prime_a = _equivalent_fcc_lattice_parameter(gamma_prime)
            misfit = (gamma_prime_a - gamma_a) / gamma_a
            misfits.append(misfit)
            status = "pass" if abs(misfit) <= 0.02 else "warn"
            _row(checks, scope="relaxed_pair", record_id=pair_id, check="equivalent_lattice_misfit", status=status, value=misfit, threshold="abs <= 0.02 pilot bound")
            valid_here += 1
        if valid_here:
            potential_count += 1
            pair_count += valid_here
            by_potential[potential_id] = valid_here
    return {
        "potential_count": potential_count,
        "pair_count": pair_count,
        "pairs_by_potential": by_potential,
        "misfit_min": min(misfits) if misfits else None,
        "misfit_max": max(misfits) if misfits else None,
    }


def _audit_darus_sample(
    path: Path | None,
    *,
    sample_size: int,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    if path is None:
        return {"available": False, "source_configuration_count": 0, "sampled_count": 0}
    if not path.exists():
        _row(checks, scope="darus", record_id=str(path), check="source_file_exists", status="fail", value=False, threshold=True)
        return {"available": False, "source_configuration_count": 0, "sampled_count": 0}

    blocks = _cfg_blocks(path.read_text(encoding="utf-8", errors="replace").splitlines())
    selected = _stratified_cfg_indices(blocks, max(0, sample_size))
    hashes: list[str] = []
    atom_counts: Counter[int] = Counter()
    for index in selected:
        block = blocks[index]
        structure = _cfg_block_to_structure(block, {"0": "Al", "1": "Ni"})
        record_id = f"cfg_{index:06d}"
        atom_counts[len(structure)] += 1
        hashes.append(canonical_structure_hash(structure.as_dict()))
        _check_structure_geometry(checks, "darus_structure", record_id, structure, require_cubic=False)
        _check(checks, "darus_structure", record_id, "al_ni_only", {str(el) for el in structure.composition.elements} <= {"Al", "Ni"})
        labels_ok = (
            _cfg_block_has_section(block, "Energy")
            and _cfg_block_has_forces(block)
            and _cfg_block_has_section(block, "PlusStress")
        )
        _check(checks, "darus_structure", record_id, "energy_force_stress_labels", labels_ok)
    duplicate_count = len(hashes) - len(set(hashes))
    if duplicate_count:
        _row(checks, scope="darus_sample", record_id="sample", check="duplicate_structure_hashes", status="warn", value=duplicate_count, threshold=0)
    return {
        "available": True,
        "source_configuration_count": len(blocks),
        "sampled_count": len(selected),
        "sampled_atom_counts": {str(key): value for key, value in sorted(atom_counts.items())},
        "unique_sampled_structures": len(set(hashes)),
        "duplicate_sampled_count": duplicate_count,
    }


def _check_structure_geometry(
    checks: list[dict[str, Any]],
    scope: str,
    record_id: str,
    structure: Structure,
    *,
    require_cubic: bool = True,
) -> None:
    minimum = _minimum_distance(structure)
    _check(checks, scope, record_id, "minimum_neighbor_distance", minimum >= MIN_NEIGHBOR_DISTANCE, value=minimum, threshold=f">= {MIN_NEIGHBOR_DISTANCE}")
    vpa = structure.volume / len(structure)
    _check(checks, scope, record_id, "volume_per_atom", MIN_VOLUME_PER_ATOM <= vpa <= MAX_VOLUME_PER_ATOM, value=vpa, threshold=f"{MIN_VOLUME_PER_ATOM}..{MAX_VOLUME_PER_ATOM}")
    if require_cubic:
        lengths = np.asarray(structure.lattice.abc, dtype=float)
        angles = np.asarray(structure.lattice.angles, dtype=float)
        cubic = float(lengths.max() / lengths.min() - 1.0) <= 0.03 and np.allclose(angles, 90.0, atol=2.0)
        _check(checks, scope, record_id, "cubic_cell", bool(cubic), value={"abc": lengths.tolist(), "angles": angles.tolist()}, threshold="length anisotropy <= 3%; angles 90 +/- 2 deg")


def _is_ideal_l12_assignment(structure: Structure, supercell: tuple[int, int, int]) -> tuple[bool, str]:
    allowed_corner = {"Al", "Ti", "Ta", "Nb"}
    corner_count = 0
    face_count = 0
    for site in structure:
        local = np.mod(np.asarray(site.frac_coords) * np.asarray(supercell), 1.0)
        zeros = [min(abs(float(value)), abs(float(value) - 1.0)) <= 0.08 for value in local]
        halves = [abs(float(value) - 0.5) <= 0.08 for value in local]
        symbol = str(site.specie)
        if all(zeros):
            corner_count += 1
            if symbol not in allowed_corner:
                return False, f"corner site contains {symbol}"
        elif sum(halves) == 2 and sum(zeros) == 1:
            face_count += 1
            if symbol != "Ni":
                return False, f"face site contains {symbol}"
        else:
            return False, f"site {site.frac_coords.tolist()} does not map to ideal L12"
    expected_cells = int(np.prod(supercell))
    valid = corner_count == expected_cells and face_count == 3 * expected_cells
    return valid, f"corner={corner_count}, face={face_count}"


def _stratified_cfg_indices(blocks: Sequence[Sequence[str]], sample_size: int) -> list[int]:
    if sample_size <= 0 or not blocks:
        return []
    groups: dict[int, list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        groups[_cfg_size(block)].append(index)
    selected: list[int] = []
    ordered_groups = sorted(groups.items())
    base = sample_size // len(ordered_groups)
    remainder = sample_size % len(ordered_groups)
    for group_index, (_, indices) in enumerate(ordered_groups):
        quota = min(len(indices), base + (1 if group_index < remainder else 0))
        if quota:
            positions = np.linspace(0, len(indices) - 1, num=quota, dtype=int)
            selected.extend(indices[int(position)] for position in positions)
    if len(selected) < min(sample_size, len(blocks)):
        selected_set = set(selected)
        selected.extend(index for index in range(len(blocks)) if index not in selected_set)
    return sorted(selected[: min(sample_size, len(blocks))])


def _cfg_size(block: Sequence[str]) -> int:
    for index, line in enumerate(block):
        if line.strip() == "Size" and index + 1 < len(block):
            return int(block[index + 1].strip())
    return 0


def _record_unimplemented_constraints(checks: list[dict[str, Any]]) -> None:
    for name in (
        "experimental_phase_equilibrium",
        "site_preference_energy",
        "formation_energy_or_hull_stability",
        "component_level_overlap_ground_truth",
    ):
        _row(checks, scope="pilot_gap", record_id="global", check=name, status="not_evaluated", value="", threshold="required before full release")


def _load_structure(root: Path, row: Mapping[str, Any], field: str) -> Structure:
    return Structure.from_dict(json.loads((root / str(row[field])).read_text(encoding="utf-8")))


def _minimum_distance(structure: Structure) -> float:
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    matrix[matrix <= 1e-8] = np.inf
    return float(matrix.min())


def _equivalent_fcc_lattice_parameter(structure: Structure) -> float:
    return float((4.0 * structure.volume / len(structure)) ** (1.0 / 3.0))


def _supercell_tuple(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item) for item in value.lower().split("x"))
    if len(parts) != 3:
        raise ValueError(f"Invalid supercell label: {value}")
    return parts


def _check(
    checks: list[dict[str, Any]],
    scope: str,
    record_id: str,
    check: str,
    condition: bool,
    *,
    value: Any = None,
    threshold: Any = True,
    notes: str = "",
) -> None:
    _row(checks, scope=scope, record_id=record_id, check=check, status="pass" if condition else "fail", value=value, threshold=threshold, notes=notes)


def _row(
    checks: list[dict[str, Any]],
    *,
    scope: str,
    record_id: str,
    check: str,
    status: str,
    value: Any,
    threshold: Any,
    notes: str = "",
) -> None:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True)
    checks.append({"scope": scope, "record_id": record_id, "check": check, "status": status, "value": value, "threshold": threshold, "notes": notes})


def _render_markdown(report: Mapping[str, Any]) -> str:
    initial = report["initial_pairs"]
    relaxed = report["relaxed_pairs"]
    darus = report["darus_sample"]
    checks = report["checks"]
    lines = [
        "# Nickel Gamma/Gamma-Prime Pilot Audit",
        "",
        f"Overall status: `{report['status']}`",
        "",
        "## Coverage",
        "",
        f"- Initial gamma/gamma-prime pairs: {initial['pair_count']}",
        f"- Unique gamma structures: {initial['unique_gamma_structures']}",
        f"- Unique gamma-prime structures: {initial['unique_gamma_prime_structures']}",
        f"- Relaxed pairs: {relaxed['pair_count']} across {relaxed['potential_count']} potentials",
        f"- DaRUS configurations sampled: {darus['sampled_count']} of {darus['source_configuration_count']}",
        "",
        "## Checks",
        "",
        f"- Pass: {checks['pass']}",
        f"- Warn: {checks['warn']}",
        f"- Fail: {checks['fail']}",
        f"- Not evaluated: {checks['not_evaluated']}",
        "",
        "## Known Gaps",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a small nickel gamma/gamma-prime structure pilot")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="audits/nickel_gamma_pilot")
    parser.add_argument("--darus-cfg")
    parser.add_argument("--darus-sample-size", type=int, default=32)
    parser.add_argument("--skip-relaxed", action="store_true")
    args = parser.parse_args(argv)
    report = run_nickel_pilot_audit(
        root=args.root,
        output_dir=args.output_dir,
        darus_cfg_path=args.darus_cfg,
        darus_sample_size=args.darus_sample_size,
        include_relaxed=not args.skip_relaxed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
