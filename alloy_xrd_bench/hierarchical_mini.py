"""Build a small hierarchical XRD release from accepted relaxed structures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from .contract import DEFAULT_XRD_CONFIG


@dataclass(frozen=True)
class SourceSpec:
    potential_id: str
    alloy_system: str
    phase: str


def build_hierarchical_mini(
    *,
    root: str | Path,
    output: str | Path,
    sources: Sequence[SourceSpec],
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = (root_path / output).resolve() if not Path(output).is_absolute() else Path(output).resolve()
    try:
        output_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Output must stay inside the benchmark root") from exc

    initial_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not initial_path.exists():
        raise FileNotFoundError(f"Missing initial structures manifest: {initial_path}")
    initials = pd.read_parquet(initial_path)
    initial_by_id = {
        str(row["initial_structure_id"]): row
        for row in initials.to_dict(orient="records")
    }

    records = _load_source_records(root_path, sources, initial_by_id)
    if not records:
        raise ValueError("No accepted relaxed structures matched the requested sources")

    output_path.mkdir(parents=True, exist_ok=True)
    patterns_dir = output_path / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    xrd_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    baseline_inputs: list[dict[str, Any]] = []

    for record in records:
        structure = Structure.from_dict(
            json.loads((root_path / record["artifact_path_json"]).read_text(encoding="utf-8"))
        )
        composition = _composition_fractions(structure)
        parent_lattice = _parent_lattice_parameter(structure, record["phase"])
        site_rows.extend(_site_occupancy_rows(record, structure))

        pattern = XRDCalculator(wavelength="CuKa").get_pattern(
            structure,
            two_theta_range=(
                float(DEFAULT_XRD_CONFIG["two_theta_min"]),
                float(DEFAULT_XRD_CONFIG["two_theta_max"]),
            ),
        )
        for observation_kind, fwhm in (("clean", 0.02), ("broadened", 0.1)):
            sample_id = f"xrd_{record['relaxed_structure_id']}_{observation_kind}"
            grid, intensity = _pattern_grid(pattern, fwhm=fwhm)
            pattern_file = patterns_dir / f"{sample_id}.npz"
            np.savez_compressed(pattern_file, two_theta=grid, intensity=intensity)
            pattern_relative = pattern_file.relative_to(root_path).as_posix()
            xrd_rows.append(
                {
                    "xrd_sample_id": sample_id,
                    "relaxed_structure_id": record["relaxed_structure_id"],
                    "initial_structure_id": record["initial_structure_id"],
                    "potential_id": record["potential_id"],
                    "alloy_system": record["alloy_system"],
                    "phase": record["phase"],
                    "composition_json": json.dumps(composition, sort_keys=True),
                    "observation_kind": observation_kind,
                    "radiation": "Cu Kalpha",
                    "two_theta_min": 20.0,
                    "two_theta_max": 100.0,
                    "step": 0.02,
                    "point_count": len(grid),
                    "profile": f"gaussian_fwhm_{fwhm:g}_deg",
                    "parent_lattice_parameter_angstrom": parent_lattice,
                    "pattern_path": pattern_relative,
                    "split": "unassigned",
                }
            )
            component_rows.extend(
                _component_peak_rows(
                    sample_id=sample_id,
                    phase=record["phase"],
                    pattern=pattern,
                    fwhm=fwhm,
                )
            )
            task_rows.extend(
                _task_rows(
                    sample_id=sample_id,
                    phase=record["phase"],
                    composition=composition,
                    parent_lattice=parent_lattice,
                )
            )
            edge_rows.append(
                {
                    "source_id": record["relaxed_structure_id"],
                    "source_type": "relaxed_structure",
                    "target_id": sample_id,
                    "target_type": "xrd_sample",
                    "relation_type": "simulated_as",
                }
            )
            if observation_kind == "broadened":
                baseline_inputs.append(
                    {
                        "xrd_sample_id": sample_id,
                        "initial_structure_id": record["initial_structure_id"],
                        "phase": record["phase"],
                        "composition": composition,
                        "parent_lattice": parent_lattice,
                        "intensity": intensity,
                    }
                )

        edge_rows.append(
            {
                "source_id": record["potential_id"],
                "source_type": "potential",
                "target_id": record["relaxed_structure_id"],
                "target_type": "relaxed_structure",
                "relation_type": "relaxed_with",
            }
        )
        edge_rows.extend(
            {
                "source_id": record["relaxed_structure_id"],
                "source_type": "relaxed_structure",
                "target_id": row["site_occupancy_id"],
                "target_type": "site_occupancy",
                "relation_type": "has_site",
            }
            for row in site_rows
            if row["relaxed_structure_id"] == record["relaxed_structure_id"]
        )

    _write_csv(output_path / "xrd_samples.csv", xrd_rows)
    _write_csv(output_path / "site_occupancy.csv", site_rows)
    _write_csv(output_path / "component_peaks.csv", component_rows)
    _write_csv(output_path / "tasks.csv", task_rows)
    _write_csv(output_path / "provenance_edges.csv", _unique_edges(edge_rows))

    source_counts = {
        source.potential_id: sum(record["potential_id"] == source.potential_id for record in records)
        for source in sources
    }
    target_diagnostics = _target_diagnostics(xrd_rows)
    limitations = [
        "Small engineering mini; baseline metrics are not scientific results.",
        "Site rows preserve occupancy truth but exact disordered site recovery is not a leaderboard task.",
        "Chemical-order, antisite, gamma/gamma-prime, and overlap labels are outside this release.",
        "Splits are unassigned and must be frozen before scientific training or evaluation.",
    ]
    if not target_diagnostics["lattice_parameter_regression_nontrivial"]:
        limitations.append(
            "Lattice parameter is constant within each alloy/phase group; its regression task is interface-only."
        )
    summary = {
        "release_id": "v0.3_hierarchical_mini",
        "release_status": "engineering_smoke_only",
        "source_kind": "accepted_mlip_relaxed_structures",
        "source_counts": source_counts,
        "structure_count": len(records),
        "xrd_sample_count": len(xrd_rows),
        "site_occupancy_row_count": len(site_rows),
        "component_peak_row_count": len(component_rows),
        "task_row_count": len(task_rows),
        "points_per_pattern": 4001,
        "observation_kinds": ["clean", "broadened"],
        "tasks": [
            "phase_classification",
            "composition_regression",
            "lattice_parameter_regression",
        ],
        "chemical_ordering_claim": False,
        "gamma_decomposition_claim": False,
        "target_diagnostics": target_diagnostics,
        "baseline": _loo_nearest_neighbor_metrics(baseline_inputs),
        "limitations": limitations,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _load_source_records(
    root: Path,
    sources: Sequence[SourceSpec],
    initial_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        manifests = root / "data" / "potentials" / source.potential_id / "generated" / "manifests"
        relaxation_path = manifests / "relaxations.parquet"
        relaxed_path = manifests / "relaxed_structures.parquet"
        if not relaxation_path.exists() or not relaxed_path.exists():
            raise FileNotFoundError(f"Missing generated manifests for {source.potential_id}: {manifests}")
        relaxations = {
            str(row["relaxation_run_id"]): row
            for row in pd.read_parquet(relaxation_path).to_dict(orient="records")
        }
        for relaxed in pd.read_parquet(relaxed_path).to_dict(orient="records"):
            if str(relaxed.get("quality_tier", "")) != "pass":
                continue
            relaxation = relaxations.get(str(relaxed["relaxation_run_id"]))
            if not relaxation or not bool(relaxation.get("converged", False)):
                continue
            initial_id = str(relaxation["initial_structure_id"])
            initial = initial_by_id.get(initial_id)
            if not initial:
                continue
            if str(initial.get("alloy_system")) != source.alloy_system:
                continue
            if str(initial.get("phase")).lower() != source.phase.lower():
                continue
            records.append(
                {
                    "relaxed_structure_id": str(relaxed["relaxed_structure_id"]),
                    "initial_structure_id": initial_id,
                    "potential_id": source.potential_id,
                    "alloy_system": source.alloy_system,
                    "phase": source.phase.lower(),
                    "artifact_path_json": str(relaxed["artifact_path_json"]),
                }
            )
    return sorted(records, key=lambda row: row["relaxed_structure_id"])


def _pattern_grid(pattern: Any, *, fwhm: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.round(np.arange(20.0, 100.0 + 0.01, 0.02), 8)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    intensity = np.zeros_like(grid)
    for center, height in zip(pattern.x, pattern.y):
        intensity += float(height) * np.exp(-0.5 * ((grid - float(center)) / sigma) ** 2)
    maximum = float(intensity.max())
    if maximum <= 0:
        raise ValueError("XRD simulation produced an empty pattern")
    return grid, intensity / maximum * 100.0


def _site_occupancy_rows(record: dict[str, Any], structure: Structure) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(structure):
        for specie, occupancy in site.species.items():
            rows.append(
                {
                    "site_occupancy_id": f"site_{record['relaxed_structure_id']}_{site_index}_{specie.symbol}",
                    "relaxed_structure_id": record["relaxed_structure_id"],
                    "site_index": site_index,
                    "element": specie.symbol,
                    "occupancy": float(occupancy),
                    "frac_x": float(site.frac_coords[0]),
                    "frac_y": float(site.frac_coords[1]),
                    "frac_z": float(site.frac_coords[2]),
                    "cart_x": float(site.coords[0]),
                    "cart_y": float(site.coords[1]),
                    "cart_z": float(site.coords[2]),
                    "phase": record["phase"],
                    "sublattice_label": "parent_lattice_site",
                }
            )
    return rows


def _component_peak_rows(
    *,
    sample_id: str,
    phase: str,
    pattern: Any,
    fwhm: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for peak_index, (center, intensity, hkls) in enumerate(zip(pattern.x, pattern.y, pattern.hkls)):
        rows.append(
            {
                "component_peak_id": f"component_{sample_id}_{peak_index:04d}",
                "xrd_sample_id": sample_id,
                "component_index": peak_index,
                "phase": phase,
                "two_theta": float(center),
                "relative_intensity": float(intensity),
                "fwhm_deg": fwhm,
                "profile_kind": "gaussian",
                "hkls_json": json.dumps(hkls, sort_keys=True),
            }
        )
    return rows


def _task_rows(
    *,
    sample_id: str,
    phase: str,
    composition: dict[str, float],
    parent_lattice: float,
) -> list[dict[str, Any]]:
    targets = {
        "phase_classification": {"phase": phase},
        "composition_regression": {"element_fractions": composition},
        "lattice_parameter_regression": {
            "parent_lattice_parameter_angstrom": parent_lattice,
        },
    }
    return [
        {
            "task_sample_id": f"task_{task_name}_{sample_id}",
            "xrd_sample_id": sample_id,
            "task_name": task_name,
            "task_eligible": True,
            "input_setting": "xrd_only",
            "target_json": json.dumps(target, sort_keys=True),
            "exclusion_reason": "",
        }
        for task_name, target in targets.items()
    ]


def _composition_fractions(structure: Structure) -> dict[str, float]:
    amounts = structure.composition.get_el_amt_dict()
    total = float(sum(amounts.values()))
    return {element: float(amount / total) for element, amount in sorted(amounts.items())}


def _parent_lattice_parameter(structure: Structure, phase: str) -> float:
    atoms_per_conventional_cell = {"fcc": 4.0, "bcc": 2.0}.get(phase.lower())
    if atoms_per_conventional_cell is None:
        raise ValueError(f"Unsupported parent phase for v0.3 mini: {phase}")
    return float((atoms_per_conventional_cell * structure.volume / len(structure)) ** (1.0 / 3.0))


def _target_diagnostics(xrd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    structures = pd.DataFrame(xrd_rows).drop_duplicates("relaxed_structure_id")
    unique_by_group = {
        f"{alloy_system}/{phase}": int(group["parent_lattice_parameter_angstrom"].nunique())
        for (alloy_system, phase), group in structures.groupby(["alloy_system", "phase"])
    }
    return {
        "lattice_parameter_unique_count": int(
            structures["parent_lattice_parameter_angstrom"].nunique()
        ),
        "lattice_parameter_unique_by_alloy_phase": unique_by_group,
        "lattice_parameter_regression_nontrivial": any(
            count > 1 for count in unique_by_group.values()
        ),
    }


def _loo_nearest_neighbor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        candidates = [
            other
            for other in rows
            if other["initial_structure_id"] != row["initial_structure_id"]
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda other: float(np.mean((row["intensity"] - other["intensity"]) ** 2)),
        )
        predictions.append((row, nearest))
    if not predictions:
        return {
            "name": "leave_source_group_out_1nn",
            "scientific_result": False,
            "prediction_count": 0,
            "status": "insufficient_groups",
        }
    elements = sorted(
        {
            element
            for truth, prediction in predictions
            for item in (truth, prediction)
            for element in item["composition"]
        }
    )
    composition_errors = [
        abs(truth["composition"].get(element, 0.0) - prediction["composition"].get(element, 0.0))
        for truth, prediction in predictions
        for element in elements
    ]
    return {
        "name": "leave_source_group_out_1nn",
        "scientific_result": False,
        "status": "pass",
        "prediction_count": len(predictions),
        "phase_accuracy": float(np.mean([truth["phase"] == pred["phase"] for truth, pred in predictions])),
        "composition_macro_mae": float(np.mean(composition_errors)),
        "lattice_parameter_mae_angstrom": float(
            np.mean([abs(truth["parent_lattice"] - pred["parent_lattice"]) for truth, pred in predictions])
        ),
    }


def _unique_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(str(row[field]) for field in (
            "source_id",
            "source_type",
            "target_id",
            "target_type",
            "relation_type",
        ))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _parse_source(value: str) -> SourceSpec:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("source must be POTENTIAL_ID:ALLOY_SYSTEM:PHASE")
    return SourceSpec(parts[0], parts[1], parts[2].lower())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a real-structure hierarchical XRD mini release.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", default=".")
    build.add_argument("--output", default="data/v0.3_hierarchical_mini")
    build.add_argument("--source", action="append", type=_parse_source, required=True)
    args = parser.parse_args(argv)

    summary = build_hierarchical_mini(
        root=args.root,
        output=args.output,
        sources=args.source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
