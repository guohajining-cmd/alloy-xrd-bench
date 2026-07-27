"""Generate occupancy-labeled XRD samples from relaxed structure CSV data."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from .contract import DEFAULT_XRD_CONFIG, canonical_structure_hash
from .occupancy_pipeline import (
    apply_review_results,
    audit_dataset,
    export_review_batches,
    inventory_dataset,
    label_dataset,
    simulate_dataset,
    split_dataset,
)
from .occupancy_production import (
    audit_production_run,
    finalize_production_run,
    plan_production_run,
    simulate_production_shard,
)
from .occupancy_expansion import (
    audit_observation_run,
    build_occupancy_benchmark,
    finalize_counterfactual_run,
    freeze_observations,
    generate_counterfactual_shard,
    plan_counterfactual_run,
)
from .xrd import _broadened_grid, _peak_table, _xrd_pattern


STRUCTURE_COLUMNS = [
    "dataset_structure_id",
    "fid",
    "formula",
    "chemical_system",
    "natoms",
    "space_group_number",
    "volume_per_atom",
    "e_per_atom",
    "e_above_hull",
    "elements",
    "artifact_path_json",
    "structure_sha256",
    "phase_hint",
    "corner_Al_fraction",
    "face_Ni_fraction",
    "l12_order_parameter",
    "xrd_eligible",
    "quality_tier",
    "quality_notes",
]
SITE_COLUMNS = [
    "dataset_structure_id",
    "fid",
    "site_index",
    "element",
    "frac_x",
    "frac_y",
    "frac_z",
    "sublattice_label",
]
XRD_SAMPLE_COLUMNS = [
    "xrd_sample_id",
    "dataset_structure_id",
    "fid",
    "formula",
    "chemical_system",
    "phase_hint",
    "xrd_config_id",
    "two_theta_min",
    "two_theta_max",
    "step",
    "wavelength",
    "broadening",
    "xrd_grid_path",
    "xrd_peak_path",
    "split",
    "superlattice_peak_count",
    "superlattice_to_fundamental_ratio",
    "quality_tier",
    "quality_notes",
]


def ingest_relaxed_structure_csv(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    min_neighbor_distance: float = 1.5,
    min_volume_per_atom: float = 5.0,
    max_volume_per_atom: float = 30.0,
) -> dict[str, Any]:
    """Extract final relaxed structures and occupancy labels from a HEA CSV."""

    csv_file = Path(csv_path)
    output_path = Path(output_dir)
    structures_dir = output_path / "structures" / "json"
    manifests_dir = output_path / "manifests"
    structures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    structure_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    seen = 0

    for chunk in pd.read_csv(csv_file, chunksize=250):
        for _, row in chunk.iterrows():
            if limit is not None and seen >= limit:
                break
            seen += 1
            parsed = _parse_csv_row(
                row.to_dict(),
                structures_dir,
                output_path,
                min_neighbor_distance,
                min_volume_per_atom,
                max_volume_per_atom,
            )
            structure_rows.append(parsed["structure_row"])
            site_rows.extend(parsed["site_rows"])
            parse_rows.append(parsed["parse_row"])
        if limit is not None and seen >= limit:
            break

    pd.DataFrame(structure_rows, columns=STRUCTURE_COLUMNS).to_parquet(
        manifests_dir / "structures.parquet"
    )
    pd.DataFrame(site_rows, columns=SITE_COLUMNS).to_parquet(
        manifests_dir / "occupancy_sites.parquet"
    )
    status = "pass" if structure_rows and all(row["quality_tier"] == "pass" for row in structure_rows) else "partial"
    report = {
        "status": status,
        "csv_path": str(csv_file),
        "structure_count": len(structure_rows),
        "pass_count": sum(1 for row in structure_rows if row["quality_tier"] == "pass"),
        "failed_count": sum(1 for row in structure_rows if row["quality_tier"] != "pass"),
        "files": parse_rows,
    }
    (manifests_dir / "parse_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def simulate_occupancy_xrd(
    *,
    dataset_dir: str | Path,
    split: str = "train",
) -> dict[str, Any]:
    """Generate XRD artifacts for pass-tier occupancy structures."""

    dataset_path = Path(dataset_dir)
    manifests_dir = dataset_path / "manifests"
    xrd_grid_dir = dataset_path / "xrd" / "grid"
    xrd_peak_dir = dataset_path / "xrd" / "peaks"
    xrd_grid_dir.mkdir(parents=True, exist_ok=True)
    xrd_peak_dir.mkdir(parents=True, exist_ok=True)

    structures = pd.read_parquet(manifests_dir / "structures.parquet").to_dict(orient="records")
    xrd_rows: list[dict[str, Any]] = []
    for row in structures:
        if row.get("quality_tier") != "pass" or not bool(row.get("xrd_eligible")):
            continue
        structure_id = str(row["dataset_structure_id"])
        xrd_sample_id = f"xrd_{structure_id}"
        structure_path = dataset_path / str(row["artifact_path_json"])
        structure = Structure.from_dict(json.loads(structure_path.read_text(encoding="utf-8")))
        pattern = _xrd_pattern(structure)
        grid = _broadened_grid(pattern)
        peaks = _annotated_peak_table(pattern, xrd_sample_id)

        grid_relative = f"xrd/grid/{xrd_sample_id}.parquet"
        peak_relative = f"xrd/peaks/{xrd_sample_id}.parquet"
        grid.to_parquet(dataset_path / grid_relative)
        peaks.to_parquet(dataset_path / peak_relative)

        super_count = int(peaks["is_superlattice_candidate"].sum()) if len(peaks) else 0
        fundamental = max(len(peaks) - super_count, 1)
        xrd_rows.append(
            {
                "xrd_sample_id": xrd_sample_id,
                "dataset_structure_id": structure_id,
                "fid": row["fid"],
                "formula": row["formula"],
                "chemical_system": row["chemical_system"],
                "phase_hint": row["phase_hint"],
                "xrd_config_id": DEFAULT_XRD_CONFIG["xrd_config_id"],
                "two_theta_min": DEFAULT_XRD_CONFIG["two_theta_min"],
                "two_theta_max": DEFAULT_XRD_CONFIG["two_theta_max"],
                "step": DEFAULT_XRD_CONFIG["step"],
                "wavelength": DEFAULT_XRD_CONFIG["radiation"],
                "broadening": "gaussian_fwhm_0p1",
                "xrd_grid_path": grid_relative,
                "xrd_peak_path": peak_relative,
                "split": split,
                "superlattice_peak_count": super_count,
                "superlattice_to_fundamental_ratio": super_count / fundamental,
                "quality_tier": "pass",
                "quality_notes": "",
            }
        )

    pd.DataFrame(xrd_rows, columns=XRD_SAMPLE_COLUMNS).to_parquet(
        manifests_dir / "xrd_occupancy_samples.parquet"
    )
    report = {
        "status": "pass" if xrd_rows else "partial",
        "dataset_dir": str(dataset_path),
        "xrd_sample_count": len(xrd_rows),
        "split": split,
    }
    (manifests_dir / "xrd_occupancy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate occupancy-labeled XRD samples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inventory and quality-check source CSV rows")
    inventory.add_argument("--csv", required=True, help="CSV containing structure_as_dict")
    inventory.add_argument("--output", required=True, help="Benchmark dataset output directory")
    inventory.add_argument("--limit", type=int, default=None)
    inventory.add_argument("--min-neighbor-distance", type=float, default=1.5)
    inventory.add_argument("--min-volume-per-atom", type=float, default=5.0)
    inventory.add_argument("--max-volume-per-atom", type=float, default=30.0)

    label = subparsers.add_parser("label", help="Generate parent-lattice and occupancy labels")
    label.add_argument("--dataset", required=True)
    label.add_argument("--pilot-size", type=int, default=1000)
    label.add_argument("--symprec", type=float, default=0.05)

    ingest = subparsers.add_parser("ingest", help="Extract relaxed structures and occupancy labels")
    ingest.add_argument("--csv", required=True, help="CSV containing structure_as_dict")
    ingest.add_argument("--output", required=True, help="Dataset output directory")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--min-neighbor-distance", type=float, default=1.5)
    ingest.add_argument("--min-volume-per-atom", type=float, default=5.0)
    ingest.add_argument("--max-volume-per-atom", type=float, default=30.0)

    simulate = subparsers.add_parser("simulate", help="Generate XRD for occupancy structures")
    simulate.add_argument("--dataset", required=True, help="Dataset directory from ingest")
    simulate.add_argument("--split", default="train")
    simulate.add_argument("--subset", default=None)
    simulate.add_argument("--shard-size", type=int, default=256)
    simulate.add_argument("--augmentations", type=int, default=0)
    simulate.add_argument("--run-id", default=None)
    simulate.add_argument("--shard-index", type=int, default=None)
    simulate.add_argument("--resume", action="store_true")
    simulate.add_argument("--overwrite", action="store_true")

    simulate_plan = subparsers.add_parser(
        "simulate-plan", help="Freeze a run-scoped production plan and source payload cache"
    )
    simulate_plan.add_argument("--dataset", required=True)
    simulate_plan.add_argument("--run-id", required=True)
    simulate_plan.add_argument("--subset", default="all")
    simulate_plan.add_argument("--config", required=True)
    simulate_plan.add_argument("--shard-size", type=int, default=256)
    simulate_plan.add_argument("--max-structures", type=int, default=None)
    simulate_plan.add_argument(
        "--purpose",
        choices=["xrd-artifact", "benchmark-release"],
        default="xrd-artifact",
        help="Defer semantic labels for XRD artifacts or require full benchmark readiness",
    )

    simulate_finalize = subparsers.add_parser(
        "simulate-finalize", help="Validate and merge all shards in a production run"
    )
    simulate_finalize.add_argument("--dataset", required=True)
    simulate_finalize.add_argument("--run-id", required=True)

    counterfactual_plan = subparsers.add_parser(
        "counterfactual-plan", help="Freeze parents and shards for occupancy counterfactuals"
    )
    counterfactual_plan.add_argument("--dataset", required=True)
    counterfactual_plan.add_argument("--run-id", required=True)
    counterfactual_plan.add_argument("--parents", type=int, required=True)
    counterfactual_plan.add_argument("--variants-per-parent", type=int, default=4)
    counterfactual_plan.add_argument("--config", required=True)
    counterfactual_plan.add_argument("--shard-size", type=int, default=64)

    counterfactual_generate = subparsers.add_parser(
        "counterfactual-generate", help="Generate one counterfactual structure and XRD shard"
    )
    counterfactual_generate.add_argument("--dataset", required=True)
    counterfactual_generate.add_argument("--run-id", required=True)
    counterfactual_generate.add_argument("--shard-index", type=int, required=True)
    counterfactual_generate.add_argument("--resume", action="store_true")

    counterfactual_finalize = subparsers.add_parser(
        "counterfactual-finalize", help="Validate and merge a counterfactual run"
    )
    counterfactual_finalize.add_argument("--dataset", required=True)
    counterfactual_finalize.add_argument("--run-id", required=True)

    observations_freeze = subparsers.add_parser(
        "observations-freeze", help="Freeze deterministic experimental observation variants"
    )
    observations_freeze.add_argument("--dataset", required=True)
    observations_freeze.add_argument("--output-dataset", default=None)
    observations_freeze.add_argument("--parent-groups", default=None)
    observations_freeze.add_argument("--run-id", required=True)
    observations_freeze.add_argument("--config", required=True)
    observations_freeze.add_argument("--variants", type=int, default=5)
    observations_freeze.add_argument("--splits", default="validation,test")
    observations_freeze.add_argument("--shard-size", type=int, default=256)

    observations_audit = subparsers.add_parser(
        "observations-audit", help="Audit a finalized frozen-observation run"
    )
    observations_audit.add_argument("--dataset", required=True)
    observations_audit.add_argument("--run-id", required=True)

    benchmark_build = subparsers.add_parser(
        "benchmark-build", help="Build occupancy task, pair, and split manifests"
    )
    benchmark_build.add_argument("--dataset", required=True)
    benchmark_build.add_argument("--counterfactual-run", required=True)
    benchmark_build.add_argument("--canonical-run", default=None)

    split = subparsers.add_parser("split", help="Create leakage-resistant benchmark splits")
    split.add_argument("--dataset", required=True)

    audit = subparsers.add_parser("audit", help="Audit accounting, XRD artifacts, and readiness")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--run-id", default=None)

    review_export = subparsers.add_parser(
        "review-export", help="Export pilot review batches with CIF and XRD evidence"
    )
    review_export.add_argument("--dataset", required=True)
    review_export.add_argument("--output", required=True)
    review_export.add_argument("--limit", type=int, default=None)
    review_export.add_argument("--batch-size", type=int, default=50)

    review_apply = subparsers.add_parser(
        "review-apply", help="Validate and apply edited pilot review CSV rows"
    )
    review_apply.add_argument("--dataset", required=True)
    review_apply.add_argument("--input", required=True)

    args = parser.parse_args(argv)
    if args.command == "inventory":
        report = inventory_dataset(
            csv_path=args.csv,
            output_dir=args.output,
            limit=args.limit,
            min_neighbor_distance=args.min_neighbor_distance,
            min_volume_per_atom=args.min_volume_per_atom,
            max_volume_per_atom=args.max_volume_per_atom,
        )
    elif args.command == "label":
        report = label_dataset(
            dataset_dir=args.dataset,
            pilot_size=args.pilot_size,
            symprec=args.symprec,
        )
    elif args.command == "ingest":
        report = ingest_relaxed_structure_csv(
            csv_path=args.csv,
            output_dir=args.output,
            limit=args.limit,
            min_neighbor_distance=args.min_neighbor_distance,
            min_volume_per_atom=args.min_volume_per_atom,
            max_volume_per_atom=args.max_volume_per_atom,
        )
    elif args.command == "simulate":
        if args.run_id is not None:
            if args.shard_index is None:
                parser.error("--shard-index is required with --run-id")
            report = simulate_production_shard(
                dataset_dir=args.dataset,
                run_id=args.run_id,
                shard_index=args.shard_index,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        else:
            staged_manifest = Path(args.dataset) / "manifests" / "structure_inventory.parquet"
            if args.subset is not None or staged_manifest.exists():
                report = simulate_dataset(
                    dataset_dir=args.dataset,
                    subset=args.subset or "pilot",
                    shard_size=args.shard_size,
                    split=args.split,
                    augmentations=args.augmentations,
                )
            else:
                report = simulate_occupancy_xrd(dataset_dir=args.dataset, split=args.split)
    elif args.command == "simulate-plan":
        report = plan_production_run(
            dataset_dir=args.dataset,
            run_id=args.run_id,
            subset=args.subset,
            config_path=args.config,
            shard_size=args.shard_size,
            max_structures=args.max_structures,
            purpose=args.purpose,
        )
    elif args.command == "simulate-finalize":
        report = finalize_production_run(dataset_dir=args.dataset, run_id=args.run_id)
    elif args.command == "counterfactual-plan":
        report = plan_counterfactual_run(
            dataset_dir=args.dataset,
            run_id=args.run_id,
            parents=args.parents,
            variants_per_parent=args.variants_per_parent,
            config_path=args.config,
            shard_size=args.shard_size,
        )
    elif args.command == "counterfactual-generate":
        report = generate_counterfactual_shard(
            dataset_dir=args.dataset,
            run_id=args.run_id,
            shard_index=args.shard_index,
            resume=args.resume,
        )
    elif args.command == "counterfactual-finalize":
        report = finalize_counterfactual_run(dataset_dir=args.dataset, run_id=args.run_id)
    elif args.command == "observations-freeze":
        report = freeze_observations(
            dataset_dir=args.dataset,
            output_dataset_dir=args.output_dataset,
            parent_groups_path=args.parent_groups,
            run_id=args.run_id,
            config_path=args.config,
            variants=args.variants,
            splits=tuple(value.strip() for value in args.splits.split(",") if value.strip()),
            shard_size=args.shard_size,
        )
    elif args.command == "observations-audit":
        report = audit_observation_run(dataset_dir=args.dataset, run_id=args.run_id)
    elif args.command == "benchmark-build":
        report = build_occupancy_benchmark(
            dataset_dir=args.dataset,
            counterfactual_run=args.counterfactual_run,
            canonical_run=args.canonical_run,
        )
    elif args.command == "split":
        report = split_dataset(dataset_dir=args.dataset)
    elif args.command == "audit":
        report = (
            audit_production_run(dataset_dir=args.dataset, run_id=args.run_id)
            if args.run_id is not None
            else audit_dataset(dataset_dir=args.dataset)
        )
    elif args.command == "review-export":
        report = export_review_batches(
            dataset_dir=args.dataset,
            output_dir=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    elif args.command == "review-apply":
        report = apply_review_results(dataset_dir=args.dataset, input_path=args.input)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def _parse_csv_row(
    row: Mapping[str, Any],
    structures_dir: Path,
    output_path: Path,
    min_neighbor_distance: float,
    min_volume_per_atom: float,
    max_volume_per_atom: float,
) -> dict[str, Any]:
    fid = _safe_id(str(row.get("fid", ""))) or "structure"
    structure_id = f"occstr_{fid}"
    notes: list[str] = []
    try:
        structure_doc = ast.literal_eval(str(row["structure_as_dict"]))
        structure = Structure.from_dict(structure_doc)
    except Exception as exc:
        structure_doc = {}
        structure = None
        notes.append(f"parse_failed:{type(exc).__name__}")

    if structure is not None:
        natoms = len(structure)
        volume_per_atom = _float_or_default(row.get("volume_per_atom"), structure.volume / max(natoms, 1))
        min_distance = _min_neighbor_distance(structure)
        if natoms < 1 or natoms > 125:
            notes.append("natoms_out_of_range")
        if not np.isfinite(structure.volume) or structure.volume <= 0:
            notes.append("invalid_volume")
        if volume_per_atom < min_volume_per_atom or volume_per_atom > max_volume_per_atom:
            notes.append("volume_per_atom_out_of_range")
        if min_distance < min_neighbor_distance:
            notes.append("min_neighbor_distance_below_threshold")
        artifact_relative = f"structures/json/{structure_id}.json"
        artifact_path = output_path / artifact_relative
        artifact_path.write_text(json.dumps(structure.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        labels = _occupancy_labels(structure)
        site_rows = _site_rows(structure_id, str(row.get("fid", "")), structure)
        elements = sorted(str(element) for element in structure.composition.elements)
        formula = str(row.get("reduced_formula") or structure.composition.reduced_formula)
        structure_hash = canonical_structure_hash(structure.as_dict())
    else:
        natoms = 0
        volume_per_atom = _float_or_default(row.get("volume_per_atom"), float("nan"))
        artifact_relative = ""
        labels = {
            "phase_hint": "invalid",
            "corner_Al_fraction": 0.0,
            "face_Ni_fraction": 0.0,
            "l12_order_parameter": 0.0,
        }
        site_rows = []
        elements = []
        formula = str(row.get("reduced_formula", ""))
        structure_hash = ""

    quality_tier = "pass" if not notes else "fail"
    structure_row = {
        "dataset_structure_id": structure_id,
        "fid": str(row.get("fid", "")),
        "formula": formula,
        "chemical_system": str(row.get("chemical_system", "")),
        "natoms": natoms,
        "space_group_number": int(_float_or_default(row.get("space_group_number"), 0)),
        "volume_per_atom": volume_per_atom,
        "e_per_atom": _float_or_default(row.get("e_per_atom"), float("nan")),
        "e_above_hull": _float_or_default(row.get("e_above_hull"), float("nan")),
        "elements": elements,
        "artifact_path_json": artifact_relative,
        "structure_sha256": structure_hash,
        "phase_hint": labels["phase_hint"],
        "corner_Al_fraction": labels["corner_Al_fraction"],
        "face_Ni_fraction": labels["face_Ni_fraction"],
        "l12_order_parameter": labels["l12_order_parameter"],
        "xrd_eligible": quality_tier == "pass",
        "quality_tier": quality_tier,
        "quality_notes": ";".join(notes),
    }
    return {
        "structure_row": structure_row,
        "site_rows": site_rows,
        "parse_row": {
            "fid": str(row.get("fid", "")),
            "status": quality_tier,
            "notes": ";".join(notes),
        },
    }


def _occupancy_labels(structure: Structure) -> dict[str, Any]:
    site_rows = _site_rows("", "", structure)
    corner = [row for row in site_rows if row["sublattice_label"] == "corner"]
    face = [row for row in site_rows if row["sublattice_label"] == "face"]
    corner_al = _fraction(corner, "Al")
    face_ni = _fraction(face, "Ni")
    l12_order = corner_al * face_ni if corner and face else 0.0
    return {
        "phase_hint": "l12_like" if l12_order >= 0.95 and len(corner) == 1 and len(face) == 3 else "unknown",
        "corner_Al_fraction": corner_al,
        "face_Ni_fraction": face_ni,
        "l12_order_parameter": l12_order,
    }


def _site_rows(structure_id: str, fid: str, structure: Structure) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, site in enumerate(structure):
        frac = np.mod(site.frac_coords, 1.0)
        rows.append(
            {
                "dataset_structure_id": structure_id,
                "fid": fid,
                "site_index": index,
                "element": _site_element(site),
                "frac_x": float(frac[0]),
                "frac_y": float(frac[1]),
                "frac_z": float(frac[2]),
                "sublattice_label": _sublattice_label(frac),
            }
        )
    return rows


def _annotated_peak_table(pattern: Any, xrd_sample_id: str) -> pd.DataFrame:
    peaks = _peak_table(pattern, xrd_sample_id)
    if peaks.empty:
        peaks = pd.DataFrame(
            columns=["xrd_sample_id", "peak_index", "two_theta", "intensity", "hkls_json"]
        )
    peaks["is_superlattice_candidate"] = [
        _hkls_have_mixed_parity(hkls_json) for hkls_json in peaks["hkls_json"]
    ]
    return peaks


def _hkls_have_mixed_parity(hkls_json: str) -> bool:
    try:
        hkls = json.loads(hkls_json)
    except json.JSONDecodeError:
        return False
    for item in hkls:
        hkl = item.get("hkl") if isinstance(item, Mapping) else None
        if hkl is None:
            continue
        parities = {abs(int(round(value))) % 2 for value in hkl}
        if len(parities) > 1:
            return True
    return False


def _sublattice_label(frac: np.ndarray, tolerance: float = 0.08) -> str:
    near_zero = [_near_periodic(value, 0.0, tolerance) for value in frac]
    near_half = [abs(float(value) - 0.5) <= tolerance for value in frac]
    if all(near_zero):
        return "corner"
    if sum(near_half) == 2 and sum(near_zero) == 1:
        return "face"
    return "other"


def _near_periodic(value: float, target: float, tolerance: float) -> bool:
    delta = abs(float(value) - target)
    return min(delta, abs(delta - 1.0)) <= tolerance


def _site_element(site: Any) -> str:
    try:
        if len(site.species) == 1:
            return str(next(iter(site.species.elements)))
    except Exception:
        pass
    return str(site.species_string)


def _fraction(rows: Sequence[Mapping[str, Any]], element: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["element"] == element) / len(rows)


def _min_neighbor_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("inf")
    matrix = np.array(structure.distance_matrix, dtype=float)
    matrix[matrix <= 1e-12] = np.inf
    return float(np.min(matrix))


def _float_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
