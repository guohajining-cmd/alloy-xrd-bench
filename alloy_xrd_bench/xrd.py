"""Simulate XRD samples from relaxed structures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from .contract import DEFAULT_XRD_CONFIG, potential_workspace_paths
from .validate import load_benchmark_manifests


def simulate_xrd_for_potential(
    *,
    root: str | Path,
    potential_id: str,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Generate XRD grids, peaks, manifest rows, and provenance edges."""

    root_path = Path(root).resolve()
    paths = {
        name: root_path / relative
        for name, relative in potential_workspace_paths(potential_id).items()
    }
    manifests_dir = paths["manifests_dir"]
    relaxed_path = manifests_dir / "relaxed_structures.parquet"
    if not relaxed_path.exists():
        raise FileNotFoundError(f"Missing relaxed structures manifest: {relaxed_path}")

    paths["xrd_grid_dir"].mkdir(parents=True, exist_ok=True)
    paths["xrd_peaks_dir"].mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    relaxed_rows = pd.read_parquet(relaxed_path).to_dict(orient="records")
    xrd_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for relaxed in relaxed_rows:
        if str(relaxed.get("quality_tier", "pass")) != "pass":
            continue
        relaxed_id = str(relaxed["relaxed_structure_id"])
        xrd_sample_id = _xrd_sample_id(relaxed_id)
        structure_path = root_path / str(relaxed["artifact_path_json"])
        structure = Structure.from_dict(json.loads(structure_path.read_text(encoding="utf-8")))
        pattern = _xrd_pattern(structure)
        grid = _broadened_grid(pattern)
        peaks = _peak_table(pattern, xrd_sample_id)

        grid_relative = f"{potential_workspace_paths(potential_id)['xrd_grid_dir']}/{xrd_sample_id}.parquet"
        peak_relative = f"{potential_workspace_paths(potential_id)['xrd_peaks_dir']}/{xrd_sample_id}.parquet"
        grid.to_parquet(root_path / grid_relative)
        peaks.to_parquet(root_path / peak_relative)

        xrd_rows.append(
            {
                "xrd_sample_id": xrd_sample_id,
                "relaxed_structure_id": relaxed_id,
                "xrd_config_id": DEFAULT_XRD_CONFIG["xrd_config_id"],
                "two_theta_min": DEFAULT_XRD_CONFIG["two_theta_min"],
                "two_theta_max": DEFAULT_XRD_CONFIG["two_theta_max"],
                "step": DEFAULT_XRD_CONFIG["step"],
                "wavelength": DEFAULT_XRD_CONFIG["radiation"],
                "broadening": "gaussian_fwhm_0p1",
                "grid_path": grid_relative,
                "peak_path": peak_relative,
                "split": split,
            }
        )
        provenance_rows.extend(
            [
                {
                    "source_id": relaxed_id,
                    "source_type": "relaxed_structure",
                    "target_id": xrd_sample_id,
                    "target_type": "xrd_sample",
                    "relation_type": "simulated_as",
                },
                {
                    "source_id": DEFAULT_XRD_CONFIG["xrd_config_id"],
                    "source_type": "xrd_config",
                    "target_id": xrd_sample_id,
                    "target_type": "xrd_sample",
                    "relation_type": "configured_by",
                },
            ]
        )

    _upsert_manifest(manifests_dir / "xrd_samples.parquet", xrd_rows, "xrd_sample_id")
    _append_unique_edges(manifests_dir / "provenance_edges.parquet", provenance_rows)
    return xrd_rows


def average_xrd_for_potential(
    *,
    root: str | Path,
    potential_id: str,
) -> list[dict[str, Any]]:
    """Generate per-composition averaged XRD grids and peaks."""

    root_path = Path(root).resolve()
    relative_paths = potential_workspace_paths(potential_id)
    paths = {
        name: root_path / relative
        for name, relative in relative_paths.items()
    }
    manifests, _ = load_benchmark_manifests(root_path)
    context_by_xrd = _xrd_contexts(manifests)
    xrd_rows = [
        row for row in manifests["xrd_samples"]
        if _xrd_belongs_to_potential(row, context_by_xrd, potential_id)
    ]
    paths["xrd_average_grid_dir"].mkdir(parents=True, exist_ok=True)
    paths["xrd_average_peaks_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifests_dir"].mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in xrd_rows:
        context = context_by_xrd[str(row["xrd_sample_id"])]
        key = (
            str(context["alloy_system"]),
            str(context["phase"]),
            _composition_key(context["composition"]),
            str(row.get("split", "id")),
        )
        grouped.setdefault(key, []).append(row)

    average_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        alloy_system, phase, composition_key, split = key
        average_id = _xrd_average_id(potential_id, alloy_system, phase, composition_key, split)
        grids = [pd.read_parquet(root_path / str(row["grid_path"])) for row in rows]
        base_two_theta = grids[0]["two_theta"].to_numpy()
        intensities = []
        for grid in grids:
            if not np.allclose(base_two_theta, grid["two_theta"].to_numpy()):
                raise ValueError(f"Cannot average non-aligned XRD grids for {average_id}")
            intensities.append(grid["intensity"].to_numpy(dtype=float))
        average_intensity = np.mean(np.vstack(intensities), axis=0)
        if average_intensity.max() > 0:
            average_intensity = average_intensity / average_intensity.max() * float(
                DEFAULT_XRD_CONFIG["intensity_normalization"]["max_intensity"]
            )
        average_grid = pd.DataFrame({"two_theta": base_two_theta, "intensity": average_intensity})
        average_peaks = _peaks_from_grid(average_grid, average_id)
        grid_relative = f"{relative_paths['xrd_average_grid_dir']}/{average_id}.parquet"
        peak_relative = f"{relative_paths['xrd_average_peaks_dir']}/{average_id}.parquet"
        average_grid.to_parquet(root_path / grid_relative)
        average_peaks.to_parquet(root_path / peak_relative)
        source_ids = sorted(str(row["xrd_sample_id"]) for row in rows)
        average_rows.append(
            {
                "xrd_average_id": average_id,
                "potential_id": potential_id,
                "alloy_system": alloy_system,
                "phase": phase,
                "composition": _composition_from_key(composition_key),
                "split": split,
                "source_xrd_sample_ids": source_ids,
                "n_sources": len(source_ids),
                "xrd_config_id": DEFAULT_XRD_CONFIG["xrd_config_id"],
                "grid_path": grid_relative,
                "peak_path": peak_relative,
            }
        )
        provenance_rows.extend(
            {
                "source_id": source_id,
                "source_type": "xrd_sample",
                "target_id": average_id,
                "target_type": "xrd_average",
                "relation_type": "averaged_into",
            }
            for source_id in source_ids
        )

    _upsert_manifest(paths["manifests_dir"] / "xrd_averages.parquet", average_rows, "xrd_average_id")
    _append_unique_edges(paths["manifests_dir"] / "provenance_edges.parquet", provenance_rows)
    return average_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate Alloy-XRD-Bench XRD samples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Simulate XRD for one potential workspace")
    simulate.add_argument("--root", default=".", help="Benchmark workspace root")
    simulate.add_argument("--potential-id", required=True)
    simulate.add_argument("--split", default="train")

    average = subparsers.add_parser("average", help="Generate per-composition averaged XRD")
    average.add_argument("--root", default=".", help="Benchmark workspace root")
    average.add_argument("--potential-id", required=True)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        rows = simulate_xrd_for_potential(
            root=args.root,
            potential_id=args.potential_id,
            split=args.split,
        )
        print(f"generated_xrd_samples: {len(rows)}")
        return 0
    if args.command == "average":
        rows = average_xrd_for_potential(root=args.root, potential_id=args.potential_id)
        print(f"generated_xrd_averages: {len(rows)}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _xrd_pattern(structure: Structure, config: Mapping[str, Any] | None = None) -> Any:
    xrd_config = config or DEFAULT_XRD_CONFIG
    radiation = str(xrd_config["radiation"])
    wavelength = "CuKa" if radiation.lower().replace(" ", "") in {"cukalpha", "cuka"} else radiation
    calculator = XRDCalculator(wavelength=wavelength)
    return calculator.get_pattern(
        structure,
        two_theta_range=(xrd_config["two_theta_min"], xrd_config["two_theta_max"]),
    )


def _broadened_grid(pattern: Any, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    xrd_config = config or DEFAULT_XRD_CONFIG
    two_theta_min = float(xrd_config["two_theta_min"])
    two_theta_max = float(xrd_config["two_theta_max"])
    step = float(xrd_config["step"])
    fwhm = float(xrd_config["broadening"]["fwhm_deg"])
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    grid = np.round(np.arange(two_theta_min, two_theta_max + step / 2.0, step), 8)
    intensity = np.zeros_like(grid)

    for peak_two_theta, peak_intensity in zip(pattern.x, pattern.y):
        intensity += float(peak_intensity) * np.exp(-0.5 * ((grid - float(peak_two_theta)) / sigma) ** 2)

    max_intensity = float(intensity.max()) if len(intensity) else 0.0
    if max_intensity > 0:
        intensity = intensity / max_intensity * float(
            xrd_config["intensity_normalization"]["max_intensity"]
        )
    return pd.DataFrame({"two_theta": grid, "intensity": intensity})


def _peak_table(pattern: Any, xrd_sample_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (two_theta, intensity, hkls) in enumerate(zip(pattern.x, pattern.y, pattern.hkls)):
        rows.append(
            {
                "xrd_sample_id": xrd_sample_id,
                "peak_index": index,
                "two_theta": float(two_theta),
                "intensity": float(intensity),
                "hkls_json": json.dumps(hkls, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _peaks_from_grid(grid: pd.DataFrame, average_id: str) -> pd.DataFrame:
    two_theta = grid["two_theta"].to_numpy(dtype=float)
    intensity = grid["intensity"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(1, len(grid) - 1):
        if intensity[index] <= 1.0:
            continue
        if intensity[index] >= intensity[index - 1] and intensity[index] >= intensity[index + 1]:
            rows.append(
                {
                    "xrd_average_id": average_id,
                    "peak_index": len(rows),
                    "two_theta": float(two_theta[index]),
                    "intensity": float(intensity[index]),
                }
            )
    if not rows and len(grid):
        max_index = int(np.argmax(intensity))
        rows.append(
            {
                "xrd_average_id": average_id,
                "peak_index": 0,
                "two_theta": float(two_theta[max_index]),
                "intensity": float(intensity[max_index]),
            }
        )
    return pd.DataFrame(rows)


def _xrd_sample_id(relaxed_structure_id: str) -> str:
    if relaxed_structure_id.startswith("rstr_"):
        return f"xrd_{relaxed_structure_id[5:]}"
    return f"xrd_{relaxed_structure_id}"


def _xrd_average_id(
    potential_id: str,
    alloy_system: str,
    phase: str,
    composition_key: str,
    split: str,
) -> str:
    label = _safe_label(f"{potential_id}_{alloy_system}_{phase}_{composition_key}_{split}")
    return f"xrdavg_{label}"


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _composition_key(composition: Mapping[str, Any]) -> str:
    parts = []
    for element, amount in sorted(composition.items()):
        if amount is None:
            continue
        try:
            number = float(amount)
        except (TypeError, ValueError):
            continue
        if np.isnan(number):
            continue
        parts.append(f"{element}{number:g}")
    return "-".join(parts)


def _composition_from_key(key: str) -> dict[str, float]:
    composition: dict[str, float] = {}
    for part in key.split("-"):
        match = re.fullmatch(r"([A-Za-z]+)([-+0-9.eE]+)", part)
        if match:
            composition[match.group(1)] = float(match.group(2))
    return composition


def _xrd_contexts(manifests: Mapping[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    initials = {str(row["initial_structure_id"]): row for row in manifests["initial_structures"]}
    relaxations = {str(row["relaxation_run_id"]): row for row in manifests["relaxations"]}
    relaxed = {str(row["relaxed_structure_id"]): row for row in manifests["relaxed_structures"]}
    contexts: dict[str, dict[str, Any]] = {}
    for xrd in manifests["xrd_samples"]:
        relaxed_id = str(xrd["relaxed_structure_id"])
        relaxed_row = relaxed.get(relaxed_id, {})
        relaxation_id = str(relaxed_row.get("relaxation_run_id", ""))
        relaxation = relaxations.get(relaxation_id, {})
        initial_id = str(relaxation.get("initial_structure_id", ""))
        initial = initials.get(initial_id, {})
        contexts[str(xrd["xrd_sample_id"])] = {
            "potential_id": str(relaxation.get("potential_id", "")),
            "alloy_system": str(initial.get("alloy_system", "")),
            "phase": str(initial.get("phase", "")),
            "composition": initial.get("composition", {}),
        }
    return contexts


def _xrd_belongs_to_potential(
    row: Mapping[str, Any],
    context_by_xrd: Mapping[str, Mapping[str, Any]],
    potential_id: str,
) -> bool:
    return context_by_xrd.get(str(row["xrd_sample_id"]), {}).get("potential_id") == potential_id


def _upsert_manifest(path: Path, rows: list[dict[str, Any]], id_field: str) -> None:
    if path.exists():
        existing = pd.read_parquet(path).to_dict(orient="records")
    else:
        existing = []
    incoming_ids = {row[id_field] for row in rows}
    merged = [row for row in existing if row.get(id_field) not in incoming_ids] + rows
    pd.DataFrame(merged).to_parquet(path)


def _append_unique_edges(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        existing = pd.read_parquet(path).to_dict(orient="records")
    else:
        existing = []
    all_rows = existing + rows
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in all_rows:
        key = (
            str(row["source_id"]),
            str(row["source_type"]),
            str(row["target_id"]),
            str(row["target_type"]),
            str(row["relation_type"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    pd.DataFrame(unique).to_parquet(path)


if __name__ == "__main__":
    raise SystemExit(main())
