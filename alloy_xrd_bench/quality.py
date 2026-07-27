"""Physics and artifact sanity audits for Alloy-XRD-Bench workspaces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from .contract import DEFAULT_RELAX_CONFIG
from .validate import load_benchmark_manifests


def run_quality_audit(
    *,
    root: str | Path,
    output_dir: str | Path,
    min_neighbor_distance: float = 1.5,
) -> dict[str, Any]:
    """Run physics sanity checks and write CSV/JSON audit outputs."""

    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifests, load_issues = load_benchmark_manifests(root_path)
    rows: list[dict[str, Any]] = []
    rows.extend(_structure_distance_checks(root_path, manifests["initial_structures"], "initial", min_neighbor_distance))
    rows.extend(
        _structure_distance_checks(
            root_path,
            manifests["relaxed_structures"],
            "relaxed",
            min_neighbor_distance,
            path_field="artifact_path_json",
            id_field="relaxed_structure_id",
        )
    )
    rows.extend(_element_conservation_checks(root_path, manifests))
    rows.extend(_relaxation_threshold_checks(manifests["relaxations"]))
    rows.extend(_xrd_checks(root_path, manifests["xrd_samples"]))

    issue_count = sum(1 for row in rows if row["status"] != "pass") + len(load_issues)
    pd.DataFrame(rows).to_csv(output_path / "physics_sanity.csv", index=False)
    summary = {
        "root": str(root_path),
        "issue_count": issue_count,
        "load_issue_count": len(load_issues),
        "check_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
    }
    (output_path / "quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Alloy-XRD-Bench physics sanity audits.")
    parser.add_argument("root", help="Benchmark workspace or release root")
    parser.add_argument("--output-dir", default="audits", help="Audit output directory")
    parser.add_argument("--min-neighbor-distance", type=float, default=1.5)
    args = parser.parse_args(argv)

    summary = run_quality_audit(
        root=args.root,
        output_dir=args.output_dir,
        min_neighbor_distance=args.min_neighbor_distance,
    )
    print(f"check_count: {summary['check_count']}")
    print(f"issue_count: {summary['issue_count']}")
    print(f"output_dir: {Path(args.output_dir).resolve()}")
    return 0 if summary["issue_count"] == 0 else 1


def _structure_distance_checks(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    kind: str,
    min_neighbor_distance: float,
    *,
    path_field: str = "artifact_path",
    id_field: str = "initial_structure_id",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get(id_field, ""))
        try:
            structure = _load_structure(root / str(record[path_field]))
            min_distance = _min_distance(structure)
            status = "pass" if min_distance >= min_neighbor_distance else "fail"
            rows.append(
                _row(
                    check_name=f"{kind}_min_neighbor_distance",
                    record_id=record_id,
                    status=status,
                    value=min_distance,
                    threshold=f">={min_neighbor_distance}",
                    notes="",
                )
            )
        except Exception as exc:
            rows.append(
                _row(
                    check_name=f"{kind}_min_neighbor_distance",
                    record_id=record_id,
                    status="fail",
                    value="",
                    threshold=f">={min_neighbor_distance}",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def _element_conservation_checks(
    root: Path,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    initial_by_id = {
        str(record["initial_structure_id"]): record
        for record in manifests["initial_structures"]
    }
    relaxation_by_id = {
        str(record["relaxation_run_id"]): record
        for record in manifests["relaxations"]
    }
    rows: list[dict[str, Any]] = []
    for relaxed in manifests["relaxed_structures"]:
        relaxed_id = str(relaxed["relaxed_structure_id"])
        relaxation = relaxation_by_id.get(str(relaxed["relaxation_run_id"]), {})
        initial = initial_by_id.get(str(relaxation.get("initial_structure_id", "")), {})
        try:
            initial_structure = _load_structure(root / str(initial["artifact_path"]))
            relaxed_structure = _load_structure(root / str(relaxed["artifact_path_json"]))
            initial_counts = _element_counts(initial_structure)
            relaxed_counts = _element_counts(relaxed_structure)
            status = "pass" if initial_counts == relaxed_counts else "fail"
            rows.append(
                _row(
                    check_name="relaxed_element_conservation",
                    record_id=relaxed_id,
                    status=status,
                    value=json.dumps(relaxed_counts, sort_keys=True),
                    threshold=json.dumps(initial_counts, sort_keys=True),
                    notes="" if status == "pass" else "element counts changed",
                )
            )
        except Exception as exc:
            rows.append(
                _row(
                    check_name="relaxed_element_conservation",
                    record_id=relaxed_id,
                    status="fail",
                    value="",
                    threshold="initial counts",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def _relaxation_threshold_checks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("relaxation_run_id", ""))
        fmax = float(record.get("fmax", float("inf")))
        volume_change = abs(float(record.get("volume_change", float("inf"))))
        steps = int(record.get("steps", 10**12))
        checks = (
            ("relaxation_fmax", fmax, DEFAULT_RELAX_CONFIG["fmax_eV_per_A"], "<="),
            (
                "relaxation_volume_change",
                volume_change,
                DEFAULT_RELAX_CONFIG["abnormal_volume_change_threshold"],
                "<=",
            ),
            ("relaxation_steps", steps, DEFAULT_RELAX_CONFIG["max_steps"], "<="),
        )
        for check_name, value, threshold, op in checks:
            rows.append(
                _row(
                    check_name=check_name,
                    record_id=record_id,
                    status="pass" if value <= threshold else "fail",
                    value=value,
                    threshold=f"{op}{threshold}",
                    notes="",
                )
            )
    return rows


def _xrd_checks(root: Path, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("xrd_sample_id", ""))
        try:
            grid = pd.read_parquet(root / str(record["grid_path"]))
            peaks = pd.read_parquet(root / str(record["peak_path"]))
            two_theta_min = float(record["two_theta_min"])
            two_theta_max = float(record["two_theta_max"])
            max_intensity = float(grid["intensity"].max()) if len(grid) else 0.0
            rows.append(
                _row(
                    check_name="xrd_grid_nonempty",
                    record_id=record_id,
                    status="pass" if len(grid) > 0 else "fail",
                    value=len(grid),
                    threshold=">0",
                    notes="",
                )
            )
            rows.append(
                _row(
                    check_name="xrd_intensity_normalized",
                    record_id=record_id,
                    status="pass" if abs(max_intensity - 100.0) <= 1e-6 else "fail",
                    value=max_intensity,
                    threshold="100",
                    notes="",
                )
            )
            peaks_in_range = (
                len(peaks) > 0
                and peaks["two_theta"].between(two_theta_min, two_theta_max).all()
            )
            rows.append(
                _row(
                    check_name="xrd_peaks_in_range",
                    record_id=record_id,
                    status="pass" if peaks_in_range else "fail",
                    value=len(peaks),
                    threshold=f"{two_theta_min}..{two_theta_max}",
                    notes="",
                )
            )
        except Exception as exc:
            rows.append(
                _row(
                    check_name="xrd_artifacts_readable",
                    record_id=record_id,
                    status="fail",
                    value="",
                    threshold="readable",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


def _load_structure(path: Path) -> Structure:
    return Structure.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _min_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("inf")
    matrix = np.array(structure.distance_matrix, dtype=float)
    matrix[matrix <= 1e-12] = np.inf
    return float(matrix.min())


def _element_counts(structure: Structure) -> dict[str, int]:
    return dict(Counter(str(site.specie) for site in structure))


def _row(
    *,
    check_name: str,
    record_id: str,
    status: str,
    value: Any,
    threshold: Any,
    notes: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "record_id": record_id,
        "status": status,
        "value": value,
        "threshold": threshold,
        "notes": notes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
