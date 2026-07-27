"""Export public/private-controlled Alloy-XRD-Bench release artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .audit import write_audits
from .validate import load_benchmark_manifests


def export_release(
    *,
    root: str | Path,
    output_dir: str | Path,
    release_version: str,
) -> dict[str, Any]:
    """Export a release view that hides private-test labels and structures."""

    root_path = Path(root).resolve()
    release_path = Path(output_dir).resolve()
    manifests, load_issues = load_benchmark_manifests(root_path)
    release_manifests = release_path / "manifests"
    release_manifests.mkdir(parents=True, exist_ok=True)

    public_xrd = [row for row in manifests["xrd_samples"] if row.get("split") != "private_test"]
    private_xrd = [row for row in manifests["xrd_samples"] if row.get("split") == "private_test"]
    all_averages = _load_xrd_averages(root_path)
    public_averages = [
        row for row in all_averages
        if row.get("split") != "private_test"
        and set(row.get("source_xrd_sample_ids", [])) <= {xrd["xrd_sample_id"] for xrd in public_xrd}
    ]
    chain = _chain_indexes(manifests)
    public_ids = _public_chain_ids(public_xrd, chain)
    public_ids["xrd_average_ids"] = {str(row["xrd_average_id"]) for row in public_averages}

    public_initials = [
        row for row in manifests["initial_structures"]
        if str(row["initial_structure_id"]) in public_ids["initial_structure_ids"]
    ]
    public_relaxations = [
        row for row in manifests["relaxations"]
        if str(row["relaxation_run_id"]) in public_ids["relaxation_run_ids"]
    ]
    public_relaxed = [
        row for row in manifests["relaxed_structures"]
        if str(row["relaxed_structure_id"]) in public_ids["relaxed_structure_ids"]
    ]
    public_edges = _filter_public_edges(manifests["provenance_edges"], public_ids)
    private_inputs = [_private_input_row(row, chain) for row in private_xrd]
    catalog = [
        row for row in manifests["potentials_catalog"]
        if str(row["potential_id"]) in public_ids["potential_ids"]
    ]

    _write_catalog(release_manifests / "potentials_catalog.yml", catalog)
    _write_parquet(release_manifests / "initial_structures.parquet", public_initials)
    _write_parquet(release_manifests / "relaxations.parquet", public_relaxations)
    _write_parquet(release_manifests / "relaxed_structures.parquet", public_relaxed)
    _write_parquet(release_manifests / "xrd_samples.parquet", public_xrd)
    _write_parquet(release_manifests / "xrd_averages.parquet", public_averages)
    _write_parquet(release_manifests / "provenance_edges.parquet", public_edges)
    _write_parquet(release_manifests / "private_inputs.parquet", private_inputs)

    for row in public_initials:
        _copy_relative(root_path, release_path, row["artifact_path"])
    for row in public_relaxed:
        _copy_relative(root_path, release_path, row["artifact_path_json"])
        _copy_relative(root_path, release_path, row["artifact_path_cif"])
    for row in public_relaxations:
        _copy_relative(root_path, release_path, row["log_path"])
    for row in public_xrd:
        _copy_relative(root_path, release_path, row["grid_path"])
        _copy_relative(root_path, release_path, row["peak_path"])
    for row in public_averages:
        _copy_relative(root_path, release_path, row["grid_path"])
        _copy_relative(root_path, release_path, row["peak_path"])
    for row in catalog:
        _copy_potential_metadata(root_path, release_path, row)

    write_audits(root=release_path, output_dir=release_path / "audits", release_version=release_version)
    _copy_release_metadata(root_path, release_path)
    _write_release_summary(
        release_path / "release_summary.json",
        {
            "release_version": release_version,
            "public_xrd_samples": len(public_xrd),
            "public_xrd_averages": len(public_averages),
            "private_inputs": len(private_inputs),
            "public_initial_structures": len(public_initials),
            "public_relaxations": len(public_relaxations),
            "public_relaxed_structures": len(public_relaxed),
            "provenance_edges": len(public_edges),
            "potentials": len(catalog),
        },
    )
    return {
        "load_issue_count": len(load_issues),
        "public_xrd": len(public_xrd),
        "public_averages": len(public_averages),
        "private_inputs": len(private_inputs),
        "output_dir": str(release_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an Alloy-XRD-Bench release view.")
    parser.add_argument("root", help="Source benchmark workspace")
    parser.add_argument("--output-dir", required=True, help="Release output directory")
    parser.add_argument("--release-version", required=True)
    args = parser.parse_args(argv)

    report = export_release(
        root=args.root,
        output_dir=args.output_dir,
        release_version=args.release_version,
    )
    print(f"public_xrd: {report['public_xrd']}")
    print(f"public_averages: {report['public_averages']}")
    print(f"private_inputs: {report['private_inputs']}")
    print(f"output_dir: {report['output_dir']}")
    return 0


def _chain_indexes(manifests: Mapping[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "initials": {str(row["initial_structure_id"]): row for row in manifests["initial_structures"]},
        "relaxations": {str(row["relaxation_run_id"]): row for row in manifests["relaxations"]},
        "relaxed": {str(row["relaxed_structure_id"]): row for row in manifests["relaxed_structures"]},
    }


def _public_chain_ids(
    public_xrd: Sequence[Mapping[str, Any]],
    chain: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, set[str]]:
    ids = {
        "xrd_sample_ids": set(),
        "relaxed_structure_ids": set(),
        "relaxation_run_ids": set(),
        "initial_structure_ids": set(),
        "potential_ids": set(),
        "relax_config_ids": set(),
        "xrd_config_ids": set(),
    }
    for xrd in public_xrd:
        xrd_id = str(xrd["xrd_sample_id"])
        relaxed_id = str(xrd["relaxed_structure_id"])
        relaxed = chain["relaxed"].get(relaxed_id, {})
        relaxation_id = str(relaxed.get("relaxation_run_id", ""))
        relaxation = chain["relaxations"].get(relaxation_id, {})
        initial_id = str(relaxation.get("initial_structure_id", ""))
        ids["xrd_sample_ids"].add(xrd_id)
        ids["relaxed_structure_ids"].add(relaxed_id)
        ids["relaxation_run_ids"].add(relaxation_id)
        ids["initial_structure_ids"].add(initial_id)
        ids["potential_ids"].add(str(relaxation.get("potential_id", "")))
        ids["relax_config_ids"].add(str(relaxation.get("relax_config_id", "")))
        ids["xrd_config_ids"].add(str(xrd.get("xrd_config_id", "")))
    return ids


def _filter_public_edges(
    edges: Sequence[Mapping[str, Any]],
    public_ids: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    allowed_ids = set().union(*public_ids.values())
    return [
        dict(edge)
        for edge in edges
        if str(edge.get("source_id")) in allowed_ids and str(edge.get("target_id")) in allowed_ids
    ]


def _private_input_row(
    xrd: Mapping[str, Any],
    chain: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    relaxed = chain["relaxed"].get(str(xrd["relaxed_structure_id"]), {})
    relaxation = chain["relaxations"].get(str(relaxed.get("relaxation_run_id", "")), {})
    initial = chain["initials"].get(str(relaxation.get("initial_structure_id", "")), {})
    return {
        "xrd_sample_id": xrd.get("xrd_sample_id", ""),
        "split": xrd.get("split", ""),
        "initial_structure_id": initial.get("initial_structure_id", ""),
        "alloy_system": initial.get("alloy_system", ""),
        "phase": initial.get("phase", ""),
        "composition": _clean_composition(initial.get("composition", {})),
        "supercell": initial.get("supercell", ""),
        "generation_method": initial.get("generation_method", ""),
        "seed": initial.get("seed", ""),
    }


def _write_catalog(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(yaml.safe_dump({"potentials": list(rows)}, sort_keys=False), encoding="utf-8")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path)


def _copy_relative(source_root: Path, release_root: Path, relative_path: Any) -> None:
    relative = Path(str(relative_path))
    source = (source_root / relative).resolve()
    destination = release_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_potential_metadata(
    source_root: Path,
    release_root: Path,
    row: Mapping[str, Any],
) -> None:
    potential_dir = Path(str(row["potential_dir"]))
    policy = str(row.get("redistribution_policy", ""))
    for filename in ("metadata.yml", "checksums.sha256", "INSTALL.md"):
        source = source_root / potential_dir / filename
        if source.exists():
            destination = release_root / potential_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if policy == "redistributable":
        source_files = source_root / potential_dir / "files"
        if source_files.exists():
            shutil.copytree(source_files, release_root / potential_dir / "files", dirs_exist_ok=True)


def _copy_release_metadata(source_root: Path, release_root: Path) -> None:
    template_root = Path(__file__).resolve().parents[1]
    for filename in ("DATASET_CARD.md", "CITATION.cff", "croissant.jsonld", "README.md"):
        source = _first_existing(source_root / filename, template_root / filename)
        if source.exists():
            shutil.copy2(source, release_root / filename)
    for dirname in ("schemas", "configs", "MODEL_CARDS"):
        source_dir = _first_existing(source_root / dirname, template_root / dirname)
        if source_dir.exists():
            shutil.copytree(source_dir, release_root / dirname, dirs_exist_ok=True)
    docs_source = _first_existing(source_root / "docs", template_root / "docs")
    docs_target = release_root / "docs"
    docs_target.mkdir(parents=True, exist_ok=True)
    for filename in (
        "BENCHMARK_CONTRACT.md",
        "COMPLETION_AUDIT.md",
        "EXECUTION_WORKFLOW.md",
        "RELEASE_CHECKLIST.md",
        "REMAINING_WORK.md",
    ):
        source = docs_source / filename
        if source.exists():
            shutil.copy2(source, docs_target / filename)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _write_release_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True), encoding="utf-8")


def _load_xrd_averages(root: Path) -> list[dict[str, Any]]:
    data_root = root / "data" if (root / "data").exists() else root
    rows: list[dict[str, Any]] = []
    for path in sorted((data_root / "potentials").glob("*/generated/manifests/xrd_averages.parquet")):
        rows.extend(pd.read_parquet(path).to_dict(orient="records"))
    fallback = root / "manifests" / "xrd_averages.parquet"
    if fallback.exists():
        rows.extend(pd.read_parquet(fallback).to_dict(orient="records"))
    return rows


def _clean_composition(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, float] = {}
    for element, amount in value.items():
        if amount is None:
            continue
        try:
            number = float(amount)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        cleaned[str(element)] = number
    return cleaned


if __name__ == "__main__":
    raise SystemExit(main())
