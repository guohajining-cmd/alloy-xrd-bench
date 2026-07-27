"""Build the Ni-Al gamma/gamma-prime overlapping-peak benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from .contract import canonical_structure_hash
from .gamma_overlap_builder import (
    extract_windows,
    isolate_pair_split_components,
    load_overlap_config,
    maximum_cross_split_cosine,
    select_balanced_windows,
    synthesize_mixture,
)
from .gamma_overlap_baseline import (
    run_public_no_spectrum_prior,
    run_public_peak_feature_baseline,
    run_public_window_baseline,
    run_window_baseline,
)
from .gamma_overlap_physics import ComponentSpec, render_components
from .gamma_overlap_public import (
    evaluate_public_submission,
    export_public_release,
    make_submission_template,
    make_oracle_submission,
    validate_public_release,
)
from .gamma_overlap_scoring import score_component_predictions
from .gamma_overlap_sources import extract_parent_reflections, ingest_darus_cfg
from .gamma_overlap_training import train_set_baseline
from .gamma_overlap_compare import compare_parent_bootstrap
from .structures import _structure_from_recipe


def generate_gamma_library(
    *,
    output_dir: str | Path,
    count: int = 128,
    seed: int = 2027,
    candidate_count: int = 32,
    lattice_a: float = 3.57,
    lattice_a_values: Sequence[float] | None = None,
    lattice_source: str = "fixed_default_3p57_angstrom",
) -> list[dict[str, Any]]:
    """Generate unique deterministic 3x3x3 Ni97Al11 FCC gamma parents."""

    if count <= 0:
        raise ValueError("count must be positive")
    lattice_values = tuple(float(value) for value in (lattice_a_values or (lattice_a,)))
    if not lattice_values or any(not np.isfinite(value) or value <= 0 for value in lattice_values):
        raise ValueError("lattice_a_values must contain positive finite values")
    output = Path(output_dir).resolve()
    structures_dir = output / "structures" / "gamma"
    structures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    attempt = 0
    max_attempts = max(100, count * 100)
    while len(rows) < count and attempt < max_attempts:
        structure_seed = int(seed) + attempt
        selected_lattice_a = lattice_values[len(rows) % len(lattice_values)]
        recipe = {
            "alloy_system": "Ni-Al",
            "phase": "fcc",
            "composition": {"Ni": 97, "Al": 11},
            "supercell": [3, 3, 3],
            "lattice_a": selected_lattice_a,
            "seed": structure_seed,
            "generation_method": "pair_balance_occupancy",
            "candidate_count": int(candidate_count),
        }
        structure, generation_score, used_candidates = _structure_from_recipe(recipe)
        structure_doc = structure.as_dict()
        structure_hash = canonical_structure_hash(structure_doc)
        attempt += 1
        if structure_hash in hashes:
            continue
        hashes.add(structure_hash)
        parent_id = f"gamma_parent_{len(rows):04d}"
        artifact = structures_dir / f"{parent_id}.json"
        payload = json.dumps(structure_doc, indent=2, sort_keys=True).encode("utf-8")
        artifact.write_bytes(payload)
        rows.append(
            {
                "parent_structure_id": parent_id,
                "phase_role": "gamma",
                "source_dataset_id": "generated_ni97al11_fcc_v1",
                "source_record_id": f"seed_{structure_seed}",
                "composition": {"Al": 11, "Ni": 97},
                "natoms": 108,
                "supercell": "3x3x3",
                "lattice_a_angstrom": selected_lattice_a,
                "lattice_parameter_source": str(lattice_source),
                "seed": structure_seed,
                "generation_score": float(generation_score),
                "candidate_count": int(used_candidates),
                "structure_sha256": structure_hash,
                "artifact_file_sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_path": artifact.relative_to(output).as_posix(),
                "eligibility": "accepted",
                "exclusion_reason": "",
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"generated {len(rows)} unique gamma structures after {attempt} attempts")
    pd.DataFrame(rows).to_parquet(output / "gamma_parent_structures.parquet", index=False)
    return rows


def build_overlap_release(
    *,
    root: str | Path,
    output_dir: str | Path,
    cfg_path: str | Path,
    config_path: str | Path,
    mode: str = "pilot",
    pair_count: int | None = None,
    variants_per_pair: int | None = None,
    gamma_count: int = 128,
    seed: int = 2027,
) -> dict[str, Any]:
    """Build a complete pilot or production overlap release."""

    if mode not in {"pilot", "production"}:
        raise ValueError("mode must be pilot or production")
    pair_count = int(pair_count if pair_count is not None else (24 if mode == "pilot" else 400))
    variants_per_pair = int(variants_per_pair if variants_per_pair is not None else (12 if mode == "pilot" else 25))
    if pair_count <= 0 or variants_per_pair <= 0:
        raise ValueError("pair_count and variants_per_pair must be positive")
    root_path = Path(root).resolve()
    output = (root_path / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_overlap_config(config_path)

    ingest_report = ingest_darus_cfg(cfg_path=cfg_path, output_dir=output)
    gamma_rows = generate_gamma_library(
        output_dir=output,
        count=gamma_count,
        seed=seed,
        lattice_a_values=config.get("gamma_lattice_a_samples_angstrom"),
        lattice_source=str(config.get("gamma_lattice_parameter_source", "fixed_default_3p57_angstrom")),
    )
    darus = pd.read_parquet(output / "darus_parent_structures.parquet")
    accepted = darus[darus["eligibility"] == "accepted_core"].to_dict(orient="records")
    selected_gamma_prime = _select_gamma_prime_records(accepted, pair_count)
    if len(selected_gamma_prime) < pair_count:
        raise ValueError(f"requested {pair_count} pairs but only {len(selected_gamma_prime)} gamma-prime records are eligible")

    parent_rows = darus.to_dict(orient="records") + gamma_rows
    pd.DataFrame(parent_rows).to_parquet(output / "parent_structures.parquet", index=False)
    phase_patterns, phase_reflections, satellites, reflection_by_parent = _build_phase_patterns(
        output=output,
        gamma_rows=gamma_rows,
        gamma_prime_rows=selected_gamma_prime,
        config=config,
    )
    pd.DataFrame(phase_patterns).to_parquet(output / "phase_patterns.parquet", index=False)
    pd.DataFrame(phase_reflections).to_parquet(output / "phase_reflections.parquet", index=False)
    pd.DataFrame(satellites).to_parquet(output / "satellite_reflections.parquet", index=False)

    parent_pairs = _make_parent_pairs(
        selected_gamma_prime,
        gamma_rows,
        seed=seed,
        split_ratios=config["split_ratios"],
    )
    representative_samples: dict[str, dict[str, Any]] = {}
    representative_fingerprints = []
    for pair_index, pair in enumerate(parent_pairs):
        sample_id = f"nial_mix_{pair_index:04d}_000"
        representative = synthesize_mixture(
            sample_id=sample_id,
            parent_pair_id=str(pair["parent_pair_id"]),
            gamma_reflections=reflection_by_parent[str(pair["gamma_parent_id"])],
            gamma_prime_reflections=reflection_by_parent[str(pair["gamma_prime_parent_id"])],
            sample_seed=int(seed) + pair_index * 100_000,
            config=config,
        )
        representative_samples[str(pair["parent_pair_id"])] = representative
        representative_fingerprints.append(np.asarray(representative["clean"], dtype=float))
    parent_pairs = isolate_pair_split_components(
        parent_pairs,
        np.asarray(representative_fingerprints),
        seed=seed,
        ratios=config["split_ratios"],
        cosine_threshold=float(config.get("cross_split_cosine_threshold", 0.999)),
    )
    pd.DataFrame(parent_pairs).to_parquet(output / "parent_pairs.parquet", index=False)
    patterns_dir = output / "patterns"
    windows_dir = output / "windows"
    private_dir = output / "private"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = [
        _edge(
            str(row["parent_structure_id"]),
            "parent_structure",
            str(row["phase_pattern_id"]),
            "phase_pattern",
            "simulated_as",
        )
        for row in phase_patterns
    ]
    for pair in parent_pairs:
        edges.extend(
            [
                _edge(str(pair["gamma_parent_id"]), "parent_structure", str(pair["parent_pair_id"]), "parent_pair", "paired_as"),
                _edge(str(pair["gamma_prime_parent_id"]), "parent_structure", str(pair["parent_pair_id"]), "parent_pair", "paired_as"),
            ]
        )
    for pair_index, pair in enumerate(parent_pairs):
        for variant_index in range(variants_per_pair):
            sample_id = f"nial_mix_{pair_index:04d}_{variant_index:03d}"
            sample_seed = int(seed) + pair_index * 100_000 + variant_index
            sample = (
                representative_samples[str(pair["parent_pair_id"])]
                if variant_index == 0
                else synthesize_mixture(
                    sample_id=sample_id,
                    parent_pair_id=pair["parent_pair_id"],
                    gamma_reflections=reflection_by_parent[pair["gamma_parent_id"]],
                    gamma_prime_reflections=reflection_by_parent[pair["gamma_prime_parent_id"]],
                    sample_seed=sample_seed,
                    config=config,
                )
            )
            pattern_path = patterns_dir / f"{sample_id}.npz"
            np.savez_compressed(
                pattern_path,
                two_theta=sample["two_theta"],
                observed=sample["observed"],
                clean=sample["clean"],
                background=sample["background"],
            )
            pattern_sha256 = hashlib.sha256(pattern_path.read_bytes()).hexdigest()
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "parent_pair_id": pair["parent_pair_id"],
                    "gamma_parent_id": pair["gamma_parent_id"],
                    "gamma_prime_parent_id": pair["gamma_prime_parent_id"],
                    "split": pair["split"],
                    "sample_seed": sample_seed,
                    "config_id": config["config_id"],
                    "point_count": len(sample["two_theta"]),
                    "pattern_path": pattern_path.relative_to(output).as_posix(),
                    "pattern_sha256": pattern_sha256,
                    "gamma_prime_weight": sample["gamma_prime_weight"],
                    "instrument_fwhm_deg": sample["instrument_fwhm_deg"],
                    "microstrain_fraction": sample["microstrain_fraction"],
                    "crystallite_size_nm": sample["crystallite_size_nm"],
                    "eta": sample["eta"],
                    "zero_shift_deg": sample["zero_shift_deg"],
                    "background_fraction": sample["background_fraction"],
                    "gaussian_noise_fraction": sample["gaussian_noise_fraction"],
                    "poisson_scale": sample["poisson_scale"],
                }
            )
            for row in sample["component_rows"]:
                component_rows.append({**row, "split": pair["split"]})
                edges.append(
                    _edge(sample_id, "mixture_sample", str(row["component_id"]), "component", "decomposed_into")
                )
            for window in extract_windows(sample, config=config):
                window_path = windows_dir / f"{window['window_id']}.npz"
                window_rows.append(
                    {
                        key: value
                        for key, value in window.items()
                        if key not in {"two_theta", "observed", "clean", "target_component_ids"}
                    }
                    | {
                        "target_component_ids_json": json.dumps(window["target_component_ids"]),
                        "window_path": window_path.relative_to(output).as_posix(),
                        "split": pair["split"],
                    }
                )
                edges.append(
                    _edge(sample_id, "mixture_sample", str(window["window_id"]), "overlap_window", "cropped_as")
                )
                for component_id in window["target_component_ids"]:
                    edges.append(
                        _edge(str(component_id), "component", str(window["window_id"]), "overlap_window", "appears_in")
                    )
            edges.extend(
                [
                    _edge(str(pair["parent_pair_id"]), "parent_pair", sample_id, "mixture_sample", "augmented_as"),
                    _edge(f"phase_pattern_{pair['gamma_parent_id']}", "phase_pattern", sample_id, "mixture_sample", "contributes_to"),
                    _edge(f"phase_pattern_{pair['gamma_prime_parent_id']}", "phase_pattern", sample_id, "mixture_sample", "contributes_to"),
                ]
            )

    samples = pd.DataFrame(sample_rows)
    components = pd.DataFrame(component_rows)
    window_candidate_count = len(window_rows)
    selected_window_rows = select_balanced_windows(
        window_rows,
        seed=seed,
        single_fraction=float(config.get("single_window_fraction_target", 0.20)),
        maximum_overlap_class_ratio=float(config.get("maximum_overlap_class_ratio", 3.0)),
    )
    selected_window_ids = {str(row["window_id"]) for row in selected_window_rows}
    selected_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in selected_window_rows:
        selected_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    sample_path_by_id = {str(row["sample_id"]): str(row["pattern_path"]) for row in sample_rows}
    for sample_id, rows in selected_by_sample.items():
        with np.load(output / sample_path_by_id[sample_id]) as artifact:
            grid = np.asarray(artifact["two_theta"])
            observed = np.asarray(artifact["observed"])
            clean = np.asarray(artifact["clean"])
        for row in rows:
            start = int(np.argmin(np.abs(grid - float(row["two_theta_min"]))))
            stop = start + 151
            window_path = output / str(row["window_path"])
            np.savez_compressed(
                window_path,
                two_theta=grid[start:stop],
                observed=observed[start:stop],
                clean=clean[start:stop],
            )
            row["window_sha256"] = hashlib.sha256(window_path.read_bytes()).hexdigest()
    edges = [
        row
        for row in edges
        if row["target_type"] != "overlap_window" or str(row["target_id"]) in selected_window_ids
    ]
    windows = pd.DataFrame(selected_window_rows)
    component_windows: dict[str, list[str]] = {}
    for row in windows.to_dict(orient="records"):
        for component_id in json.loads(str(row["target_component_ids_json"])):
            component_windows.setdefault(str(component_id), []).append(str(row["window_id"]))
    components["window_ids_json"] = components["component_id"].map(
        lambda component_id: json.dumps(sorted(component_windows.get(str(component_id), [])))
    )
    samples.to_parquet(output / "mixture_samples.parquet", index=False)
    components[components["split"] != "private_test"].to_parquet(output / "components.parquet", index=False)
    components[components["split"] == "private_test"].to_parquet(private_dir / "components.parquet", index=False)
    public_windows = windows.copy()
    private_mask = public_windows["split"] == "private_test"
    private_truth = public_windows[private_mask].copy()
    for field in ("component_count", "difficulty", "minimum_separation_ratio", "target_component_ids_json"):
        public_windows.loc[private_mask, field] = None
    public_windows.to_parquet(output / "overlap_windows.parquet", index=False)
    private_truth.to_parquet(private_dir / "overlap_windows_truth.parquet", index=False)
    splits = samples[["sample_id", "parent_pair_id", "gamma_parent_id", "gamma_prime_parent_id", "split"]].copy()
    splits.to_parquet(output / "splits.parquet", index=False)
    pd.DataFrame(_unique_edges(edges)).to_parquet(output / "provenance_edges.parquet", index=False)

    report = audit_overlap_release(output_dir=output, config_path=config_path)
    summary = {
        "release_id": config["config_id"],
        "release_status": "pilot" if mode == "pilot" else "production",
        "synthetic_benchmark": True,
        "ingest": ingest_report,
        "gamma_parent_count": len(gamma_rows),
        "gamma_prime_parent_count": len(selected_gamma_prime),
        "parent_pair_count": len(parent_pairs),
        "mixture_sample_count": len(samples),
        "component_count": len(components),
        "window_count": len(windows),
        "window_candidate_count": window_candidate_count,
        "split_counts": samples["split"].value_counts().sort_index().to_dict(),
        "audit_status": report["status"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def audit_overlap_release(*, output_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    config = load_overlap_config(config_path)
    samples = pd.read_parquet(output / "mixture_samples.parquet")
    public_components = pd.read_parquet(output / "components.parquet")
    private_components = pd.read_parquet(output / "private" / "components.parquet")
    components = pd.concat([public_components, private_components], ignore_index=True)
    public_windows = pd.read_parquet(output / "overlap_windows.parquet")
    private_windows = pd.read_parquet(output / "private" / "overlap_windows_truth.parquet")
    checks: list[dict[str, Any]] = []
    components_by_sample = {key: group.to_dict(orient="records") for key, group in components.groupby("sample_id")}
    for sample in samples.to_dict(orient="records"):
        sample_id = str(sample["sample_id"])
        path = output / str(sample["pattern_path"])
        with np.load(path) as artifact:
            grid = artifact["two_theta"]
            clean = artifact["clean"]
            finite = all(np.isfinite(artifact[key]).all() for key in artifact.files)
        artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        specs = [
            ComponentSpec(
                phase=str(row["phase"]),
                hkl=(int(row["h"]), int(row["k"]), int(row["l"])),
                center_deg=float(row["center_deg"]),
                fwhm_deg=float(row["fwhm_deg"]),
                integrated_intensity=float(row["integrated_intensity"]),
                eta=float(row["eta"]),
            )
            for row in components_by_sample.get(sample_id, [])
        ]
        reconstruction, _ = render_components(grid, specs)
        error = float(np.max(np.abs(reconstruction - clean)))
        _audit_row(checks, "sample", sample_id, "finite_arrays", finite)
        _audit_row(
            checks,
            "sample",
            sample_id,
            "pattern_file_sha256",
            artifact_hash == str(sample["pattern_sha256"]),
            value=artifact_hash,
        )
        _audit_row(checks, "sample", sample_id, "point_count_4001", len(grid) == 4001, value=len(grid))
        _audit_row(checks, "sample", sample_id, "component_reconstruction", error < 1e-8, value=error)
    for row in components.to_dict(orient="records"):
        hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
        allowed = hkl != (0, 0, 0)
        if row["phase"] == "gamma":
            allowed = allowed and len({value % 2 for value in hkl}) == 1
        _audit_row(checks, "component", str(row["component_id"]), "legal_parent_hkl", allowed, value=hkl)
        finite_parameters = all(
            np.isfinite(float(row[field]))
            for field in ("center_deg", "fwhm_deg", "integrated_intensity", "eta")
        )
        _audit_row(checks, "component", str(row["component_id"]), "finite_peak_parameters", finite_parameters)
    split_leakage = samples.groupby("parent_pair_id")["split"].nunique().max() > 1
    _audit_row(checks, "release", "global", "parent_pair_split_isolation", not bool(split_leakage))
    for parent_field in ("gamma_parent_id", "gamma_prime_parent_id"):
        parent_leakage = samples.groupby(parent_field)["split"].nunique().max() > 1
        _audit_row(checks, "release", "global", f"{parent_field}_split_isolation", not bool(parent_leakage))
    parents = pd.read_parquet(output / "parent_structures.parquet")
    accepted_gamma_prime = parents[
        (parents["phase_role"] == "gamma_prime") & (parents["eligibility"] == "accepted_core")
    ]
    labels_complete = bool(
        len(accepted_gamma_prime)
        and accepted_gamma_prime[["has_energy", "has_forces", "has_stress"]].astype(bool).all(axis=None)
    )
    _audit_row(checks, "release", "global", "accepted_gamma_prime_energy_force_stress", labels_complete)
    parent_split_rows = []
    for sample in samples.to_dict(orient="records"):
        parent_split_rows.extend(
            [
                {"parent_structure_id": sample["gamma_parent_id"], "split": sample["split"]},
                {"parent_structure_id": sample["gamma_prime_parent_id"], "split": sample["split"]},
            ]
        )
    parent_splits = pd.DataFrame(parent_split_rows).drop_duplicates()
    selected_parents = parent_splits.merge(
        parents[["parent_structure_id", "structure_sha256"]], on="parent_structure_id", how="left"
    )
    hash_leakage = selected_parents.groupby("structure_sha256")["split"].nunique().max() > 1
    _audit_row(checks, "release", "global", "structure_hash_split_isolation", not bool(hash_leakage))
    clean_hash_splits: dict[str, set[str]] = {}
    for sample in samples.to_dict(orient="records"):
        with np.load(output / str(sample["pattern_path"])) as artifact:
            clean_hash = hashlib.sha256(np.asarray(artifact["clean"], dtype="<f8").tobytes()).hexdigest()
        clean_hash_splits.setdefault(clean_hash, set()).add(str(sample["split"]))
    exact_pattern_leakage = any(len(splits) > 1 for splits in clean_hash_splits.values())
    _audit_row(checks, "release", "global", "exact_clean_pattern_split_isolation", not exact_pattern_leakage)
    representative = samples.sort_values("sample_seed").groupby("parent_pair_id", as_index=False).first()
    fingerprints = []
    for row in representative.to_dict(orient="records"):
        with np.load(output / str(row["pattern_path"])) as artifact:
            fingerprints.append(np.asarray(artifact["clean"], dtype=float))
    cosine = maximum_cross_split_cosine(
        np.asarray(fingerprints),
        representative["split"].astype(str).tolist(),
        representative["parent_pair_id"].astype(str).tolist(),
    )
    cosine_threshold = float(config.get("cross_split_cosine_threshold", 0.999))
    _audit_row(
        checks,
        "release",
        "global",
        "parent_pair_fingerprint_cosine_isolation",
        cosine["maximum_cosine_similarity"] < cosine_threshold,
        value=cosine["maximum_cosine_similarity"],
    )
    for row in public_windows.to_dict(orient="records"):
        path = output / str(row["window_path"])
        with np.load(path) as artifact:
            valid = len(artifact["two_theta"]) == 151 and all(np.isfinite(artifact[key]).all() for key in artifact.files)
        _audit_row(checks, "window", str(row["window_id"]), "window_151_finite", valid)
        window_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        _audit_row(
            checks,
            "window",
            str(row["window_id"]),
            "window_file_sha256",
            window_hash == str(row["window_sha256"]),
            value=window_hash,
        )
    private_public_rows = public_windows[public_windows["split"] == "private_test"]
    hidden_fields = ("component_count", "difficulty", "minimum_separation_ratio", "target_component_ids_json")
    private_hidden = all(private_public_rows[field].isna().all() for field in hidden_fields)
    _audit_row(checks, "release", "global", "private_window_labels_hidden", private_hidden)
    all_window_truth = pd.concat(
        [public_windows[public_windows["split"] != "private_test"], private_windows],
        ignore_index=True,
    )
    single_fraction = float((all_window_truth["difficulty"] == "single").mean()) if len(all_window_truth) else 0.0
    single_target = float(config.get("single_window_fraction_target", 0.20))
    _audit_row(checks, "release", "global", "single_window_fraction", single_fraction >= single_target, value=single_fraction)
    split_class_ratios = []
    for _, split_windows in all_window_truth[all_window_truth["difficulty"] != "single"].groupby("split"):
        overlap_counts = split_windows["difficulty"].value_counts()
        split_class_ratios.append(float(overlap_counts.max() / overlap_counts.min()) if len(overlap_counts) else 1.0)
    overlap_class_ratio = max(split_class_ratios, default=1.0)
    maximum_class_ratio = float(config.get("maximum_overlap_class_ratio", 3.0))
    _audit_row(
        checks,
        "release",
        "global",
        "maximum_split_overlap_window_class_ratio",
        overlap_class_ratio <= maximum_class_ratio,
        value=overlap_class_ratio,
    )
    frame = pd.DataFrame(checks)
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(audit_dir / "checks.csv", index=False)
    fail_count = int((frame["status"] == "fail").sum())
    report = {
        "status": "pass" if fail_count == 0 else "fail",
        "checks": {"pass": int((frame["status"] == "pass").sum()), "fail": fail_count},
        "max_reconstruction_error": float(frame.loc[frame["check"] == "component_reconstruction", "value"].astype(float).max()),
        "single_window_fraction": single_fraction,
        "overlap_window_class_ratio": overlap_class_ratio,
        "maximum_cross_split_parent_pair_cosine": cosine["maximum_cosine_similarity"],
        "nearest_cross_split_parent_pairs": cosine["record_ids"],
    }
    (audit_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (audit_dir / "report.md").write_text(_audit_markdown(report), encoding="utf-8")
    return report


def _build_phase_patterns(
    *,
    output: Path,
    gamma_rows: Sequence[Mapping[str, Any]],
    gamma_prime_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    phase_dir = output / "phase_patterns"
    phase_dir.mkdir(parents=True, exist_ok=True)
    pattern_rows: list[dict[str, Any]] = []
    reflection_rows: list[dict[str, Any]] = []
    satellite_rows: list[dict[str, Any]] = []
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in list(gamma_rows) + list(gamma_prime_rows):
        parent_id = str(row["parent_structure_id"])
        phase = "gamma" if row["phase_role"] == "gamma" else "gamma_prime"
        structure = Structure.from_dict(json.loads((output / str(row["artifact_path"])).read_text(encoding="utf-8")))
        extracted = extract_parent_reflections(
            structure,
            phase=phase,
            wavelength_angstrom=float(config["wavelength_angstrom"]),
            two_theta_range=(float(config["two_theta_min"]), float(config["two_theta_max"])),
        )
        parents = [{**item, "parent_structure_id": parent_id} for item in extracted["parent_reflections"]]
        satellites_here = [{**item, "parent_structure_id": parent_id} for item in extracted["satellite_reflections"]]
        by_parent[parent_id] = parents
        reflection_rows.extend(parents)
        satellite_rows.extend(satellites_here)
        grid = np.round(np.arange(20.0, 100.0 + 0.01, 0.02), 8)
        specs = [
            ComponentSpec(phase, (item["h"], item["k"], item["l"]), item["center_deg"], 0.1, item["raw_integrated_intensity"], 0.5)
            for item in parents
        ]
        intensity, _ = render_components(grid, specs)
        phase_path = phase_dir / f"{parent_id}.npz"
        np.savez_compressed(phase_path, two_theta=grid, intensity=intensity)
        pattern_rows.append(
            {
                "phase_pattern_id": f"phase_pattern_{parent_id}",
                "parent_structure_id": parent_id,
                "phase": phase,
                "parent_reflection_count": len(parents),
                "satellite_reflection_count": len(satellites_here),
                "pattern_path": phase_path.relative_to(output).as_posix(),
            }
        )
    return pattern_rows, reflection_rows, satellite_rows, by_parent


def _select_gamma_prime_records(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: int(item["source_index"])):
        grouped.setdefault(str(row["source_run_id"]), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    depth = 0
    groups = [grouped[key] for key in sorted(grouped)]
    while len(selected) < count:
        added = False
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def _make_parent_pairs(
    gamma_prime_rows: Sequence[Mapping[str, Any]],
    gamma_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    split_ratios: Mapping[str, float],
) -> list[dict[str, Any]]:
    run_sizes: dict[str, int] = {}
    for row in gamma_prime_rows:
        run_id = str(row["source_run_id"])
        run_sizes[run_id] = run_sizes.get(run_id, 0) + 1
    ordered_runs = sorted(
        run_sizes,
        key=lambda value: (
            -run_sizes[value],
            hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
        ),
    )
    split_names = ("train", "validation", "private_test")
    targets = {split: len(gamma_prime_rows) * float(split_ratios[split]) for split in split_names}
    assigned_counts = {split: 0 for split in split_names}
    run_split: dict[str, str] = {}
    for run_id in ordered_runs:
        split = max(
            split_names,
            key=lambda name: (targets[name] - assigned_counts[name], -split_names.index(name)),
        )
        run_split[run_id] = split
        assigned_counts[split] += run_sizes[run_id]
    gamma_by_split = {split: [] for split in ("train", "validation", "private_test")}
    for index, row in enumerate(gamma_rows):
        split = ("train", "validation", "private_test")[index % 3]
        gamma_by_split[split].append(row)
    counters = {split: 0 for split in gamma_by_split}
    pairs = []
    for index, gamma_prime in enumerate(gamma_prime_rows):
        split = run_split[str(gamma_prime["source_run_id"])]
        candidates = gamma_by_split[split]
        gamma = candidates[counters[split] % len(candidates)]
        counters[split] += 1
        pairs.append(
            {
                "parent_pair_id": f"parent_pair_{index:04d}",
                "gamma_parent_id": gamma["parent_structure_id"],
                "gamma_prime_parent_id": gamma_prime["parent_structure_id"],
                "source_run_id": str(gamma_prime["source_run_id"]),
                "split": split,
            }
        )
    return pairs


def _edge(source_id: str, source_type: str, target_id: str, target_type: str, relation: str) -> dict[str, str]:
    return {"source_id": source_id, "source_type": source_type, "target_id": target_id, "target_type": target_type, "relation_type": relation}


def _unique_edges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = tuple(row[field] for field in ("source_id", "source_type", "target_id", "target_type", "relation_type"))
        if key not in seen:
            seen.add(key)
            result.append(dict(row))
    return result


def _audit_row(checks: list[dict[str, Any]], scope: str, record_id: str, check: str, condition: bool, *, value: Any = "") -> None:
    checks.append({"scope": scope, "record_id": record_id, "check": check, "status": "pass" if condition else "fail", "value": value})


def _audit_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Ni-Al Gamma/Gamma-Prime Overlap Audit",
            "",
            f"Status: `{report['status']}`",
            "",
            f"- Passed checks: {report['checks']['pass']}",
            f"- Failed checks: {report['checks']['fail']}",
            f"- Maximum clean reconstruction error: {report['max_reconstruction_error']:.3e}",
            f"- Single-window fraction: {report['single_window_fraction']:.3f}",
            f"- Maximum within-split overlap-window class ratio: {report['overlap_window_class_ratio']:.3f}",
            f"- Maximum cross-split parent-pair cosine: {report['maximum_cross_split_parent_pair_cosine']:.6f}",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ni-Al gamma/gamma-prime overlap benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--cfg", required=True)
    ingest.add_argument("--output", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--cfg", required=True)
    screen.add_argument("--output", required=True)
    for command in ("pilot", "generate"):
        item = subparsers.add_parser(command)
        item.add_argument("--root", default=".")
        item.add_argument("--cfg", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--config", default="configs/nickel_gamma_overlap_v1.json")
        item.add_argument("--pair-count", type=int)
        item.add_argument("--variants-per-pair", type=int)
        item.add_argument("--gamma-count", type=int, default=128)
        item.add_argument("--seed", type=int, default=2027)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--output", required=True)
    audit.add_argument("--config", default="configs/nickel_gamma_overlap_v1.json")
    score = subparsers.add_parser("score")
    score.add_argument("--truth", required=True)
    score.add_argument("--predictions", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--release", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--split", default="validation", choices=["train", "validation", "private_test"])
    baseline.add_argument("--max-windows", type=int)
    export_public = subparsers.add_parser("export-public")
    export_public.add_argument("--source", required=True)
    export_public.add_argument("--output", required=True)
    export_public.add_argument("--include-provenance", action="store_true")
    validate_release = subparsers.add_parser("validate-release")
    validate_release.add_argument("--release", required=True)
    template = subparsers.add_parser("make-submission-template")
    template.add_argument("--release", required=True)
    template.add_argument("--task", required=True, choices=["full", "window"])
    template.add_argument("--split", required=True, choices=["train", "validation", "test"])
    template.add_argument("--output", required=True)
    oracle = subparsers.add_parser("make-oracle")
    oracle.add_argument("--release", required=True)
    oracle.add_argument("--task", required=True, choices=["full", "window"])
    oracle.add_argument("--split", required=True, choices=["validation", "test"])
    oracle.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--release", required=True)
    evaluate.add_argument("--task", required=True, choices=["full", "window"])
    evaluate.add_argument("--split", required=True, choices=["validation", "test"])
    evaluate.add_argument("--submission", required=True)
    evaluate.add_argument("--output", required=True)
    train_baseline = subparsers.add_parser("train-baseline")
    train_baseline.add_argument("--release", required=True)
    train_baseline.add_argument("--task", required=True, choices=["full", "window"])
    train_baseline.add_argument("--split", default="validation", choices=["validation", "test"])
    train_baseline.add_argument(
        "--model", default="bic",
        choices=["no-spectrum", "peak-feature", "single", "bic", "resnet-set", "transformer-set"],
    )
    train_baseline.add_argument("--output", required=True)
    train_baseline.add_argument("--max-windows", type=int)
    train_baseline.add_argument("--seed", type=int, default=2027)
    train_baseline.add_argument("--epochs", type=int, default=50)
    train_baseline.add_argument("--patience", type=int, default=10)
    train_baseline.add_argument("--batch-size", type=int)
    train_baseline.add_argument("--embedding-dim", type=int, default=128)
    train_baseline.add_argument("--device", choices=["cpu", "cuda"])
    train_baseline.add_argument("--max-train-samples", type=int)
    train_baseline.add_argument("--max-validation-samples", type=int)
    train_baseline.add_argument("--max-evaluation-samples", type=int)
    train_baseline.add_argument("--reconstruction-weight", type=float, default=0.1)
    train_baseline.add_argument("--mask-weight", type=float, default=1.0)
    train_baseline.add_argument("--class-weight", type=float, default=1.0)
    train_baseline.add_argument("--parameter-weight", type=float, default=1.0)
    train_baseline.add_argument("--existence-weight", type=float, default=1.0)
    train_baseline.add_argument("--n-estimators", type=int, default=300)
    train_baseline.add_argument("--minimum-relative-intensity", type=float, default=0.0)
    train_baseline.add_argument("--training-spectrum-view", choices=["observed", "clean"], default="observed")
    compare_baselines = subparsers.add_parser("compare-baselines")
    compare_baselines.add_argument("--reference", required=True)
    compare_baselines.add_argument("--candidate", required=True)
    compare_baselines.add_argument("--output", required=True)
    compare_baselines.add_argument("--metric", default="component_f1")
    compare_baselines.add_argument("--replicates", type=int, default=2000)
    compare_baselines.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    if args.command == "compare-baselines":
        report = compare_parent_bootstrap(
            args.reference,
            args.candidate,
            args.output,
            metric=args.metric,
            replicates=args.replicates,
            seed=args.seed,
        )
    elif args.command in {"ingest", "screen"}:
        report = ingest_darus_cfg(cfg_path=args.cfg, output_dir=args.output)
    elif args.command in {"pilot", "generate"}:
        report = build_overlap_release(
            root=args.root,
            output_dir=args.output,
            cfg_path=args.cfg,
            config_path=args.config,
            mode="pilot" if args.command == "pilot" else "production",
            pair_count=args.pair_count,
            variants_per_pair=args.variants_per_pair,
            gamma_count=args.gamma_count,
            seed=args.seed,
        )
    elif args.command == "audit":
        report = audit_overlap_release(output_dir=args.output, config_path=args.config)
    elif args.command == "score":
        truth = pd.read_parquet(args.truth).to_dict(orient="records")
        predictions = pd.read_parquet(args.predictions).to_dict(orient="records")
        report = score_component_predictions(truth, predictions)
    elif args.command == "baseline":
        report = run_window_baseline(
            release_dir=args.release,
            output_path=args.output,
            split=args.split,
            max_windows=args.max_windows,
        )
    elif args.command == "export-public":
        report = export_public_release(
            args.source, args.output, include_provenance=args.include_provenance
        )
    elif args.command == "validate-release":
        report = validate_public_release(args.release)
    elif args.command == "make-submission-template":
        report = make_submission_template(
            args.release, task=args.task, split=args.split, output_dir=args.output
        )
    elif args.command == "make-oracle":
        report = make_oracle_submission(
            args.release, task=args.task, split=args.split, output_dir=args.output
        )
    elif args.command == "evaluate":
        report = evaluate_public_submission(
            args.release, args.task, args.split, args.submission, args.output
        )
    elif args.model == "no-spectrum":
        report = run_public_no_spectrum_prior(
            release_dir=args.release,
            output_dir=args.output,
            task=args.task,
            split=args.split,
        )
    elif args.model == "peak-feature":
        report = run_public_peak_feature_baseline(
            release_dir=args.release,
            output_dir=args.output,
            task=args.task,
            split=args.split,
            seed=args.seed,
            n_estimators=args.n_estimators,
        )
    elif args.model in {"single", "bic"}:
        if args.task != "window":
            parser.error("single and bic baselines support only --task window")
        report = run_public_window_baseline(
            release_dir=args.release,
            output_dir=args.output,
            split=args.split,
            baseline_kind=args.model,
            max_windows=args.max_windows,
        )
    else:
        report = train_set_baseline(
            release_dir=args.release,
            output_dir=args.output,
            task=args.task,
            encoder_name="resnet1d" if args.model == "resnet-set" else "spectrum_transformer",
            evaluation_split=args.split,
            seed=args.seed,
            maximum_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            device_name=args.device,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            max_evaluation_samples=args.max_evaluation_samples,
            reconstruction_weight=args.reconstruction_weight,
            mask_weight=args.mask_weight,
            class_weight=args.class_weight,
            parameter_weight=args.parameter_weight,
            existence_weight=args.existence_weight,
            minimum_relative_intensity=args.minimum_relative_intensity,
            training_spectrum_view=args.training_spectrum_view,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
