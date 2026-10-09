"""Controlled occupancy counterfactuals and experimental XRD observations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

from .contract import canonical_structure_hash
from .occupancy_pipeline import (
    XRD_COLUMNS,
    _parent_mapping_rms,
    _phase_peak_table,
    _sro,
    _structure_from_record,
    audit_dataset,
)
from .occupancy_production import (
    _atomic_write_h5,
    _atomic_write_json,
    _atomic_write_parquet,
    _atomic_write_text,
    _code_sha256,
    _config_points,
    _mapping_sha256,
    _read_json,
    _resolve_source_csv,
    _sha256_file,
    _validate_run_id,
    _write_structure_payload_cache,
)
from .xrd import _broadened_grid, _xrd_pattern


OBSERVATION_RENDERER_VERSION = "integrated_area_v2"
LEGACY_OBSERVATION_RENDERER_VERSION = "peak_height_v1"


COUNTERFACTUAL_STRATEGIES = (
    "local_swap",
    "randomized",
    "ordering_enhanced",
    "clustering_enhanced",
)

DERIVED_STRUCTURE_COLUMNS = [
    "structure_state_id", "dataset_structure_id", "parent_structure_id", "parent_group_id",
    "fid", "formula", "chemical_system", "nelements", "natoms", "strategy", "seed",
    "fidelity", "parent_lattice", "structure_family", "structure_sha256", "structure_json",
    "site_elements_json", "swapped_site_count", "composition_preserved",
    "template_mapping_rms", "occupancy_entropy", "occupancy_fingerprint",
    "sro_shell1_before_json", "sro_shell2_before_json", "sro_shell1_json",
    "sro_shell2_json", "sro_distance", "same_neighbor_fraction_before",
    "same_neighbor_fraction_after", "xrd_l1_distance", "xrd_cosine_similarity",
    "low_identifiability",
]


def adjudicate_parent_family(ptm_best: str, dominant_fraction: float, rmsd_p90: float) -> str:
    """Apply the frozen rule-assisted PTM family adjudication policy."""

    allowed = {"bcc_like", "fcc_like", "hcp_like"}
    if (
        ptm_best in allowed
        and float(dominant_fraction) >= 0.50
        and float(rmsd_p90) <= 0.20
    ):
        return ptm_best
    return "complex"


def plan_counterfactual_run(
    *,
    dataset_dir: str | Path,
    run_id: str,
    parents: int,
    variants_per_parent: int,
    config_path: str | Path,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Freeze eligible parents, source structures, config, and shard assignment."""

    dataset = Path(dataset_dir).resolve()
    _validate_run_id(run_id)
    if parents < 1 or variants_per_parent < 1 or shard_size < 1:
        return {"status": "fail", "reason": "counts_must_be_positive"}
    audit = audit_dataset(dataset_dir=dataset)
    if audit["status"] != "pass":
        return {"status": "fail", "reason": "dataset_engineering_gate_failed", "issues": audit["issues"]}
    if parents > 1000 and not bool(audit.get("benchmark_ready", False)):
        return {
            "status": "fail", "reason": "benchmark_release_gate_failed",
            "benchmark_ready": False,
            "manual_review_complete": bool(audit.get("manual_review_complete", False)),
            "manual_review_pass_rate": float(audit.get("manual_review_pass_rate", 0.0)),
        }
    config = _load_counterfactual_config(Path(config_path))
    strategies = list(config["strategies"])
    if variants_per_parent != len(strategies):
        return {
            "status": "fail", "reason": "variants_per_parent_must_match_strategies",
            "variants_per_parent": variants_per_parent, "strategy_count": len(strategies),
        }

    structures = pd.read_parquet(dataset / "manifests" / "structures.parquet")
    eligible = _eligible_parents(structures, config)
    selected = _select_balanced_parents(eligible, parents)
    if len(selected) < parents:
        return {
            "status": "fail", "reason": "insufficient_eligible_parents",
            "requested": parents, "available": len(eligible), "selected": len(selected),
        }
    run_root = dataset / "derived" / "occupancy" / run_id
    plan_path = run_root / "run_plan.json"
    requested_ids = selected["dataset_structure_id"].astype(str).tolist()
    if plan_path.exists():
        existing = _read_json(plan_path)
        compatible = (
            existing.get("parent_structure_ids") == requested_ids
            and int(existing.get("variants_per_parent", 0)) == variants_per_parent
            and int(existing.get("shard_size", 0)) == shard_size
            and _mapping_sha256(_read_json(run_root / "counterfactual_config.json"))
            == _mapping_sha256(config)
        )
        if not compatible:
            return {"status": "fail", "reason": "run_plan_incompatible", "run_id": run_id}
        return {
            "status": "pass", "action": "existing", "run_id": run_id,
            "parent_count": existing["parent_count"],
            "expected_variant_count": existing["expected_variant_count"],
            "shard_count": existing["shard_count"],
        }
    if run_root.exists() and any(run_root.iterdir()):
        return {"status": "fail", "reason": "run_directory_exists_without_plan", "run_id": run_id}

    for name in ("structures", "labels", "sites", "completed"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    xrd_root = dataset / "xrd" / "runs" / f"{run_id}_xrd"
    for name in ("grids", "peaks", "manifests"):
        (xrd_root / name).mkdir(parents=True, exist_ok=True)
    config_out = run_root / "counterfactual_config.json"
    _atomic_write_json(config_out, config)
    summary = _read_json(dataset / "reports" / "inventory_summary.json")
    source = _resolve_source_csv(dataset, summary)
    source_cache = run_root / "structure_payloads.parquet"
    _write_structure_payload_cache(source, selected, source_cache)
    parent_groups = selected.copy()
    parent_groups["parent_group_id"] = parent_groups["dataset_structure_id"]
    parent_groups["fidelity"] = "source_relaxed_dft"
    _atomic_write_parquet(run_root / "parent_groups.parquet", parent_groups)

    source_rows = selected["source_row"].astype(int).tolist()
    parent_hashes = selected["structure_sha256"].astype(str).tolist()
    shard_count = math.ceil(parents / shard_size)
    shards = []
    for index in range(shard_count):
        start = index * shard_size
        stop = min(start + shard_size, parents)
        shards.append(
            {
                "index": index, "start": start, "stop": stop, "count": stop - start,
                "parent_structure_ids": requested_ids[start:stop],
                "parent_hashes": parent_hashes[start:stop],
                "source_rows": source_rows[start:stop],
            }
        )
    plan: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "parent_count": parents,
        "variants_per_parent": variants_per_parent,
        "expected_variant_count": parents * variants_per_parent,
        "shard_size": shard_size, "shard_count": shard_count,
        "parent_structure_ids": requested_ids, "parent_hashes": parent_hashes,
        "config_sha256": _sha256_file(config_out),
        "source_payload_sha256": _sha256_file(source_cache),
        "code_sha256": _code_sha256(Path(__file__).resolve().parents[1]),
        "xrd_run_id": f"{run_id}_xrd", "points_per_pattern": _config_points(config["xrd_config"]),
        "shards": shards,
    }
    plan["plan_sha256"] = _mapping_sha256(plan)
    _atomic_write_json(plan_path, plan)
    return {
        "status": "pass", "action": "created", "run_id": run_id,
        "parent_count": parents, "expected_variant_count": parents * variants_per_parent,
        "shard_count": shard_count, "plan_sha256": plan["plan_sha256"],
    }


def generate_counterfactual_shard(
    *, dataset_dir: str | Path, run_id: str, shard_index: int, resume: bool = False,
) -> dict[str, Any]:
    """Generate one atomic shard of occupancy variants and clean XRD."""

    dataset = Path(dataset_dir).resolve()
    run_root = dataset / "derived" / "occupancy" / run_id
    plan, config = _load_counterfactual_plan(run_root)
    if shard_index < 0 or shard_index >= int(plan["shard_count"]):
        return {"status": "fail", "reason": "shard_index_out_of_range", "shard_index": shard_index}
    marker = run_root / "completed" / f"shard_{shard_index:04d}.json"
    if marker.exists():
        valid, issues = _validate_counterfactual_shard(dataset, run_root, plan, shard_index)
        if valid and resume:
            return {"status": "pass", "action": "skipped", "run_id": run_id, "shard_index": shard_index}
        if not valid:
            return {"status": "fail", "reason": "completed_shard_invalid", "issues": issues}
        return {"status": "fail", "reason": "shard_already_complete_use_resume"}

    shard = plan["shards"][shard_index]
    parent_groups = pd.read_parquet(run_root / "parent_groups.parquet")
    parents = parent_groups[parent_groups["dataset_structure_id"].isin(shard["parent_structure_ids"])]
    parents = parents.set_index("dataset_structure_id").loc[shard["parent_structure_ids"]].reset_index()
    payloads = pd.read_parquet(run_root / "structure_payloads.parquet")
    records = {int(row["source_row"]): row for row in payloads.to_dict(orient="records")}
    derived_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    xrd_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    sample_ids: list[str] = []
    two_theta: np.ndarray | None = None
    xrd_config = config["xrd_config"]
    for parent in parents.to_dict(orient="records"):
        structure = _structure_from_record(records[int(parent["source_row"])])
        if not all(site.is_ordered and math.isclose(float(site.species.num_atoms), 1.0) for site in structure):
            raise ValueError(f"Parent structure is not fully occupied: {parent['dataset_structure_id']}")
        parent_grid_frame = _broadened_grid(_xrd_pattern(structure, xrd_config), xrd_config)
        parent_grid = parent_grid_frame["intensity"].to_numpy(dtype=float)
        variants = generate_counterfactual_variants(
            structure,
            parent_structure_id=str(parent["dataset_structure_id"]),
            parent_lattice=str(parent["parent_lattice"]),
            parent_hash=str(parent["structure_sha256"]),
            config=config,
        )
        for variant in variants:
            derived = Structure.from_dict(json.loads(variant["structure_json"]))
            pattern = _xrd_pattern(derived, xrd_config)
            grid_frame = _broadened_grid(pattern, xrd_config)
            if two_theta is None:
                two_theta = grid_frame["two_theta"].to_numpy(dtype=np.float32)
            grid = grid_frame["intensity"].to_numpy(dtype=np.float32)
            variant["xrd_l1_distance"] = float(np.mean(np.abs(grid.astype(float) - parent_grid)) / 100.0)
            denominator = float(np.linalg.norm(grid) * np.linalg.norm(parent_grid))
            variant["xrd_cosine_similarity"] = float(np.dot(grid, parent_grid) / denominator) if denominator else 1.0
            variant["low_identifiability"] = bool(variant["xrd_l1_distance"] < 0.01)
            variant.update(
                {
                    "fid": parent["fid"], "formula": parent["formula"],
                    "chemical_system": parent["chemical_system"],
                    "nelements": int(parent["nelements"]), "natoms": int(parent["natoms"]),
                }
            )
            derived_rows.append(variant)
            label_rows.append({key: variant[key] for key in (
                "structure_state_id", "parent_structure_id", "parent_lattice", "strategy",
                "sro_shell1_json", "sro_shell2_json", "sro_distance", "occupancy_entropy",
                "same_neighbor_fraction_before", "same_neighbor_fraction_after",
                "xrd_l1_distance", "xrd_cosine_similarity", "low_identifiability",
            )})
            for site_index, site in enumerate(derived):
                frac = np.mod(site.frac_coords, 1.0)
                site_rows.append(
                    {
                        "structure_state_id": variant["structure_state_id"],
                        "parent_structure_id": variant["parent_structure_id"],
                        "site_index": site_index, "element": site.specie.symbol,
                        "frac_x": float(frac[0]), "frac_y": float(frac[1]), "frac_z": float(frac[2]),
                        "parent_lattice_site": "parent_lattice_all_sites",
                    }
                )
            xrd_id = f"xrd_{variant['structure_state_id']}_clean"
            local_peaks = _phase_peak_table(pattern, variant)
            for peak in local_peaks:
                peak_rows.append({"xrd_sample_id": xrd_id, **peak})
            grid_index = len(grids)
            grids.append(grid)
            sample_ids.append(xrd_id)
            relative_root = f"xrd/runs/{plan['xrd_run_id']}"
            xrd_rows.append(
                {
                    "xrd_sample_id": xrd_id, "dataset_structure_id": variant["structure_state_id"],
                    "fid": parent["fid"], "formula": parent["formula"],
                    "chemical_system": parent["chemical_system"],
                    "structure_family": variant["structure_family"],
                    "parent_lattice": variant["parent_lattice"],
                    "benchmark_subset": "hea_bcc_fcc_sqs", "variant": "clean",
                    "xrd_config_id": xrd_config["xrd_config_id"],
                    "two_theta_min": xrd_config["two_theta_min"],
                    "two_theta_max": xrd_config["two_theta_max"], "step": xrd_config["step"],
                    "wavelength": xrd_config["radiation"],
                    "broadening": f"gaussian_fwhm_{xrd_config['broadening']['fwhm_deg']}",
                    "grid_shard_path": f"{relative_root}/grids/grids_{shard_index:04d}.h5",
                    "grid_row_index": grid_index,
                    "peak_shard_path": f"{relative_root}/peaks/peaks_{shard_index:04d}.parquet",
                    "split": "unassigned", "superlattice_peak_count": 0,
                    "weak_peak_count": sum(float(p["relative_intensity"]) < 1.0 for p in local_peaks),
                    "peak_count": len(local_peaks), "perturbation_json": "{}",
                }
            )

    structure_path = run_root / "structures" / f"structures_{shard_index:04d}.parquet"
    label_path = run_root / "labels" / f"sro_{shard_index:04d}.parquet"
    site_path = run_root / "sites" / f"sites_{shard_index:04d}.parquet"
    xrd_root = dataset / "xrd" / "runs" / plan["xrd_run_id"]
    grid_path = xrd_root / "grids" / f"grids_{shard_index:04d}.h5"
    peak_path = xrd_root / "peaks" / f"peaks_{shard_index:04d}.parquet"
    sample_path = xrd_root / "manifests" / f"samples_{shard_index:04d}.parquet"
    artifacts = (structure_path, label_path, site_path, grid_path, peak_path, sample_path)
    if any(path.exists() for path in artifacts):
        return {"status": "fail", "reason": "unmarked_shard_artifacts_exist", "shard_index": shard_index}
    _atomic_write_parquet(structure_path, pd.DataFrame(derived_rows, columns=DERIVED_STRUCTURE_COLUMNS))
    _atomic_write_parquet(label_path, pd.DataFrame(label_rows))
    _atomic_write_parquet(site_path, pd.DataFrame(site_rows))
    _atomic_write_h5(grid_path, two_theta, grids, sample_ids)
    _atomic_write_parquet(peak_path, pd.DataFrame(peak_rows))
    _atomic_write_parquet(sample_path, pd.DataFrame(xrd_rows, columns=XRD_COLUMNS))
    marker_payload = {
        "schema_version": 1, "run_id": run_id, "shard_index": shard_index,
        "plan_sha256": plan["plan_sha256"], "parent_count": int(shard["count"]),
        "variant_count": len(derived_rows),
        "checksums": {path.relative_to(dataset).as_posix(): _sha256_file(path) for path in artifacts},
    }
    _atomic_write_json(marker, marker_payload)
    return {
        "status": "pass", "action": "generated", "run_id": run_id,
        "shard_index": shard_index, "parent_count": int(shard["count"]),
        "counterfactual_count": len(derived_rows),
    }


def finalize_counterfactual_run(*, dataset_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Validate all counterfactual shards and publish merged manifests."""

    dataset = Path(dataset_dir).resolve()
    run_root = dataset / "derived" / "occupancy" / run_id
    plan, _ = _load_counterfactual_plan(run_root)
    missing: list[int] = []
    invalid: dict[str, list[str]] = {}
    structures: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    sites: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []
    for shard_index in range(int(plan["shard_count"])):
        marker = run_root / "completed" / f"shard_{shard_index:04d}.json"
        if not marker.exists():
            missing.append(shard_index)
            continue
        valid, issues = _validate_counterfactual_shard(dataset, run_root, plan, shard_index)
        if not valid:
            invalid[str(shard_index)] = issues
            continue
        structures.append(pd.read_parquet(run_root / "structures" / f"structures_{shard_index:04d}.parquet"))
        labels.append(pd.read_parquet(run_root / "labels" / f"sro_{shard_index:04d}.parquet"))
        sites.append(pd.read_parquet(run_root / "sites" / f"sites_{shard_index:04d}.parquet"))
        xrd_root = dataset / "xrd" / "runs" / plan["xrd_run_id"]
        samples.append(pd.read_parquet(xrd_root / "manifests" / f"samples_{shard_index:04d}.parquet"))
    if missing or invalid:
        return {
            "status": "fail", "reason": "run_incomplete_or_invalid",
            "missing_shards": missing, "invalid_shards": invalid,
        }
    derived = pd.concat(structures, ignore_index=True).sort_values("structure_state_id", kind="stable")
    sro = pd.concat(labels, ignore_index=True).sort_values("structure_state_id", kind="stable")
    occupancy_sites = pd.concat(sites, ignore_index=True).sort_values(
        ["structure_state_id", "site_index"], kind="stable"
    )
    xrd_samples = pd.concat(samples, ignore_index=True).sort_values("dataset_structure_id", kind="stable")
    expected = int(plan["expected_variant_count"])
    issues: list[str] = []
    if len(derived) != expected or derived["structure_state_id"].nunique() != expected:
        issues.append("counterfactual_count_or_id_uniqueness_invalid")
    if derived["structure_sha256"].nunique() != expected:
        issues.append("duplicate_counterfactual_structure_hash")
    parent_hashes = set(pd.read_parquet(run_root / "parent_groups.parquet")["structure_sha256"].astype(str))
    if set(derived["structure_sha256"].astype(str)) & parent_hashes:
        issues.append("counterfactual_hash_matches_parent_structure")
    if not derived["composition_preserved"].astype(bool).all():
        issues.append("composition_not_preserved")
    if len(xrd_samples) != expected or xrd_samples["dataset_structure_id"].nunique() != expected:
        issues.append("xrd_count_or_structure_uniqueness_invalid")
    strategy_scores = derived.pivot(
        index="parent_group_id", columns="strategy", values="same_neighbor_fraction_after"
    )
    ordering_success = float(
        (strategy_scores["ordering_enhanced"] < strategy_scores["randomized"]).mean()
    ) if {"ordering_enhanced", "randomized"}.issubset(strategy_scores.columns) else 1.0
    clustering_success = float(
        (strategy_scores["clustering_enhanced"] > strategy_scores["randomized"]).mean()
    ) if {"clustering_enhanced", "randomized"}.issubset(strategy_scores.columns) else 1.0
    if ordering_success < 0.95:
        issues.append("ordering_direction_success_below_95_percent")
    if clustering_success < 0.95:
        issues.append("clustering_direction_success_below_95_percent")
    if issues:
        report = {
            "status": "fail", "reason": "scientific_or_integrity_gate_failed",
            "counterfactual_count": len(derived), "issues": issues,
            "ordering_direction_success": ordering_success,
            "clustering_direction_success": clustering_success,
        }
        _atomic_write_json(dataset / "reports" / f"counterfactual_{run_id}_audit.json", report)
        return report

    _atomic_write_parquet(run_root / "structures.parquet", derived)
    _atomic_write_parquet(run_root / "sro_labels.parquet", sro)
    _atomic_write_parquet(run_root / "occupancy_sites.parquet", occupancy_sites)
    xrd_root = dataset / "xrd" / "runs" / plan["xrd_run_id"]
    _atomic_write_parquet(xrd_root / "xrd_samples.parquet", xrd_samples)
    files = sorted(
        [path for path in run_root.rglob("*") if path.is_file() and path.name != "checksums.sha256"]
        + [path for path in xrd_root.rglob("*") if path.is_file() and path.name != "checksums.sha256"]
    )
    checksum_lines = [f"{_sha256_file(path)}  {path.relative_to(dataset).as_posix()}" for path in files]
    _atomic_write_text(run_root / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    report = {
        "status": "pass", "run_id": run_id, "parent_count": int(plan["parent_count"]),
        "counterfactual_count": len(derived), "xrd_sample_count": len(xrd_samples),
        "points_per_pattern": int(plan["points_per_pattern"]),
        "ordering_direction_success": ordering_success,
        "clustering_direction_success": clustering_success,
        "low_identifiability_count": int(derived["low_identifiability"].sum()),
    }
    _atomic_write_json(run_root / "run_summary.json", report)
    _atomic_write_json(dataset / "reports" / f"counterfactual_{run_id}_audit.json", report)
    return report


def build_occupancy_benchmark(
    *,
    dataset_dir: str | Path,
    counterfactual_run: str,
    canonical_run: str | None = None,
) -> dict[str, Any]:
    """Build task, pair, identifiability, and leakage-resistant split manifests."""

    dataset = Path(dataset_dir).resolve()
    cf_root = dataset / "derived" / "occupancy" / counterfactual_run
    if not (cf_root / "run_summary.json").exists():
        return {"status": "fail", "reason": "counterfactual_run_not_finalized"}
    canonical_run = canonical_run or _discover_canonical_run(dataset, exclude=f"{counterfactual_run}_xrd")
    if not canonical_run:
        return {"status": "fail", "reason": "canonical_run_missing_or_ambiguous"}
    canonical_root = dataset / "xrd" / "runs" / canonical_run
    canonical_samples_path = canonical_root / "xrd_samples.parquet"
    if not canonical_samples_path.exists():
        return {"status": "fail", "reason": "canonical_run_not_finalized", "canonical_run": canonical_run}

    source_structures = pd.read_parquet(dataset / "manifests" / "structures.parquet")
    canonical_samples = pd.read_parquet(canonical_samples_path)
    canonical = source_structures.merge(
        canonical_samples[
            ["dataset_structure_id", "xrd_sample_id", "grid_shard_path", "grid_row_index", "peak_shard_path"]
        ],
        on="dataset_structure_id",
        how="inner",
    )
    canonical_states = pd.DataFrame(
        {
            "structure_state_id": canonical["dataset_structure_id"],
            "parent_structure_id": canonical["dataset_structure_id"],
            "parent_group_id": canonical["dataset_structure_id"],
            "fidelity": "source_relaxed_dft",
            "strategy": "source",
            "formula": canonical["formula"],
            "chemical_system": canonical["chemical_system"],
            "structure_family": canonical["structure_family"],
            "parent_lattice": canonical["parent_lattice"],
            "structure_sha256": canonical["structure_sha256"],
            "sro_shell1_json": canonical["sro_shell1_json"],
            "sro_shell2_json": canonical["sro_shell2_json"],
            "occupancy_entropy": canonical["occupancy_entropy"],
            "xrd_sample_id": canonical["xrd_sample_id"],
            "grid_shard_path": canonical["grid_shard_path"],
            "grid_row_index": canonical["grid_row_index"],
            "peak_shard_path": canonical["peak_shard_path"],
            "xrd_l1_distance": 0.0,
            "xrd_cosine_similarity": 1.0,
            "low_identifiability": False,
        }
    )
    derived = pd.read_parquet(cf_root / "structures.parquet")
    cf_samples = pd.read_parquet(dataset / "xrd" / "runs" / f"{counterfactual_run}_xrd" / "xrd_samples.parquet")
    cf = derived.merge(
        cf_samples[
            ["dataset_structure_id", "xrd_sample_id", "grid_shard_path", "grid_row_index", "peak_shard_path"]
        ],
        left_on="structure_state_id",
        right_on="dataset_structure_id",
        how="inner",
    )
    derived_states = cf[
        [
            "structure_state_id", "parent_structure_id", "parent_group_id", "fidelity", "strategy",
            "formula", "chemical_system", "structure_family", "parent_lattice", "structure_sha256",
            "sro_shell1_json", "sro_shell2_json", "occupancy_entropy", "xrd_sample_id",
            "grid_shard_path", "grid_row_index", "peak_shard_path", "xrd_l1_distance",
            "xrd_cosine_similarity", "low_identifiability",
        ]
    ]
    states = pd.concat([canonical_states, derived_states], ignore_index=True)
    split_rows = assign_group_splits(states[["structure_state_id", "parent_group_id"]])
    split_rows = split_rows.merge(
        states[["structure_state_id", "formula", "chemical_system", "sro_shell1_json"]],
        on="structure_state_id",
        how="left",
    )
    split_rows["composition_ood_split"] = split_rows.apply(
        lambda row: "test" if _holdout(str(row["formula"])) else (
            "validation" if row["id_split"] == "validation" else "train"
        ), axis=1,
    )
    split_rows["chemistry_ood_split"] = split_rows.apply(
        lambda row: "test" if _holdout(str(row["chemical_system"])) else (
            "validation" if row["id_split"] == "validation" else "train"
        ), axis=1,
    )
    group_sro = split_rows.groupby("parent_group_id", sort=False)["sro_shell1_json"].first().map(_sro_norm)
    low, high = group_sro.quantile([0.15, 0.85]).tolist()
    sro_ood = {
        group: "test" if value <= low or value >= high else (
            "validation" if _id_split(str(group)) == "validation" else "train"
        )
        for group, value in group_sro.items()
    }
    split_rows["sro_ood_split"] = split_rows["parent_group_id"].map(sro_ood)
    split_rows = split_rows.drop(columns=["formula", "chemical_system", "sro_shell1_json"])
    states = states.merge(split_rows, on=["structure_state_id", "parent_group_id"], how="left")
    states["task_names_json"] = states.apply(_task_names, axis=1)

    pairs = derived[
        [
            "structure_state_id", "parent_structure_id", "parent_group_id", "strategy",
            "sro_distance", "xrd_l1_distance", "xrd_cosine_similarity", "low_identifiability",
        ]
    ].copy()
    pairs.insert(0, "pair_id", pairs.apply(
        lambda row: f"pair_{row['parent_structure_id']}_{row['strategy']}", axis=1
    ))
    pairs = pairs.rename(columns={"structure_state_id": "candidate_structure_state_id"})
    pairs = pairs.merge(
        split_rows[["parent_group_id", "id_split"]].drop_duplicates("parent_group_id"),
        on="parent_group_id",
        how="left",
    )
    manifests = dataset / "manifests"
    _atomic_write_parquet(manifests / "occupancy_pairs.parquet", pairs)
    _atomic_write_parquet(manifests / "task_samples.parquet", states)
    _atomic_write_parquet(manifests / "split_assignments.parquet", split_rows)
    composition_features = _composition_only_features(states)
    _atomic_write_parquet(manifests / "composition_only_features.parquet", composition_features)
    report = {
        "status": "pass", "canonical_run": canonical_run,
        "counterfactual_run": counterfactual_run,
        "canonical_structure_count": len(canonical_states),
        "counterfactual_structure_count": len(derived_states),
        "structure_state_count": len(states), "pair_count": len(pairs),
        "composition_only_feature_count": len(composition_features),
        "low_identifiability_pair_count": int(pairs["low_identifiability"].sum()),
        "parent_group_leakage_count": _parent_group_leakage_count(split_rows),
    }
    if report["parent_group_leakage_count"]:
        report["status"] = "fail"
        report["reason"] = "parent_group_split_leakage"
    _atomic_write_json(dataset / "reports" / "occupancy_identifiability.json", report)
    _atomic_write_json(dataset / "reports" / "derived_data_audit.json", report)
    return report


def freeze_observations(
    *,
    dataset_dir: str | Path,
    output_dataset_dir: str | Path | None = None,
    parent_groups_path: str | Path | None = None,
    run_id: str,
    config_path: str | Path,
    variants: int = 5,
    splits: Sequence[str] = ("validation", "test"),
    shard_size: int = 256,
) -> dict[str, Any]:
    """Freeze deterministic experimental views for selected benchmark splits."""

    dataset = Path(dataset_dir).resolve()
    output_dataset = (
        Path(output_dataset_dir).resolve()
        if output_dataset_dir is not None
        else dataset
    )
    if (
        dataset == output_dataset
        and dataset.as_posix() == "/home/GJM/data/hea/benchmark_v1"
    ):
        return {"status": "fail", "reason": "immutable_source_dataset"}
    _validate_run_id(run_id)
    if variants < 1 or shard_size < 1:
        return {"status": "fail", "reason": "variants_and_shard_size_must_be_positive"}
    task_path = dataset / "manifests" / "task_samples.parquet"
    if not task_path.exists():
        return {"status": "fail", "reason": "benchmark_must_be_built_before_observations"}
    config = _load_perturbation_config(Path(config_path))
    task_samples = pd.read_parquet(task_path)
    selected = task_samples[task_samples["id_split"].isin(set(splits))].copy()
    parent_filter_sha256 = None
    selected_parent_group_count = None
    if parent_groups_path is not None:
        filter_path = Path(parent_groups_path).resolve()
        if filter_path.suffix.lower() == ".parquet":
            parent_frame = pd.read_parquet(filter_path)
        elif filter_path.suffix.lower() == ".csv":
            parent_frame = pd.read_csv(filter_path)
        else:
            return {"status": "fail", "reason": "unsupported_parent_group_filter"}
        if "parent_group_id" not in parent_frame:
            return {"status": "fail", "reason": "parent_group_filter_missing_id"}
        parent_ids = set(parent_frame["parent_group_id"].astype(str))
        selected = selected.loc[
            selected["parent_group_id"].astype(str).isin(parent_ids)
        ]
        parent_filter_sha256 = _sha256_file(filter_path)
        selected_parent_group_count = len(parent_ids)
    selected = selected.drop_duplicates("structure_state_id").sort_values("structure_state_id")
    if selected.empty:
        return {"status": "fail", "reason": "no_structure_states_in_requested_splits"}
    run_root = output_dataset / "xrd" / "runs" / run_id
    summary_path = run_root / "run_summary.json"
    requested = {
        "source_task_manifest_sha256": _sha256_file(task_path),
        "config_sha256": _mapping_sha256(config),
        "renderer_version": config["renderer_version"],
        "variants_per_structure": variants,
        "splits": sorted(set(splits)),
        "structure_state_count": len(selected),
        "shard_size": shard_size,
        "source_dataset_root": str(dataset),
        "output_dataset_root": str(output_dataset),
        "parent_group_filter_sha256": parent_filter_sha256,
        "selected_parent_group_count": selected_parent_group_count,
    }
    if summary_path.exists():
        existing = _read_json(summary_path)
        accepted_config_hashes = {
            requested["config_sha256"],
            _sha256_file(Path(config_path).resolve()),
            _sha256_file(run_root / "perturbation_config.json")
            if (run_root / "perturbation_config.json").exists() else "",
        }
        compatible = (
            existing.get("source_task_manifest_sha256") == requested["source_task_manifest_sha256"]
            and existing.get("renderer_version", LEGACY_OBSERVATION_RENDERER_VERSION) == config["renderer_version"]
            and existing.get("config_sha256") in accepted_config_hashes
            and int(existing.get("variants_per_structure", 0)) == variants
            and existing.get("splits") == sorted(set(splits))
            and int(existing.get("structure_state_count", 0)) == len(selected)
            and int(existing.get("shard_count", 0)) == math.ceil(len(selected) / shard_size)
            and int(existing.get("shard_size", shard_size)) == shard_size
        )
        if compatible:
            return {**existing, "action": "existing"}
        return {"status": "fail", "reason": "observation_run_incompatible", "run_id": run_id}
    for name in ("grids", "manifests", "completed"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    config_out = run_root / "perturbation_config.json"
    if config_out.exists() and _mapping_sha256(_read_json(config_out)) != _mapping_sha256(config):
        return {"status": "fail", "reason": "observation_config_incompatible", "run_id": run_id}
    if not config_out.exists():
        _atomic_write_json(config_out, config)
    rows: list[dict[str, Any]] = []
    shard_count = math.ceil(len(selected) / shard_size)
    plan_payload: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "points_per_pattern": 4001,
        "shard_count": shard_count,
        "structure_state_ids": selected["structure_state_id"].astype(str).tolist(),
        **requested,
    }
    plan_payload["plan_sha256"] = _mapping_sha256(plan_payload)
    plan_path = run_root / "observation_plan.json"
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        if existing_plan != plan_payload:
            return {"status": "fail", "reason": "observation_plan_incompatible", "run_id": run_id}
    else:
        _atomic_write_json(plan_path, plan_payload)
    for shard_index in range(shard_count):
        shard = selected.iloc[shard_index * shard_size:(shard_index + 1) * shard_size]
        grid_path = run_root / "grids" / f"grids_{shard_index:04d}.h5"
        manifest_path = run_root / "manifests" / f"samples_{shard_index:04d}.parquet"
        marker_path = run_root / "completed" / f"shard_{shard_index:04d}.json"
        if marker_path.exists():
            valid, issues = _validate_observation_shard(
                marker_path, grid_path, manifest_path, len(shard) * variants
            )
            if not valid:
                return {
                    "status": "fail", "reason": "completed_observation_shard_invalid",
                    "shard_index": shard_index, "issues": issues,
                }
            marker = _read_json(marker_path)
            if marker.get("plan_sha256") != plan_payload["plan_sha256"]:
                marker["plan_sha256"] = plan_payload["plan_sha256"]
                _atomic_write_json(marker_path, marker)
            rows.extend(pd.read_parquet(manifest_path).to_dict(orient="records"))
            continue
        if grid_path.exists() or manifest_path.exists():
            return {
                "status": "fail", "reason": "unmarked_observation_shard_artifacts_exist",
                "shard_index": shard_index,
            }
        grids: list[np.ndarray] = []
        ids: list[str] = []
        shard_rows: list[dict[str, Any]] = []
        axis: np.ndarray | None = None
        peak_cache: dict[str, pd.DataFrame] = {}
        for state in shard.to_dict(orient="records"):
            if axis is None:
                axis = np.round(
                    np.arange(
                        float(state.get("two_theta_min", 20.0)),
                        float(state.get("two_theta_max", 100.0)) + float(state.get("step", 0.02)) / 2.0,
                        float(state.get("step", 0.02)),
                    ),
                    8,
                )
                if len(axis) != 4001:
                    axis = np.linspace(20.0, 100.0, 4001)
            peak_path = str(state["peak_shard_path"])
            if peak_path not in peak_cache:
                peak_cache[peak_path] = pd.read_parquet(dataset / peak_path)
            all_peaks = peak_cache[peak_path]
            peaks = all_peaks[all_peaks["xrd_sample_id"] == state["xrd_sample_id"]]
            if peaks.empty:
                raise ValueError(f"No indexed peaks for {state['xrd_sample_id']}")
            for variant_index in range(variants):
                params = sample_observation_parameters(
                    str(state["structure_sha256"]), variant_index, config
                )
                observation = render_observation(axis, peaks, params)
                observation_id = f"obs_{state['structure_state_id']}_{variant_index:02d}"
                row_index = len(grids)
                grids.append(observation)
                ids.append(observation_id)
                shard_rows.append(
                    {
                        "xrd_sample_id": observation_id,
                        "structure_state_id": state["structure_state_id"],
                        "parent_group_id": state["parent_group_id"],
                        "source_xrd_sample_id": state["xrd_sample_id"],
                        "variant": f"observation_{variant_index:02d}",
                        "id_split": state["id_split"],
                        "grid_shard_path": f"xrd/runs/{run_id}/grids/grids_{shard_index:04d}.h5",
                        "grid_row_index": row_index,
                        "peak_shard_path": peak_path,
                        "perturbation_config_id": config["config_id"],
                        "perturbation_json": json.dumps(params, sort_keys=True),
                    }
                )
        _atomic_write_h5(grid_path, axis, grids, ids)
        _atomic_write_parquet(manifest_path, pd.DataFrame(shard_rows))
        marker = {
            "schema_version": 1, "shard_index": shard_index, "sample_count": len(shard_rows),
            "plan_sha256": plan_payload["plan_sha256"],
            "grid_sha256": _sha256_file(grid_path), "manifest_sha256": _sha256_file(manifest_path),
        }
        _atomic_write_json(marker_path, marker)
        rows.extend(shard_rows)
    observations = pd.DataFrame(rows).sort_values("xrd_sample_id", kind="stable")
    expected_observations = len(selected) * variants
    if (
        len(observations) != expected_observations
        or observations["xrd_sample_id"].nunique() != expected_observations
    ):
        return {"status": "fail", "reason": "observation_count_or_uniqueness_invalid"}
    if observations.groupby("parent_group_id")["id_split"].nunique().gt(1).any():
        return {"status": "fail", "reason": "observation_parent_group_split_leakage"}
    _atomic_write_parquet(run_root / "xrd_samples.parquet", observations)
    checksummed = sorted(path for path in run_root.rglob("*") if path.is_file() and path.name not in {
        "checksums.sha256", "run_summary.json"
    })
    _atomic_write_text(
        run_root / "checksums.sha256",
        "\n".join(f"{_sha256_file(path)}  {path.relative_to(run_root).as_posix()}" for path in checksummed) + "\n",
    )
    report = {
        "status": "pass", "run_id": run_id, "observation_count": len(observations),
        "structure_state_count": len(selected), "variants_per_structure": variants,
        "shard_count": shard_count, "points_per_pattern": 4001,
        "plan_sha256": plan_payload["plan_sha256"], **requested,
    }
    _atomic_write_json(summary_path, report)
    return report


def audit_observation_run(*, dataset_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Independently audit a finalized frozen-observation run."""

    dataset = Path(dataset_dir).resolve()
    run_root = dataset / "xrd" / "runs" / run_id
    summary_path = run_root / "run_summary.json"
    manifest_path = run_root / "xrd_samples.parquet"
    issues: list[str] = []
    if not summary_path.exists() or not manifest_path.exists():
        return {
            "status": "fail", "run_id": run_id, "observation_count": 0,
            "issues": ["missing_summary_or_merged_manifest"],
        }
    summary = _read_json(summary_path)
    observations = pd.read_parquet(manifest_path)
    expected_count = int(summary.get("observation_count", 0))
    if len(observations) != expected_count or observations["xrd_sample_id"].nunique() != expected_count:
        issues.append("observation_count_or_uniqueness_mismatch")
    variants = int(summary.get("variants_per_structure", 0))
    if variants < 1 or not observations.groupby("structure_state_id").size().eq(variants).all():
        issues.append("variants_per_structure_mismatch")
    if observations.groupby("parent_group_id")["id_split"].nunique().gt(1).any():
        issues.append("parent_group_split_leakage")
    try:
        if not observations["perturbation_json"].map(lambda value: bool(json.loads(value))).all():
            issues.append("empty_perturbation_parameters")
    except (TypeError, ValueError, json.JSONDecodeError):
        issues.append("invalid_perturbation_json")

    shard_count = int(summary.get("shard_count", 0))
    for shard_index in range(shard_count):
        relative = f"xrd/runs/{run_id}/grids/grids_{shard_index:04d}.h5"
        shard_rows = observations[observations["grid_shard_path"] == relative]
        marker_path = run_root / "completed" / f"shard_{shard_index:04d}.json"
        grid_path = run_root / "grids" / f"grids_{shard_index:04d}.h5"
        fragment_path = run_root / "manifests" / f"samples_{shard_index:04d}.parquet"
        if not marker_path.exists():
            issues.append(f"shard_{shard_index:04d}:missing_completion_marker")
            continue
        valid, shard_issues = _validate_observation_shard(
            marker_path, grid_path, fragment_path, len(shard_rows)
        )
        if not valid:
            issues.extend(f"shard_{shard_index:04d}:{issue}" for issue in shard_issues)
    temporary_files = [path for path in run_root.rglob("*.tmp") if path.is_file()]
    if temporary_files:
        issues.append(f"temporary_files_present:{len(temporary_files)}")
    checksum_path = run_root / "checksums.sha256"
    if not checksum_path.exists():
        issues.append("missing_checksums")
    else:
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                issues.append("invalid_checksum_line")
                continue
            digest, relative = parts
            path = (run_root / relative).resolve()
            try:
                path.relative_to(run_root.resolve())
            except ValueError:
                issues.append(f"checksum_path_outside_run:{relative}")
                continue
            if not path.exists():
                issues.append(f"checksum_path_missing:{relative}")
            elif _sha256_file(path) != digest:
                issues.append(f"checksum_mismatch:{relative}")
    report = {
        "status": "pass" if not issues else "fail", "run_id": run_id,
        "observation_count": len(observations), "structure_state_count": observations["structure_state_id"].nunique(),
        "variants_per_structure": variants, "shard_count": shard_count,
        "points_per_pattern": int(summary.get("points_per_pattern", 0)),
        "run_integrity": not issues, "issues": issues,
    }
    _atomic_write_json(dataset / "reports" / f"xrd_observation_{run_id}_audit.json", report)
    return report


def _validate_observation_shard(
    marker_path: Path, grid_path: Path, manifest_path: Path, expected_count: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    marker = _read_json(marker_path)
    if not grid_path.exists():
        issues.append("missing_grid")
    elif _sha256_file(grid_path) != marker.get("grid_sha256"):
        issues.append("grid_checksum_mismatch")
    if not manifest_path.exists():
        issues.append("missing_manifest")
    elif _sha256_file(manifest_path) != marker.get("manifest_sha256"):
        issues.append("manifest_checksum_mismatch")
    if issues:
        return False, issues
    manifest = pd.read_parquet(manifest_path)
    if len(manifest) != expected_count or manifest["xrd_sample_id"].nunique() != expected_count:
        issues.append("manifest_count_or_uniqueness_mismatch")
    try:
        with h5py.File(grid_path, "r") as handle:
            values = handle["intensity"][:]
            ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["xrd_sample_id"][:]]
            if values.shape != (expected_count, 4001):
                issues.append("grid_shape_mismatch")
            elif not np.isfinite(values).all() or float(values.min()) < 0:
                issues.append("invalid_grid_values")
            elif not np.allclose(values.max(axis=1), 100.0, atol=1e-4):
                issues.append("invalid_grid_normalization")
            if ids != manifest["xrd_sample_id"].astype(str).tolist():
                issues.append("grid_manifest_id_mismatch")
            if len(handle["two_theta"]) != 4001:
                issues.append("two_theta_width_mismatch")
    except (OSError, KeyError, ValueError) as exc:
        issues.append(f"invalid_hdf5:{type(exc).__name__}")
    return not issues, issues


def _eligible_parents(structures: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    mapping_limit = float(config.get("max_template_mapping_rms", 0.25))
    min_elements = int(config.get("min_elements", 3))
    min_natoms = int(config.get("min_natoms_for_sro", 27))
    duplicate_free = structures["duplicate_of"].isna() | structures["duplicate_of"].astype(str).isin({"", "nan"})
    mapping = pd.to_numeric(structures["template_mapping_rms"], errors="coerce").fillna(np.inf)
    eligible = structures[
        structures["xrd_eligible"].astype(bool)
        & structures["quality_tier"].eq("pass")
        & structures["structure_family"].isin({"bcc_like", "fcc_like"})
        & structures["parent_lattice"].isin({"bcc", "fcc"})
        & (pd.to_numeric(structures["nelements"], errors="coerce") >= min_elements)
        & (pd.to_numeric(structures["natoms"], errors="coerce") >= min_natoms)
        & duplicate_free
        & (mapping <= mapping_limit)
    ].copy()
    return eligible


def _select_balanced_parents(eligible: pd.DataFrame, requested: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    base_quota = requested // 2
    quotas = {"bcc": base_quota, "fcc": base_quota}
    if requested % 2:
        larger = max(("bcc", "fcc"), key=lambda family: len(eligible[eligible["parent_lattice"] == family]))
        quotas[larger] += 1
    leftovers: list[pd.DataFrame] = []
    for family in ("bcc", "fcc"):
        frame = eligible[eligible["parent_lattice"] == family].copy()
        frame["_sro_metric"] = frame["sro_shell1_json"].map(_sro_norm)
        unique_metrics = frame["_sro_metric"].nunique()
        frame["_sro_bin"] = (
            pd.qcut(frame["_sro_metric"], q=min(5, unique_metrics), labels=False, duplicates="drop")
            if unique_metrics > 1 else 0
        )
        frame["_rank"] = frame["structure_sha256"].astype(str).map(
            lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
        )
        frame = frame.sort_values(
            ["chemical_system", "nelements", "natoms", "_rank"], kind="stable"
        )
        groups = [
            group for _, group in frame.groupby(
                ["chemical_system", "nelements", "natoms", "_sro_bin"], sort=True, dropna=False
            )
        ]
        ordered_rows = []
        while groups:
            next_groups = []
            for group in groups:
                ordered_rows.append(group.iloc[0])
                if len(group) > 1:
                    next_groups.append(group.iloc[1:])
            groups = next_groups
        ordered = pd.DataFrame(ordered_rows).drop(
            columns=["_rank", "_sro_metric", "_sro_bin"], errors="ignore"
        )
        selected.append(ordered.head(quotas[family]))
        leftovers.append(ordered.iloc[quotas[family]:])
    result = pd.concat(selected, ignore_index=True)
    if len(result) < requested:
        extra = pd.concat(leftovers, ignore_index=True).head(requested - len(result))
        result = pd.concat([result, extra], ignore_index=True)
    return result.sort_values("dataset_structure_id", kind="stable").head(requested).reset_index(drop=True)


def _load_counterfactual_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.resolve())
    required = {
        "config_id", "strategies", "candidates_per_strategy", "local_swap_fraction",
        "monte_carlo_steps", "min_elements", "max_template_mapping_rms", "xrd_config",
        "min_natoms_for_sro",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing counterfactual config keys: {missing}")
    if tuple(config["strategies"]) != COUNTERFACTUAL_STRATEGIES:
        raise ValueError(f"v1 strategies must be {list(COUNTERFACTUAL_STRATEGIES)}")
    if _config_points(config["xrd_config"]) != 4001:
        raise ValueError("Counterfactual XRD config must contain 4001 points")
    return config


def _load_counterfactual_plan(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(run_root / "run_plan.json")
    expected = plan.pop("plan_sha256", None)
    actual = _mapping_sha256(plan)
    plan["plan_sha256"] = expected
    if expected != actual:
        raise ValueError("counterfactual_run_plan_checksum_mismatch")
    config_path = run_root / "counterfactual_config.json"
    payload_path = run_root / "structure_payloads.parquet"
    if _sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("counterfactual_config_checksum_mismatch")
    if _sha256_file(payload_path) != plan["source_payload_sha256"]:
        raise ValueError("counterfactual_source_payload_checksum_mismatch")
    if _code_sha256(Path(__file__).resolve().parents[1]) != plan["code_sha256"]:
        raise ValueError("counterfactual_code_checksum_mismatch")
    return plan, _load_counterfactual_config(config_path)


def _validate_counterfactual_shard(
    dataset: Path, run_root: Path, plan: Mapping[str, Any], shard_index: int,
) -> tuple[bool, list[str]]:
    marker_path = run_root / "completed" / f"shard_{shard_index:04d}.json"
    if not marker_path.exists():
        return False, ["missing_completion_marker"]
    marker = _read_json(marker_path)
    issues: list[str] = []
    if marker.get("plan_sha256") != plan["plan_sha256"]:
        issues.append("plan_checksum_mismatch")
    checksums = marker.get("checksums", {})
    for relative, digest in checksums.items():
        path = dataset / relative
        try:
            path.resolve().relative_to(dataset.resolve())
        except ValueError:
            issues.append(f"artifact_outside_dataset:{relative}")
            continue
        if not path.exists():
            issues.append(f"missing_artifact:{relative}")
        elif _sha256_file(path) != digest:
            issues.append(f"checksum_mismatch:{relative}")
    if len(checksums) != 6:
        issues.append("artifact_count_mismatch")
    expected = int(plan["shards"][shard_index]["count"]) * int(plan["variants_per_parent"])
    structure_path = run_root / "structures" / f"structures_{shard_index:04d}.parquet"
    xrd_root = dataset / "xrd" / "runs" / plan["xrd_run_id"]
    sample_path = xrd_root / "manifests" / f"samples_{shard_index:04d}.parquet"
    grid_path = xrd_root / "grids" / f"grids_{shard_index:04d}.h5"
    if not issues:
        structures = pd.read_parquet(structure_path)
        samples = pd.read_parquet(sample_path)
        if len(structures) != expected or structures["structure_state_id"].nunique() != expected:
            issues.append("structure_count_or_uniqueness_mismatch")
        if len(samples) != expected or samples["dataset_structure_id"].nunique() != expected:
            issues.append("xrd_sample_count_or_uniqueness_mismatch")
        try:
            with h5py.File(grid_path, "r") as handle:
                values = handle["intensity"][:]
                if values.shape != (expected, int(plan["points_per_pattern"])):
                    issues.append("grid_shape_mismatch")
                elif not np.isfinite(values).all() or float(values.min()) < 0:
                    issues.append("invalid_grid_values")
                elif not np.allclose(values.max(axis=1), 100.0, atol=1e-4):
                    issues.append("invalid_grid_normalization")
        except (OSError, KeyError, ValueError) as exc:
            issues.append(f"invalid_hdf5:{type(exc).__name__}")
    return not issues, issues


def _load_perturbation_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.resolve())
    config.setdefault("renderer_version", OBSERVATION_RENDERER_VERSION)
    _validate_renderer_version(config["renderer_version"])
    ranges = (
        "zero_shift_deg", "instrument_fwhm_deg", "microstrain_fraction",
        "crystallite_size_nm", "background_fraction", "gaussian_noise_fraction",
        "poisson_scale", "march_dollase_r",
    )
    missing = sorted({"config_id", "texture_axes", *ranges} - set(config))
    if missing:
        raise ValueError(f"Missing perturbation config keys: {missing}")
    for key in ranges:
        value = config[key]
        if not isinstance(value, list) or len(value) != 2 or float(value[1]) < float(value[0]):
            raise ValueError(f"Invalid perturbation range: {key}")
    if not config["texture_axes"]:
        raise ValueError("texture_axes cannot be empty")
    return config


def sample_observation_parameters(
    structure_hash: str, variant_index: int, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one reproducible online or frozen observation parameter set."""

    renderer_version = config.get("renderer_version", OBSERVATION_RENDERER_VERSION)
    _validate_renderer_version(renderer_version)
    seed = int(hashlib.sha256(
        f"{config['config_id']}:{structure_hash}:{variant_index}".encode("utf-8")
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    def sample(name: str) -> float:
        low, high = (float(value) for value in config[name])
        return float(rng.uniform(low, high))

    axes = list(config["texture_axes"])
    return {
        "renderer_version": renderer_version,
        "zero_shift_deg": sample("zero_shift_deg"),
        "instrument_fwhm_deg": sample("instrument_fwhm_deg"),
        "microstrain_fraction": sample("microstrain_fraction"),
        "crystallite_size_nm": sample("crystallite_size_nm"),
        "background_fraction": sample("background_fraction"),
        "background_coefficients": [1.0, float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.2, 0.2))],
        "gaussian_noise_fraction": sample("gaussian_noise_fraction"),
        "poisson_scale": sample("poisson_scale"),
        "march_dollase_r": sample("march_dollase_r"),
        "texture_axis": list(axes[int(rng.integers(0, len(axes)))]),
        "wavelength_angstrom": 1.5406,
        "seed": seed,
    }


def _discover_canonical_run(dataset: Path, *, exclude: str) -> str | None:
    candidates = []
    for summary_path in sorted((dataset / "xrd" / "runs").glob("*/run_summary.json")):
        run_id = summary_path.parent.name
        if run_id == exclude:
            continue
        summary = _read_json(summary_path)
        if bool(summary.get("clean_only", False)) and (summary_path.parent / "xrd_samples.parquet").exists():
            candidates.append(run_id)
    return candidates[0] if len(candidates) == 1 else None


def _task_names(row: pd.Series) -> str:
    tasks = ["family_classification", "robust_inversion"]
    if str(row["parent_lattice"]) in {"bcc", "fcc"}:
        tasks.append("sro_regression")
    if str(row["strategy"]) != "source":
        tasks.extend(["occupancy_pair_ranking", "occupancy_retrieval", "occupancy_identifiability"])
    return json.dumps(tasks)


def _sro_norm(value: str) -> float:
    try:
        values = [float(item) for item in json.loads(value).values()]
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return float(np.linalg.norm(values)) if values else 0.0


def _holdout(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10_000 >= 8_500


def _parent_group_leakage_count(splits: pd.DataFrame) -> int:
    counts = splits.groupby("parent_group_id")["id_split"].nunique()
    return int((counts > 1).sum())


def _composition_only_features(states: pd.DataFrame) -> pd.DataFrame:
    compositions: list[dict[str, float]] = []
    elements: set[str] = set()
    for formula in states["formula"].astype(str):
        try:
            fractions = {
                str(element): float(amount)
                for element, amount in Composition(formula).fractional_composition.items()
            }
        except (TypeError, ValueError):
            fractions = {}
        compositions.append(fractions)
        elements.update(fractions)
    rows = []
    for state, fractions in zip(states.to_dict(orient="records"), compositions):
        row = {
            "structure_state_id": state["structure_state_id"],
            "parent_group_id": state["parent_group_id"],
            "structure_family": state["structure_family"],
            "parent_lattice": state["parent_lattice"],
            "id_split": state["id_split"],
            "occupancy_entropy": state["occupancy_entropy"],
            "sro_shell1_json": state["sro_shell1_json"],
        }
        row.update({f"fraction_{element}": fractions.get(element, 0.0) for element in sorted(elements)})
        rows.append(row)
    return pd.DataFrame(rows)


def generate_counterfactual_variants(
    structure: Structure,
    *,
    parent_structure_id: str,
    parent_lattice: str,
    parent_hash: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Generate deterministic, composition-preserving occupancy variants."""

    strategies = tuple(config.get("strategies", COUNTERFACTUAL_STRATEGIES))
    unsupported = sorted(set(strategies) - set(COUNTERFACTUAL_STRATEGIES))
    if unsupported:
        raise ValueError(f"Unsupported counterfactual strategies: {unsupported}")
    if parent_lattice not in {"bcc", "fcc"}:
        raise ValueError("Counterfactual parent lattice must be bcc or fcc")
    if len(structure.composition.elements) < 2:
        raise ValueError("Counterfactual generation requires at least two elements")

    symbols = np.asarray([site.specie.symbol for site in structure], dtype=object)
    adjacency = _first_shell_pairs(structure, parent_lattice)
    before_sro1, before_sro2 = _sro(structure, parent_lattice)
    forbidden = {tuple(symbols.tolist())}
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        seed = _seed(parent_hash, strategy)
        candidate_symbols = _choose_candidate(
            symbols,
            adjacency,
            strategy=strategy,
            seed=seed,
            candidates=int(config.get("candidates_per_strategy", 16)),
            local_swap_fraction=float(config.get("local_swap_fraction", 0.05)),
            monte_carlo_steps=int(config.get("monte_carlo_steps", 256)),
            forbidden=forbidden,
        )
        forbidden.add(tuple(candidate_symbols.tolist()))
        derived = structure.copy()
        for index, symbol in enumerate(candidate_symbols):
            if symbol != symbols[index]:
                derived.replace(index, str(symbol))
        after_sro1, after_sro2 = _sro(derived, parent_lattice)
        structure_hash = canonical_structure_hash(derived.as_dict())
        variant_id = f"cf_{parent_structure_id}_{strategy}"
        rows.append(
            {
                "structure_state_id": variant_id,
                "dataset_structure_id": variant_id,
                "parent_structure_id": parent_structure_id,
                "parent_group_id": parent_structure_id,
                "strategy": strategy,
                "seed": seed,
                "fidelity": "fixed_lattice_counterfactual",
                "parent_lattice": parent_lattice,
                "structure_family": f"{parent_lattice}_like",
                "structure_sha256": structure_hash,
                "structure_json": json.dumps(derived.as_dict(), sort_keys=True),
                "site_elements_json": json.dumps(candidate_symbols.tolist()),
                "swapped_site_count": int(np.count_nonzero(candidate_symbols != symbols)),
                "composition_preserved": derived.composition == structure.composition,
                "template_mapping_rms": _parent_mapping_rms(derived, parent_lattice),
                "occupancy_entropy": _composition_entropy(candidate_symbols),
                "occupancy_fingerprint": hashlib.sha256(
                    json.dumps(candidate_symbols.tolist()).encode("utf-8")
                ).hexdigest(),
                "sro_shell1_before_json": json.dumps(before_sro1, sort_keys=True),
                "sro_shell2_before_json": json.dumps(before_sro2, sort_keys=True),
                "sro_shell1_json": json.dumps(after_sro1, sort_keys=True),
                "sro_shell2_json": json.dumps(after_sro2, sort_keys=True),
                "sro_distance": _sro_distance(before_sro1, after_sro1),
                "same_neighbor_fraction_before": _same_neighbor_fraction(symbols, adjacency),
                "same_neighbor_fraction_after": _same_neighbor_fraction(candidate_symbols, adjacency),
            }
        )
    return rows


def _validate_renderer_version(version: str) -> None:
    if version not in {OBSERVATION_RENDERER_VERSION, LEGACY_OBSERVATION_RENDERER_VERSION}:
        raise ValueError(f"Unknown observation renderer version: {version}")


def render_observation(
    two_theta: Sequence[float] | np.ndarray,
    peaks: pd.DataFrame,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Render integrated reflection weights, then normalize the whole observation.

    Gaussian areas are normalized analytically, not over the visible grid: tails
    outside the measurement window stay truncated. Use peak_height_v1 explicitly
    only to replay the historical, non-area-preserving perturbation protocol.
    """

    axis = np.asarray(two_theta, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("two_theta must be a finite increasing one-dimensional grid")
    renderer_version = params.get("renderer_version", OBSERVATION_RENDERER_VERSION)
    _validate_renderer_version(renderer_version)
    values = np.zeros_like(axis)
    shift = float(params.get("zero_shift_deg", 0.0))
    instrument = float(params.get("instrument_fwhm_deg", 0.1))
    microstrain = float(params.get("microstrain_fraction", 0.0))
    crystallite_nm = params.get("crystallite_size_nm")
    texture_r = float(params.get("march_dollase_r", 1.0))
    texture_axis = np.asarray(params.get("texture_axis", [1, 0, 0]), dtype=float)
    wavelength_nm = float(params.get("wavelength_angstrom", 1.5406)) * 0.1
    if not np.isfinite([shift, instrument, microstrain, wavelength_nm]).all():
        raise ValueError("Shift, widths and wavelength must be finite")
    if instrument < 0 or microstrain < 0 or wavelength_nm <= 0:
        raise ValueError("Widths must be non-negative and wavelength positive")
    if crystallite_nm is not None and (not np.isfinite(float(crystallite_nm)) or float(crystallite_nm) <= 0):
        raise ValueError("Crystallite size must be positive and finite, or None")

    for peak in peaks.to_dict(orient="records"):
        center = float(peak["two_theta"]) + shift
        intensity = float(peak["intensity"])
        if not np.isfinite([center, intensity]).all() or intensity < 0:
            raise ValueError("Reflection positions and non-negative weights must be finite")
        theta = math.radians(float(peak["two_theta"]) / 2.0)
        strain_width = math.degrees(4.0 * microstrain * math.tan(theta))
        size_width = 0.0
        if crystallite_nm is not None and float(crystallite_nm) > 0:
            size_width = math.degrees(
                0.9 * wavelength_nm / (float(crystallite_nm) * max(math.cos(theta), 1e-8))
            )
        fwhm = math.sqrt(instrument**2 + strain_width**2 + size_width**2)
        sigma = max(fwhm, 1e-6) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        texture = _texture_factor(str(peak.get("hkls_json", "[]")), texture_axis, texture_r)
        profile = np.exp(
            -0.5 * ((axis - center) / sigma) ** 2
        )
        if renderer_version == OBSERVATION_RENDERER_VERSION:
            profile /= sigma * math.sqrt(2.0 * math.pi)
        values += intensity * texture * profile

    peak_max = float(values.max()) if len(values) else 0.0
    if peak_max > 0:
        values = values / peak_max * 100.0
    background_fraction = float(params.get("background_fraction", 0.0))
    coefficients = list(params.get("background_coefficients", [1.0, 0.0, 0.0]))
    while len(coefficients) < 3:
        coefficients.append(0.0)
    x = np.linspace(-1.0, 1.0, len(axis))
    background_shape = np.clip(
        float(coefficients[0]) + float(coefficients[1]) * x + float(coefficients[2]) * x**2,
        0.0,
        None,
    )
    if float(background_shape.max()) > 0:
        background_shape /= float(background_shape.max())
    values += background_fraction * 100.0 * background_shape

    rng = np.random.default_rng(int(params.get("seed", 0)))
    poisson_scale = float(params.get("poisson_scale", 0.0))
    if poisson_scale > 0:
        values = rng.poisson(np.clip(values, 0.0, None) * poisson_scale) / poisson_scale
    noise = float(params.get("gaussian_noise_fraction", 0.0))
    if noise > 0:
        values += rng.normal(0.0, noise * 100.0, len(values))
    values = np.clip(values, 0.0, None)
    if float(values.max()) > 0:
        values = values / float(values.max()) * 100.0
    return values.astype(np.float32)


def assign_group_splits(states: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic 70/15/15 ID splits at parent-group granularity."""

    required = {"structure_state_id", "parent_group_id"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"Missing split columns: {missing}")
    result = states.copy()
    result["id_split"] = result["parent_group_id"].astype(str).map(_id_split)
    return result


def _choose_candidate(
    symbols: np.ndarray,
    adjacency: np.ndarray,
    *,
    strategy: str,
    seed: int,
    candidates: int,
    local_swap_fraction: float,
    monte_carlo_steps: int,
    forbidden: set[tuple[Any, ...]],
) -> np.ndarray:
    if candidates < 1 or monte_carlo_steps < 1:
        raise ValueError("Candidate and Monte Carlo counts must be positive")
    generated: list[tuple[float, tuple[str, ...], np.ndarray]] = []
    for candidate_index in range(candidates):
        rng = np.random.default_rng(seed + candidate_index)
        candidate = symbols.copy()
        if strategy == "local_swap":
            pair_count = max(1, int(round(len(symbols) * local_swap_fraction / 2.0)))
            for _ in range(pair_count):
                _random_unlike_swap(candidate, rng)
            score = float(np.count_nonzero(candidate != symbols))
        elif strategy == "randomized":
            candidate = rng.permutation(candidate)
            score = abs(_same_neighbor_fraction(candidate, adjacency) - _random_same_fraction(symbols))
        else:
            minimize = strategy == "ordering_enhanced"
            current_score = _same_neighbor_fraction(candidate, adjacency)
            for _ in range(monte_carlo_steps):
                proposal = candidate.copy()
                _random_unlike_swap(proposal, rng)
                proposal_score = _same_neighbor_fraction(proposal, adjacency)
                if (minimize and proposal_score < current_score) or (
                    not minimize and proposal_score > current_score
                ):
                    candidate, current_score = proposal, proposal_score
            score = current_score if minimize else -current_score
        key = tuple(str(value) for value in candidate.tolist())
        if key not in forbidden and key != tuple(str(value) for value in symbols.tolist()):
            generated.append((score, key, candidate))
    if generated:
        generated.sort(key=lambda item: (item[0], item[1]))
        return generated[0][2]

    rng = np.random.default_rng(seed + 100_000)
    for _ in range(10_000):
        candidate = rng.permutation(symbols)
        key = tuple(str(value) for value in candidate.tolist())
        if key not in forbidden and key != tuple(str(value) for value in symbols.tolist()):
            return candidate
    raise ValueError("Could not generate a unique composition-preserving occupancy variant")


def _random_unlike_swap(symbols: np.ndarray, rng: np.random.Generator) -> None:
    if len(set(symbols.tolist())) < 2:
        raise ValueError("Cannot swap a single-element structure")
    first = int(rng.integers(0, len(symbols)))
    choices = np.flatnonzero(symbols != symbols[first])
    second = int(choices[int(rng.integers(0, len(choices)))])
    symbols[first], symbols[second] = symbols[second], symbols[first]


def _first_shell_pairs(structure: Structure, parent_lattice: str) -> np.ndarray:
    volume_per_atom = structure.volume / len(structure)
    if parent_lattice == "fcc":
        first = (4.0 * volume_per_atom) ** (1.0 / 3.0) / math.sqrt(2.0)
        second = first * math.sqrt(2.0)
    else:
        first = math.sqrt(3.0) * (2.0 * volume_per_atom) ** (1.0 / 3.0) / 2.0
        second = (2.0 * volume_per_atom) ** (1.0 / 3.0)
    centers, neighbors, _, _ = structure.get_neighbor_list((first + second) / 2.0)
    pairs = {
        (min(int(center), int(neighbor)), max(int(center), int(neighbor)))
        for center, neighbor in zip(centers, neighbors)
        if int(center) != int(neighbor)
    }
    if not pairs:
        raise ValueError("Parent structure has no first-shell neighbor pairs")
    return np.asarray(sorted(pairs), dtype=int)


def _same_neighbor_fraction(symbols: np.ndarray, adjacency: np.ndarray) -> float:
    return float(np.mean(symbols[adjacency[:, 0]] == symbols[adjacency[:, 1]]))


def _random_same_fraction(symbols: np.ndarray) -> float:
    _, counts = np.unique(symbols, return_counts=True)
    fractions = counts / len(symbols)
    return float(np.sum(fractions**2))


def _composition_entropy(symbols: np.ndarray) -> float:
    _, counts = np.unique(symbols, return_counts=True)
    fractions = counts / len(symbols)
    return float(-np.sum(fractions * np.log(fractions)))


def _sro_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    return float(math.sqrt(sum((float(left.get(key, 0.0)) - float(right.get(key, 0.0))) ** 2 for key in keys)))


def _texture_factor(hkls_json: str, axis: np.ndarray, r: float) -> float:
    if math.isclose(r, 1.0) or float(np.linalg.norm(axis)) == 0:
        return 1.0
    try:
        reflections = json.loads(hkls_json)
        hkl = np.asarray(reflections[0]["hkl"], dtype=float)
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return 1.0
    if len(hkl) == 4:
        h, k, _, l = hkl
        hkl = np.asarray([h + 0.5 * k, math.sqrt(3.0) * 0.5 * k, l], dtype=float)
    elif len(hkl) != 3:
        return 1.0
    if float(np.linalg.norm(hkl)) == 0:
        return 1.0
    cosine = float(np.dot(hkl, axis) / (np.linalg.norm(hkl) * np.linalg.norm(axis)))
    cosine_sq = min(1.0, max(0.0, cosine * cosine))
    return float((r * r * cosine_sq + (1.0 / r) * (1.0 - cosine_sq)) ** -1.5)


def _seed(parent_hash: str, strategy: str) -> int:
    return int(hashlib.sha256(f"{parent_hash}:{strategy}".encode("utf-8")).hexdigest()[:8], 16)


def _id_split(parent_group_id: str) -> str:
    rank = int(hashlib.sha256(parent_group_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    if rank < 7_000:
        return "train"
    if rank < 8_500:
        return "validation"
    return "test"
