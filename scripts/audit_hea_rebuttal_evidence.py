"""Read-only audit of released HEA spectra and identifiability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values) -> dict:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "quantiles": {
            str(q): float(np.quantile(finite, q))
            for q in (0, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        } if finite.size else {},
    }


def load_grids(root: Path, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    rows = frame.reset_index(drop=True)
    arrays = np.empty((len(rows), 4001), dtype=np.float32)
    paths = []
    for relative, indices in rows.groupby("grid_shard_path", sort=True).groups.items():
        positions = np.asarray(indices, dtype=int)
        selected = rows.iloc[positions]
        grid_rows = selected["grid_row_index"].to_numpy(dtype=int)
        path = root / relative
        with h5py.File(path, "r") as handle:
            unique, inverse = np.unique(grid_rows, return_inverse=True)
            values = handle["intensity"][unique][inverse]
            ids = handle["xrd_sample_id"].asstr()[unique][inverse]
            if not np.array_equal(ids, selected["xrd_sample_id"].to_numpy()):
                raise ValueError(f"HDF5 sample-ID mismatch in {path}")
            if values.shape != (len(positions), 4001) or not np.isfinite(values).all():
                raise ValueError(f"Invalid spectral arrays in {path}")
            axis = np.asarray(handle["two_theta"], dtype=float)
            if not np.allclose(axis, np.linspace(20, 100, 4001), atol=4e-6, rtol=0):
                raise ValueError(f"Unexpected angular grid in {path}")
            arrays[positions] = values
        paths.append({"path": str(path), "sha256": sha256(path)})
    return arrays, {"sample_count": len(rows), "shard_count": len(paths), "shards": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    source = root / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    release = source.with_name("benchmark_v1_1")
    if output == root / "data" or root / "data" in output.parents:
        raise ValueError("Audit outputs must be outside source data")
    evidence = []

    def read_frame(path: Path) -> pd.DataFrame:
        evidence.append({"path": str(path), "sha256": sha256(path)})
        return pd.read_parquet(path)

    def read_json(path: Path) -> dict:
        evidence.append({"path": str(path), "sha256": sha256(path)})
        return json.loads(path.read_text(encoding="utf-8"))

    clean_run = source / "xrd/runs/hea_relaxed_clean_xrd_v1"
    observed_run = release / "xrd/runs/hea_observation_v1_1"
    clean_config = read_json(clean_run / "xrd_config.json")
    clean_plan = read_json(clean_run / "run_plan.json")
    observation_config = read_json(observed_run / "perturbation_config.json")
    observation_plan = read_json(observed_run / "observation_plan.json")
    config_checks = {
        "clean_config_matches_plan": sha256(clean_run / "xrd_config.json") == clean_plan["config_sha256"],
        "observation_config_matches_plan": hashlib.sha256(json.dumps(
            observation_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()).hexdigest() == observation_plan["config_sha256"],
    }
    pairs = read_frame(release / "manifests/occupancy_pairs_calibrated.parquet")
    tasks = read_frame(source / "manifests/task_samples.parquet")
    observations = read_frame(release / "xrd/runs/hea_observation_v1_1/xrd_samples.parquet")
    stored_distances = read_frame(release / "reports/observation_noise_distances.parquet")
    stored_floors = read_frame(release / "reports/parent_noise_floors.parquet")
    if pairs["pair_id"].duplicated().any() or tasks["structure_state_id"].duplicated().any():
        raise ValueError("Duplicate pair or structure-state IDs")
    if observations["xrd_sample_id"].duplicated().any():
        raise ValueError("Duplicate observation IDs")
    task_index = tasks.set_index("structure_state_id", drop=False)
    state_ids = sorted(set(observations["structure_state_id"]))
    clean_rows = task_index.loc[state_ids].reset_index(drop=True)
    clean_lookup = {state: i for i, state in enumerate(state_ids)}
    expected = task_index.loc[observations["structure_state_id"]]
    if not np.array_equal(expected["xrd_sample_id"], observations["source_xrd_sample_id"]):
        raise ValueError("Observation source does not match canonical task manifest")
    for column in ("parent_group_id", "id_split"):
        if not np.array_equal(expected[column], observations[column]):
            raise ValueError(f"Observation {column} differs from canonical task manifest")
    print(f"Reading {len(clean_rows)} clean and {len(observations)} observed spectra", flush=True)
    clean, clean_files = load_grids(source, clean_rows)
    observed, observed_files = load_grids(release, observations)
    parent_idx = np.asarray([clean_lookup[s] for s in pairs["parent_structure_id"]])
    candidate_idx = np.asarray([clean_lookup[s] for s in pairs["candidate_structure_state_id"]])

    pair_l1 = np.empty(len(pairs))
    noise_l1 = np.empty(len(observations))
    noise_cosine = np.empty(len(observations))
    for start in range(0, len(pairs), 256):
        end = min(start + 256, len(pairs))
        parent = clean[parent_idx[start:end]].astype(np.float64)
        candidate = clean[candidate_idx[start:end]].astype(np.float64)
        pair_l1[start:end] = np.mean(np.abs(parent - candidate), axis=1) / 100
    observed_clean_idx = np.asarray([clean_lookup[s] for s in observations["structure_state_id"]])
    for start in range(0, len(observations), 256):
        end = min(start + 256, len(observations))
        truth = clean[observed_clean_idx[start:end]].astype(np.float64)
        view = observed[start:end].astype(np.float64)
        noise_l1[start:end] = np.mean(np.abs(view - truth), axis=1) / 100
        denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(view, axis=1)
        cosine = np.divide(np.sum(truth * view, axis=1), denominator,
                           out=np.ones_like(denominator), where=denominator > 0)
        noise_cosine[start:end] = np.maximum(0, 1 - cosine)
    distances = observations[["xrd_sample_id", "structure_state_id", "parent_group_id"]].copy()
    distances["noise_l1_distance"] = noise_l1
    distances["noise_cosine_distance"] = noise_cosine
    floors = distances.groupby("parent_group_id")["noise_l1_distance"].quantile(0.95)
    ratio = pair_l1 / floors.loc[pairs["parent_group_id"]].to_numpy()
    reference = stored_distances.set_index("xrd_sample_id").loc[observations["xrd_sample_id"]]
    floor_reference = stored_floors.set_index("parent_group_id").loc[floors.index]

    errors = {
        "pair_l1_max_absolute_error": float(np.max(np.abs(pair_l1 - pairs["xrd_l1_distance"]))),
        "noise_l1_max_absolute_error": float(np.max(np.abs(noise_l1 - reference["noise_l1_distance"]))),
        "noise_cosine_max_absolute_error": float(np.max(np.abs(noise_cosine - reference["noise_cosine_distance"]))),
        "parent_q95_max_absolute_error": float(np.max(np.abs(floors.to_numpy() - floor_reference["noise_l1_q95"]))),
        "ratio_max_absolute_error": float(np.max(np.abs(ratio - pairs["identifiability_ratio"]))),
    }
    tolerances = {
        "pair_l1_max_absolute_error": 1e-7,
        "noise_l1_max_absolute_error": 1e-10,
        "noise_cosine_max_absolute_error": 1e-10,
        "parent_q95_max_absolute_error": 1e-10,
        "ratio_max_absolute_error": 1e-5,
    }
    checks = {name: errors[name] <= tolerance for name, tolerance in tolerances.items()}
    checks.update(config_checks)
    candidates = task_index.loc[pairs["candidate_structure_state_id"]]
    for column in ("parent_group_id", "id_split"):
        checks[f"pair_{column}_matches_task_manifest"] = bool(
            np.array_equal(candidates[column], pairs[column])
        )
    groups = observations.groupby("structure_state_id").indices
    accuracy = np.empty(len(pairs))
    print("Recomputed pair distances and parent floors; checking fixed nearest prototypes", flush=True)
    for index, pair in enumerate(pairs.itertuples(index=False)):
        first = clean[clean_lookup[pair.parent_structure_id]].astype(np.float64)
        second = clean[clean_lookup[pair.candidate_structure_state_id]].astype(np.float64)
        a = observed[groups[pair.parent_structure_id]].astype(np.float64)
        b = observed[groups[pair.candidate_structure_state_id]].astype(np.float64)
        # Match the existing diagnostic's deterministic tie rule, without fitting.
        a_correct = np.mean(np.abs(a - first), axis=1) <= np.mean(np.abs(a - second), axis=1)
        b_correct = np.mean(np.abs(b - second), axis=1) < np.mean(np.abs(b - first), axis=1)
        accuracy[index] = np.concatenate([a_correct, b_correct]).mean()

    selected_parents = sorted(set(pairs.nlargest(2, "identifiability_ratio")["parent_structure_id"]))
    selected_parents += [str(pairs["parent_structure_id"].min())]
    selected_states = sorted(set(pairs.loc[pairs["parent_structure_id"].isin(selected_parents),
                                           "candidate_structure_state_id"]) | set(selected_parents))
    rendered_checks = []
    axis = np.round(np.arange(20, 100 + 0.02 / 2, 0.02), 8)
    sigma = 0.1 / (2 * np.sqrt(2 * np.log(2)))
    peak_cache = {}
    for state in selected_states:
        row = task_index.loc[state]
        path = source / row["peak_shard_path"]
        if path not in peak_cache:
            peak_cache[path] = read_frame(path)
        peaks = peak_cache[path]
        peaks = peaks.loc[peaks["xrd_sample_id"].eq(row["xrd_sample_id"])]
        if peaks.empty:
            raise ValueError(f"No recorded reflections for {state}")
        values = np.zeros(4001)
        for peak in peaks.itertuples(index=False):
            values += float(peak.intensity) * np.exp(-0.5 * ((axis - float(peak.two_theta)) / sigma)**2)
        if values.max() > 0:
            values *= 100 / values.max()
        error = float(np.max(np.abs(values.astype(np.float32) - clean[clean_lookup[state]])))
        rendered_checks.append({"structure_state_id": state, "reflection_count": len(peaks),
                                "max_intensity_error": error, "pass": error <= 1e-4})

    sys.path.insert(0, str(root))
    from alloy_xrd_bench.hea_identifiability import assign_identifiability_classes
    missing_case = assign_identifiability_classes(pd.DataFrame({
        "identifiability_ratio": [2.0], "oracle_accuracy": [np.nan]
    })).iloc[0]
    parent_counts = observations.groupby("parent_group_id").size()
    state_counts = observations.groupby("structure_state_id").size()
    split_leakage = observations.groupby("parent_group_id")["id_split"].nunique().gt(1).sum()
    checks["no_parent_split_leakage"] = int(split_leakage) == 0
    checks["all_parents_have_25_observations"] = bool(parent_counts.eq(25).all())
    checks["all_states_have_5_observations"] = bool(state_counts.eq(5).all())
    checks["sampled_peak_reconstruction"] = all(r["pass"] for r in rendered_checks)
    checks["stored_ratio_threshold_count_matches"] = int((ratio >= 1).sum()) == int((pairs["identifiability_ratio"] >= 1).sum())
    detail = pairs[["pair_id", "parent_structure_id", "candidate_structure_state_id", "id_split",
                    "identifiability_ratio", "oracle_accuracy", "identifiability_status"]].copy()
    detail["recomputed_pair_l1"] = pair_l1
    detail["recomputed_parent_noise_q95"] = floors.loc[pairs["parent_group_id"]].to_numpy()
    detail["recomputed_ratio"] = ratio
    detail["new_fixed_prototype_accuracy"] = accuracy
    report = {
        "audit_scope": "read-only released-array verification plus separately labelled fixed-prototype diagnostic",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
        "script_sha256": sha256(Path(__file__)),
        "input_files": evidence,
        "clean_xrd_config": clean_config,
        "observation_config": observation_config,
        "clean_arrays": clean_files,
        "observed_arrays": observed_files,
        "checks": checks,
        "all_array_checks_pass": all(checks.values()),
        "numeric_errors": errors,
        "numeric_tolerances": tolerances,
        "pairs": len(pairs),
        "parents": int(pairs["parent_group_id"].nunique()),
        "state_view_counts": distribution(state_counts),
        "parent_view_counts": distribution(parent_counts),
        "pair_l1": distribution(pair_l1),
        "parent_noise_q95": distribution(floors),
        "recomputed_ratio": distribution(ratio),
        "pairs_ratio_ge_1": int((ratio >= 1).sum()),
        "stored_oracle_finite_count": int(np.isfinite(pairs["oracle_accuracy"]).sum()),
        "stored_status_counts": pairs["identifiability_status"].value_counts().to_dict(),
        "missing_oracle_reproduction": {
            "input_ratio": 2.0, "output_class": missing_case["identifiability_class"],
            "output_status": missing_case["identifiability_status"],
            "output_noise_identifiable": bool(missing_case["noise_identifiable"]),
            "interpretation": "Missing calibration is classified as non_identifiable even with ratio above 1."
        },
        "new_fixed_prototype_diagnostic": {
            "status": "new audit calculation, not a published or independently validated model result",
            "accuracy": distribution(accuracy),
            "pair_accuracy_ge_0_8": int((accuracy >= 0.8).sum()),
            "ratio_ge_1_and_accuracy_ge_0_8": int(((ratio >= 1) & (accuracy >= 0.8)).sum()),
            "caveat": "Five frozen views per class; pairs share parent views. This is not a population accuracy estimate or an identifiability proof."
        },
        "sampled_clean_peak_reconstruction": rendered_checks,
        "limits": [
            "No experimental noise calibration or real SRO labels were added.",
            "A ratio below 1 is failure of this diagnostic criterion, not a theorem of non-identifiability.",
            "Stored reflection reconstruction is not independent validation of the scattering physics.",
            "Canonical structure-to-reflection generation is code/provenance inspected, not rerun for all 84019 structures."
        ],
    }
    for relative in ("alloy_xrd_bench/xrd.py", "alloy_xrd_bench/occupancy_production.py",
                     "alloy_xrd_bench/occupancy_expansion.py", "alloy_xrd_bench/hea_identifiability.py"):
        report["input_files"].append({"path": str(root / relative), "sha256": sha256(root / relative)})
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "pair_audit.csv", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "all_array_checks_pass", "numeric_errors", "recomputed_ratio", "pairs_ratio_ge_1",
        "stored_status_counts", "new_fixed_prototype_diagnostic"
    )}, indent=2), flush=True)
    if not report["all_array_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
