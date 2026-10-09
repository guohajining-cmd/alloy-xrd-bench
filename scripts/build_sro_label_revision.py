"""Recompute every released SRO label without modifying source data or splits."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.occupancy_pipeline import _sro
from alloy_xrd_bench.sro_revision import check_label_correction, file_sha256


def recompute(item):
    row, payload, encoding = item
    structure = Structure.from_dict(json.loads(payload) if encoding == "json" else ast.literal_eval(payload))
    corrected = _sro(structure, row["parent_lattice"])
    result = {key: row[key] for key in ("structure_state_id", "parent_group_id", "xrd_sample_id", "structure_sha256")}
    for shell, values in enumerate(corrected, 1):
        column = f"sro_shell{shell}_json"
        result[f"added_shell{shell}"] = check_label_correction(json.loads(row[column]), values)
        result[column] = json.dumps(values, sort_keys=True, allow_nan=False)
    return result


def audit_ood(tasks, labels, out):
    from alloy_xrd_bench.occupancy_expansion import _id_split, _sro_norm

    combined = tasks[["parent_group_id", "sro_shell1_json", "sro_ood_split"]].copy()
    combined["corrected"] = labels.sro_shell1_json.to_numpy()
    groups = combined.groupby("parent_group_id", sort=False).first()
    groups["old_norm"] = groups.sro_shell1_json.map(_sro_norm)
    groups["corrected_norm"] = groups.corrected.map(_sro_norm)
    thresholds = {}
    for name in ("old", "corrected"):
        low, high = groups[f"{name}_norm"].quantile([.15, .85]).tolist()
        thresholds[name] = [low, high]
        groups[f"{name}_rule_split"] = [
            "test" if value <= low or value >= high else
            ("validation" if _id_split(str(group)) == "validation" else "train")
            for group, value in groups[f"{name}_norm"].items()]
    details = groups.drop(columns=["sro_shell1_json", "corrected"])
    details.to_csv(out / "sro_ood_audit.csv")
    return {"thresholds": thresholds,
            "old_rule_disagrees_with_stored": int((groups.old_rule_split != groups.sro_ood_split).sum()),
            "groups_that_would_change_under_corrected_rule": int((groups.corrected_rule_split != groups.sro_ood_split).sum()),
            "applied_to_benchmark": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hea_validation_v1.json")
    parser.add_argument("--independent-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, out = args.dataset.resolve(), args.output.resolve()
    if out == root or out.is_relative_to(root) or out == ROOT / "data" or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    paths = dict(manifest=root / "manifests/task_samples.parquet",
                 structures=root / "manifests/structures.parquet",
                 payloads=root / "xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet",
                 counterfactuals=root / "derived/occupancy/hea_sro_cf_pilot_v2/structures.parquet",
                 independent_sample=args.independent_sample.resolve(),
                 implementation=ROOT / "alloy_xrd_bench/occupancy_pipeline.py",
                 builder=Path(__file__).resolve(), config=args.config.resolve())
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    meta = dict(schema_version="sro-label-revision-v1", status="in_progress",
                source_manifest_sha256=hashes["manifest"], source_hashes=hashes,
                source_paths={key: str(path) for key, path in paths.items()},
                sro_targets=config["sro_targets"], workers=args.workers,
                qualification="Arithmetic correction under the existing radial-shell convention; not experimental validation.",
                split_policy="Frozen original splits; SRO-dependent split definition audited separately.",
                structure_hash_policy="Retain original IDs and stored hashes; do not replace them with runtime serialization hashes.")
    (out / "revision.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    start = time.monotonic()
    tasks = pd.read_parquet(paths["manifest"])
    structures = pd.read_parquet(paths["structures"]).set_index("dataset_structure_id")
    payloads = pd.read_parquet(paths["payloads"]).set_index("source_row")
    counterfactuals = pd.read_parquet(paths["counterfactuals"]).set_index("structure_state_id")
    if tasks.structure_state_id.duplicated().any() or not structures.index.is_unique or not payloads.index.is_unique or not counterfactuals.index.is_unique:
        raise ValueError("Duplicate source identity")
    def inputs():
        for row in tasks.to_dict("records"):
            if row["fidelity"] == "source_relaxed_dft":
                source_row = structures.loc[row["structure_state_id"], "source_row"]
                yield row, payloads.loc[source_row, "structure_as_dict"], "literal"
            elif row["fidelity"] == "fixed_lattice_counterfactual":
                yield row, counterfactuals.loc[row["structure_state_id"], "structure_json"], "json"
            else:
                raise ValueError(f"Unexpected fidelity {row['fidelity']}")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(recompute, inputs(), chunksize=32), 1):
            records.append(result)
            if index % 2000 == 0:
                print(f"Recomputed {index}/{len(tasks)} in {time.monotonic()-start:.1f}s", flush=True)
    labels = pd.DataFrame(records)
    sample = pd.read_parquet(paths["independent_sample"]).set_index("structure_state_id")
    indexed = labels.set_index("structure_state_id")
    if not set(sample.index).issubset(indexed.index):
        raise ValueError("Independent sample is not covered")
    for identifier, row in sample.iterrows():
        if row.max_ase_error > 1e-10 or row.mask_disagreement or row.old_values_changed:
            raise ValueError("Independent reference did not pass")
        for shell in (1, 2):
            actual = json.loads(indexed.loc[identifier, f"sro_shell{shell}_json"])
            expected = json.loads(row[f"corrected_shell{shell}_json"])
            if set(actual) != set(expected) or any(abs(actual[key]-expected[key]) > 1e-10 for key in expected):
                raise ValueError(f"Independent sample mismatch: {identifier}")
    labels.to_parquet(out / "labels.parquet", index=False)
    changes = tasks[["structure_state_id", "fidelity", *config["split_columns"]]].copy()
    changes["added_labels"] = labels.added_shell1 + labels.added_shell2
    changes.to_csv(out / "label_changes.csv", index=False)
    scope = {name: dict(states=len(part), affected_states=int((part.added_labels > 0).sum()),
                        added_labels=int(part.added_labels.sum())) for name, part in changes.groupby("fidelity")}
    ood = audit_ood(tasks, labels, out)
    if any(file_sha256(path) != hashes[key] for key, path in paths.items()):
        raise ValueError("Input changed during recomputation")
    meta.update(status="verified", labels_sha256=file_sha256(out / "labels.parquet"),
                row_count=len(labels), scope=scope, independent_sample_matches=len(sample),
                sro_ood_audit=ood, elapsed_seconds=time.monotonic()-start)
    (out / "revision.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({key: meta[key] for key in ("status", "scope", "sro_ood_audit", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
