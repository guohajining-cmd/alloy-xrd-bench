"""Check target changes for all fixed-cell pairs using the full label revision."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.sro_revision import file_sha256


def pair_change(parent, child, tolerance=1e-10):
    changes, masks = [], 0
    for left, right in zip(parent, child):
        if not all(math.isfinite(v) for v in [*left.values(), *right.values()]):
            raise ValueError("Nonfinite SRO target")
        masks += len(set(left) ^ set(right))
        changes.extend(abs(left[key]-right[key]) for key in sorted(set(left) & set(right)))
    return dict(common_targets=len(changes), mask_symmetric_difference=masks,
                changed_targets=sum(v > tolerance for v in changes),
                mean_abs_change=float(np.mean(changes)) if changes else None,
                max_abs_change=max(changes) if changes else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--spectral-audit", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Do not overwrite source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    manifest_path = args.dataset / "manifests/task_samples.parquet"
    meta = json.loads((args.revision / "revision.json").read_text(encoding="utf-8"))
    if meta["status"] != "verified" or meta["source_manifest_sha256"] != file_sha256(manifest_path) or meta["labels_sha256"] != file_sha256(args.revision / "labels.parquet"):
        raise ValueError("Unverified input binding")
    tasks = pd.read_parquet(manifest_path)
    labels = pd.read_parquet(args.revision / "labels.parquet").set_index("structure_state_id")
    cf = tasks[tasks.fidelity == "fixed_lattice_counterfactual"]
    if not labels.index.is_unique:
        raise ValueError("Duplicate label identity")
    records = []
    for row in cf.itertuples(index=False):
        parent, child = labels.loc[row.parent_structure_id], labels.loc[row.structure_state_id]
        if parent.parent_group_id != child.parent_group_id or child.parent_group_id != row.parent_group_id:
            raise ValueError("Pair parent identity mismatch")
        record = pair_change(tuple(json.loads(parent[f"sro_shell{s}_json"]) for s in (1, 2)),
                             tuple(json.loads(child[f"sro_shell{s}_json"]) for s in (1, 2)))
        records.append(dict(structure_state_id=row.structure_state_id, parent_structure_id=row.parent_structure_id,
                            strategy=row.strategy, id_split=row.id_split, **record))
    frame = pd.DataFrame(records)
    spectral = pd.read_csv(args.spectral_audit).rename(columns={"candidate_structure_state_id": "structure_state_id"})
    geometry = pd.read_csv(args.geometry_audit)
    for prior in (spectral, geometry):
        if set(prior.structure_state_id) != set(frame.structure_state_id):
            raise ValueError("Prior audit covers different pairs")
    frame = frame.merge(spectral[["structure_state_id", "parent_structure_id", "recomputed_pair_l1", "recomputed_ratio"]],
                        on=["structure_state_id", "parent_structure_id"], how="left", validate="one_to_one")
    frame = frame.merge(geometry, on=["structure_state_id", "parent_structure_id"], how="left", validate="one_to_one")
    if frame.recomputed_ratio.isna().any() or not frame.same_composition.all() or frame.max_cell_delta_A.max() > 1e-10 or frame.max_fractional_position_delta.max() > 1e-10:
        raise ValueError("Missing or incompatible prior geometry/spectral evidence")
    changed = (frame.changed_targets > 0) & (frame.mask_symmetric_difference == 0)
    frame.to_csv(out / "all_pairs.csv", index=False)
    report = dict(pair_count=len(frame), equal_mask_pairs=int((frame.mask_symmetric_difference == 0).sum()),
                  pairs_with_changed_defined_sro=int(changed.sum()),
                  pairs_with_unchanged_defined_sro=int(((frame.changed_targets == 0) & (frame.common_targets > 0)).sum()),
                  pairs_without_common_targets=int((frame.common_targets == 0).sum()),
                  mean_abs_sro_change_quantiles=frame.mean_abs_change.quantile([0, .5, .95, 1]).to_dict(),
                  changed_sro_pairs_at_or_above_synthetic_nuisance_floor=int((changed & (frame.recomputed_ratio >= 1)).sum()),
                  changed_sro_ratio_max=float(frame.loc[changed, "recomputed_ratio"].max()),
                  by_strategy={name: dict(pairs=len(part), changed_defined_sro=int((part.changed_targets > 0).sum()))
                               for name, part in frame.groupby("strategy")},
                  input_hashes={str(p): file_sha256(p) for p in [manifest_path, args.revision / "labels.parquet", args.spectral_audit, args.geometry_audit]},
                  qualification="Complete pair-label check from full coordinate-recomputed labels. Geometry and spectral differences reuse prior audits; no new XRD simulation or experimental-noise calibration. Numeric nuisance comparison is not the full formal identifiability qualification.")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
