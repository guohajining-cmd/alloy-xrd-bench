"""Assess held-out fixed-cell SRO changes using already verified predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.sro_revision import file_sha256, apply_label_revision
from scripts.verify_sro_signal_increment import bootstrap_contrast
from scripts.rescore_sro_matrix import stem


def pair_scores(parent_truth, child_truth, parent_mask, child_mask, parent_prediction, child_prediction):
    arrays = [parent_truth, child_truth, parent_mask, child_mask, parent_prediction, child_prediction]
    if any(a.shape != parent_truth.shape for a in arrays) or parent_truth.ndim != 2:
        raise ValueError("Unaligned pairs")
    if not np.array_equal(parent_mask, child_mask) or not parent_mask.any(axis=1).all():
        raise ValueError("Unequal or empty pair masks")
    if not all(np.isfinite(a).all() for a in [parent_truth, child_truth, parent_prediction, child_prediction]):
        raise ValueError("Nonfinite pair values")
    observed = child_truth-parent_truth
    predicted = child_prediction-parent_prediction

    def reduce(values):
        return np.where(parent_mask, np.abs(values), 0).sum(axis=1) / parent_mask.sum(axis=1)

    return pd.DataFrame(dict(change_error=reduce(predicted-observed),
                             zero_change_error=reduce(observed),
                             predicted_change_magnitude=reduce(predicted),
                             changed_target_count=((np.abs(observed)>1e-10) & parent_mask).sum(axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT/"data") or out.is_relative_to(args.dataset.resolve()):
        raise ValueError("Fresh non-data output required")
    proof = json.loads((args.verification/"protocol.json").read_text())
    if proof["status"] != "verified":
        raise ValueError("Independent full membership/score verification required")
    manifest = args.dataset/"manifests/task_samples.parquet"
    config = json.loads((args.runs[0]/"config.json").read_text())
    tasks = apply_label_revision(pd.read_parquet(manifest), manifest, args.revision, config["sro_targets"])
    if tasks.structure_state_id.duplicated().any():
        raise ValueError("Duplicate structure state")
    tasks = tasks.set_index("structure_state_id", drop=False)
    audit = pd.read_csv(args.pair_audit).set_index("structure_state_id")
    cf = tasks[tasks.fidelity.eq("fixed_lattice_counterfactual")]
    if set(cf.index) != set(audit.index) or not audit.same_composition.all() or audit.max_cell_delta_A.max()>1e-10 or audit.max_fractional_position_delta.max()>1e-10:
        raise ValueError("Incomplete or incompatible fixed-cell audit")
    cases = []
    for directory in args.runs:
        protocol = json.loads((directory/"protocol.json").read_text())
        for model, seed, split in protocol["runs"]:
            cases.append((model, seed, split, directory/"predictions"/f"{stem(model,seed,split)}.parquet"))
    if args.controls:
        control = json.loads((args.controls/"protocol.json").read_text())
        for split, feature in control["planned_runs"]:
            if split in proof["included_splits"]:
                cases.append((f"control-xgboost-{feature}", control["seed"], split,
                              args.controls/"predictions"/f"{split}_{feature}.parquet"))
    inputs = {str(p.resolve()): file_sha256(p) for p in [manifest, args.revision/"labels.parquet",
               args.verification/"protocol.json", args.pair_audit, Path(__file__)]}
    parents, records = {}, []
    out.mkdir(parents=True)
    for model, seed, split, path in cases:
        digest = file_sha256(path)
        if proof["input_hashes"].get(str(path.resolve())) != digest:
            raise ValueError("Prediction not bound by independent verification")
        inputs[str(path.resolve())] = digest
        child = cf[cf[split].eq("test")].copy()
        if child.empty:
            continue
        parent = tasks.loc[child.parent_structure_id].copy()
        if not parent[split].eq("test").all() or not np.array_equal(parent.parent_group_id, child.parent_group_id):
            raise ValueError("Pair membership is not held out together")
        values = []
        for frame in [parent, child]:
            pairs = [vectorize_sro(r.sro_shell1_json, r.sro_shell2_json, config["sro_targets"]) for r in frame.itertuples(index=False)]
            values.append((np.stack([v[0] for v in pairs]).astype(float), np.stack([v[1] for v in pairs])))
        prediction = pd.read_parquet(path).set_index("xrd_sample_id")
        ps = np.stack(prediction.loc[parent.xrd_sample_id].sro_predictions).astype(float)
        cs = np.stack(prediction.loc[child.xrd_sample_id].sro_predictions).astype(float)
        result = pair_scores(values[0][0], values[1][0], values[0][1], values[1][1], ps, cs)
        result["parent_group_id"] = child.parent_group_id.to_numpy()
        result["structure_state_id"] = child.index.to_numpy()
        result["strategy"] = child.strategy.to_numpy()
        result.to_csv(out/f"{split}_{model}_{seed}.pairs.csv", index=False)
        changed = result[result.changed_target_count > 0]
        if changed.empty:
            raise ValueError("No target-changing held-out pairs")
        parent_result = changed.groupby("parent_group_id")[["change_error", "zero_change_error", "predicted_change_magnitude"]].mean()
        parents[(split,model,seed)] = parent_result
        records.append(dict(split=split, model=model, seed=seed, all_test_pairs=len(result),
                            target_changing_pairs=len(changed), parents=len(parent_result),
                            change_error=float(parent_result.change_error.mean()),
                            zero_change_error=float(parent_result.zero_change_error.mean()),
                            predicted_change_magnitude=float(parent_result.predicted_change_magnitude.mean())))
    table = pd.DataFrame(records)
    table.to_csv(out/"scores.csv", index=False)
    results = []
    for (split,model), part in table.groupby(["split","model"]):
        candidate = pd.DataFrame({s:parents[(split,model,s)].change_error for s in sorted(part.seed)})
        reference = pd.DataFrame({s:parents[(split,model,s)].zero_change_error for s in sorted(part.seed)})
        results.append(dict(split=split, model=model, **bootstrap_contrast(candidate, reference)))
    (out/"comparisons.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    if any(file_sha256(p)!=digest for p,digest in inputs.items()):
        raise ValueError("Inputs changed during pair assessment")
    protocol = dict(status="verified", input_hashes=inputs, completed_cases=len(records),
                    metric="MAE of predicted child-minus-parent SRO change, versus zero predicted change; average targets within pair, changing pairs within parent, then parents. Zero-change target pairs excluded and counted.",
                    qualification="Post-review exploratory held-out fixed-cell synthetic diagnostic; no nuisance or experimental validation. This contrast is different from absolute SRO regression and cannot be substituted for its leaderboard.")
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
