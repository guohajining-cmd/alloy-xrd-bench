"""Full-data matched-capacity C, peak-X, and C+X controls on corrected SRO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import load_validation_rows, audit_parent_group_leakage
from alloy_xrd_bench.sro_revision import file_sha256
from alloy_xrd_bench.sro_reassessment import parent_errors
from alloy_xrd_bench.hea_validation_metrics import masked_parent_macro_sro


def feature_sets(composition, peaks):
    if composition.ndim != 2 or peaks.ndim != 2 or len(composition) != len(peaks):
        raise ValueError("Unaligned feature matrices")
    if not np.isfinite(composition).all() or not np.isfinite(peaks).all():
        raise ValueError("Nonfinite features")
    return {"C": composition, "X": peaks, "CX": np.column_stack([composition, peaks])}


def fixed_parameters(budget, seed, threads):
    if threads < 1:
        raise ValueError("Positive CPU thread count required")
    return dict(n_estimators=int(budget["n_estimators"]), max_depth=int(budget["max_depth"]),
                learning_rate=float(budget["learning_rate"]), tree_method="hist", device="cpu",
                random_state=seed, n_jobs=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Use a new output directory outside source data")
    from alloy_xrd_bench.hea_validation_models import PerTargetXGBoost, peak_feature_matrix
    import xgboost
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["sro_label_revision"] = str(args.revision.resolve())
    budget = config["model_budgets"]["peak_xgboost"]
    parameters = fixed_parameters(budget, args.seed, args.threads)
    inputs = [args.dataset / "manifests/task_samples.parquet",
              args.dataset / "manifests/composition_only_features.parquet",
              args.revision / "revision.json", args.revision / "labels.parquet", args.config,
              Path(__file__), ROOT / "alloy_xrd_bench/hea_validation_models.py",
              ROOT / "alloy_xrd_bench/hea_validation_data.py", ROOT / "alloy_xrd_bench/sro_revision.py"]
    hashes = {str(p.resolve()): file_sha256(p) for p in inputs}
    rows = load_validation_rows(args.dataset, config)
    for split in config["split_columns"]:
        if audit_parent_group_leakage(rows, split):
            raise ValueError(f"Parent leakage: {split}")
    for path in sorted({r.grid_shard_path for r in rows}):
        hashes[path] = file_sha256(path)
    out.mkdir(parents=True)
    (out / "predictions").mkdir()
    protocol = dict(status="running", parameters=parameters, seed=args.seed, source_hashes=hashes,
                    config=config, python=platform.python_version(), numpy=np.__version__,
                    xgboost=xgboost.__version__, sample_limit=None,
                    source_manifest_sha256=hashes[str(inputs[0].resolve())],
                    labels_sha256=hashes[str((args.revision / "labels.parquet").resolve())],
                    planned_runs=[[s, f] for s in config["split_columns"] for f in ("C", "X", "CX")],
                    qualification="Post-review exploratory full-data feature ablation; fixed original Peak-XGBoost budget, no test-set tuning; one fitted seed, CPU hist throughout. Not a new registered leaderboard.")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Loading peak features for all {len(rows)} rows", flush=True)
    with threadpool_limits(limits=args.threads):
        peaks = peak_feature_matrix(rows, max_peaks=int(budget["max_peaks"]))
    features = feature_sets(np.stack([r.composition for r in rows]), peaks)
    y, mask = np.stack([r.sro_values for r in rows]), np.stack([r.sro_mask for r in rows])
    shells = [int(t[5]) for t in config["sro_targets"]]
    records = []
    for split in config["split_columns"]:
        train = np.array([r.splits[split] == "train" for r in rows])
        test = np.array([r.splits[split] == "test" for r in rows])
        identity = pd.DataFrame([dict(xrd_sample_id=r.xrd_sample_id, parent_group_id=r.parent_group_id,
                                     structure_state_id=r.structure_state_id)
                                 for r, keep in zip(rows, test) if keep])
        for name, x in features.items():
            print(f"START {split} {name}: train={train.sum()}, test={test.sum()}", flush=True)
            with threadpool_limits(limits=args.threads):
                estimator = PerTargetXGBoost(**parameters).fit(x[train], y[train], mask[train])
                prediction = estimator.predict(x[test])
            if prediction.shape != y[test].shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid prediction")
            identifier = f"{split}_{name}"
            frame = identity.assign(model=f"control-xgboost-{name}", seed=args.seed, split_column=split,
                                    sro_predictions=list(prediction))
            frame.to_parquet(out / "predictions" / f"{identifier}.parquet", index=False)
            joblib.dump(estimator, out / "predictions" / f"{identifier}.joblib")
            errors = parent_errors(y[test], prediction, mask[test], identity.parent_group_id)
            errors.rename("mae").to_csv(out / f"{identifier}.parents.csv", index_label="parent_group_id")
            score = masked_parent_macro_sro(y[test], prediction, mask[test], identity.parent_group_id, shells)
            if abs(errors.mean() - score["masked_parent_macro_mae"]) > 1e-10:
                raise ValueError("Independent parent reduction disagrees with scorer")
            record = dict(split_column=split, feature_set=name, seed=args.seed, train_count=int(train.sum()),
                          test_count=int(test.sum()), sro=score)
            (out / f"{identifier}.metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            records.append(record)
            (out / "progress.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
            print(f"DONE {split} {name}: MAE={errors.mean():.9f}", flush=True)
            del estimator
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("Source input changed during training")
    protocol.update(status="complete", completed_run_count=len(records), input_hashes_unchanged=True,
                    output_hashes={str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*")
                                   if p.is_file() and p.name != "protocol.json"})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"COMPLETE {len(records)} matched-capacity cases", flush=True)


if __name__ == "__main__":
    main()
