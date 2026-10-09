"""Reconstruct the deterministic released Ridge, then evaluate corrected labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import load_validation_rows, audit_parent_group_leakage
from alloy_xrd_bench.hea_validation_metrics import masked_parent_macro_sro
from alloy_xrd_bench.hea_validation_models import PerTargetRidge
from alloy_xrd_bench.sro_reassessment import align_predictions, assess_revision
from alloy_xrd_bench.sro_revision import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hea_validation_v1.json")
    parser.add_argument("--revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["A", "C"], required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == args.dataset.resolve() or out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    stage = out / args.stage
    stage.mkdir(parents=True, exist_ok=True)
    if any(stage.iterdir()):
        raise ValueError("Stage output already exists; preserve it and select a fresh directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("sro_label_revision"):
        raise ValueError("Base config must refer to original labels")
    old_rows = load_validation_rows(args.dataset, config)
    if args.stage == "C":
        if args.revision is None:
            raise ValueError("C requires a verified label revision")
        config = {**config, "sro_label_revision": str(args.revision.resolve())}
        rows = load_validation_rows(args.dataset, config)
    else:
        rows = old_rows
    (stage / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    protocol = dict(stage=args.stage, model="composition-ridge", seed=2027,
                    origin="Deterministic SRO-only reconstruction with the original PerTargetRidge, full data, fixed alpha; not a recovered historical prediction file.",
                    reproduction_tolerance=1e-7,
                    source_manifest_sha256=file_sha256(args.dataset / "manifests/task_samples.parquet"),
                    config_sha256=file_sha256(stage / "config.json"),
                    model_implementation_sha256=file_sha256(ROOT / "alloy_xrd_bench/hea_validation_models.py"),
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                    sro_targets=config["sro_targets"], split_policy="All original assignments frozen",
                    normalization="None; same 8 float32 composition features converted to float64 by PerTargetRidge")
    if args.stage == "C":
        protocol["revision_sha256"] = file_sha256(args.revision / "revision.json")
        protocol["labels_sha256"] = file_sha256(args.revision / "labels.parquet")
        original_protocol = json.loads((out / "A/protocol.json").read_text(encoding="utf-8"))
        if original_protocol["sro_targets"] != config["sro_targets"] or original_protocol["source_manifest_sha256"] != protocol["source_manifest_sha256"]:
            raise ValueError("A predictions are bound to different data or target order")
    (stage / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    published = pd.read_csv(ROOT / "docs/paper/generated/hea_sro_results.csv")
    published = published[published.model == "composition-ridge"].set_index("split_column")
    x = np.stack([row.composition for row in rows])
    y = np.stack([row.sro_values for row in rows])
    mask = np.stack([row.sro_mask for row in rows])
    old_y = np.stack([row.sro_values for row in old_rows])
    old_mask = np.stack([row.sro_mask for row in old_rows])
    shells = [int(target[5]) for target in config["sro_targets"]]
    results = {}
    for split in config["split_columns"]:
        if audit_parent_group_leakage(rows, split):
            raise ValueError(f"Parent leakage: {split}")
        train = np.array([row.splits[split] == "train" for row in rows])
        test = np.array([row.splits[split] == "test" for row in rows])
        expected = pd.DataFrame([dict(xrd_sample_id=row.xrd_sample_id, parent_group_id=row.parent_group_id)
                                 for row, keep in zip(rows, test) if keep])
        print(f"{args.stage}: fitting {split}, train={train.sum()}, test={test.sum()}", flush=True)
        with threadpool_limits(limits=1):
            estimator = PerTargetRidge(alpha=float(config["model_budgets"]["composition_ridge"]["alpha"]))
            prediction = estimator.fit(x[train], y[train], mask[train]).predict(x[test])
        joblib.dump(estimator, stage / f"{split}.joblib")
        frame = expected.assign(model="composition-ridge", seed=2027, split_column=split,
                                sro_predictions=[row.tolist() for row in prediction])
        frame.to_parquet(stage / f"{split}.parquet", index=False)
        if args.stage == "A":
            metrics = masked_parent_macro_sro(y[test], prediction, mask[test], expected.parent_group_id.tolist(), shells)
            reference = float(published.loc[split, "parent_macro_mae_mean"])
            difference = metrics["masked_parent_macro_mae"] - reference
            summary = dict(A=metrics, published_mae=reference, reproduction_difference=difference,
                           reproduced_within_tolerance=abs(difference) <= protocol["reproduction_tolerance"])
        else:
            old_prediction = align_predictions(expected, pd.read_parquet(out / "A" / f"{split}.parquet"), len(shells))
            summary, parents = assess_revision(old_y[test], old_mask[test], y[test], mask[test],
                                                old_prediction, expected.parent_group_id.tolist(), shells, prediction)
            parents.to_csv(stage / f"{split}.parent_errors.csv")
            for phase in ("A", "B", "C"):
                summary[phase]["relative_to_same_phase_ridge"] = 0.
            summary["old_prediction_origin"] = protocol["origin"]
        results[split] = summary
        (stage / f"{split}.metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key in ("A", "B", "C", "reproduction_difference", "C_minus_B")}), flush=True)
    (stage / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if file_sha256(args.dataset / "manifests/task_samples.parquet") != protocol["source_manifest_sha256"]:
        raise ValueError("Source manifest changed during training; results must not be used")


if __name__ == "__main__":
    main()
