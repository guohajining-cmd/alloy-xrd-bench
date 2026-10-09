"""Score saved old predictions on both released and corrected SRO labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import load_validation_rows, audit_parent_group_leakage
from alloy_xrd_bench.sro_reassessment import align_predictions, assess_revision
from alloy_xrd_bench.sro_revision import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction-config", type=Path, required=True,
                        help="Configuration used to produce the saved vectors; establishes target order")
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--corrected-predictions", type=Path)
    parser.add_argument("--corrected-config", type=Path)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.prediction_config.read_text(encoding="utf-8"))
    if config.get("sro_label_revision") or args.split_column not in config["split_columns"]:
        raise ValueError("Old prediction config must use original labels and a supported split")
    out = args.output.resolve()
    if out == args.dataset.resolve() or out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    old_rows = load_validation_rows(args.dataset, config)
    revised_config = {**config, "sro_label_revision": str(args.revision.resolve())}
    new_rows = load_validation_rows(args.dataset, revised_config)
    if audit_parent_group_leakage(old_rows, args.split_column):
        raise ValueError("Parent split leakage")
    old_test = [row for row in old_rows if row.splits[args.split_column] == "test"]
    new_test = [row for row in new_rows if row.splits[args.split_column] == "test"]
    expected = pd.DataFrame([dict(xrd_sample_id=row.xrd_sample_id, parent_group_id=row.parent_group_id) for row in old_test])
    if [row.xrd_sample_id for row in old_test] != [row.xrd_sample_id for row in new_test]:
        raise ValueError("Correction changed test membership")
    def predictions(path):
        frame = pd.read_parquet(path)
        for column, value in [("model", args.model), ("seed", args.seed), ("split_column", args.split_column)]:
            if column not in frame or set(frame[column]) != {value}:
                raise ValueError(f"Prediction metadata mismatch: {column}")
        return align_predictions(expected, frame, len(config["sro_targets"]))
    old_prediction = predictions(args.predictions)
    new_prediction = None
    hashes = dict(predictions=file_sha256(args.predictions), prediction_config=file_sha256(args.prediction_config),
                  source_manifest=file_sha256(args.dataset / "manifests/task_samples.parquet"),
                  revision=file_sha256(args.revision / "revision.json"),
                  implementation=file_sha256(ROOT / "alloy_xrd_bench/sro_reassessment.py"))
    if args.corrected_predictions:
        if not args.corrected_config:
            raise ValueError("Corrected predictions require their training configuration")
        corrected_config = json.loads(args.corrected_config.read_text(encoding="utf-8"))
        if corrected_config["sro_targets"] != config["sro_targets"] or not corrected_config.get("sro_label_revision"):
            raise ValueError("Corrected prediction configuration mismatch")
        trained_revision = Path(corrected_config["sro_label_revision"]) / "revision.json"
        if file_sha256(trained_revision) != hashes["revision"]:
            raise ValueError("Corrected model trained with a different revision")
        hashes.update(corrected_predictions=file_sha256(args.corrected_predictions),
                      corrected_config=file_sha256(args.corrected_config))
        new_prediction = predictions(args.corrected_predictions)
    summary, parents = assess_revision(
        np.stack([row.sro_values for row in old_test]), np.stack([row.sro_mask for row in old_test]),
        np.stack([row.sro_values for row in new_test]), np.stack([row.sro_mask for row in new_test]),
        old_prediction, expected.parent_group_id.tolist(), [int(target[5]) for target in config["sro_targets"]], new_prediction)
    summary.update(model=args.model, seed=args.seed, split_column=args.split_column,
                   artifact_hashes=hashes, sro_targets=config["sro_targets"],
                   target_order_provenance="Asserted by supplied training configuration; target order cannot be inferred from vector values")
    parents.to_csv(out / "parent_errors.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("A", "B", "newly_valid_target_count")}, indent=2))


if __name__ == "__main__":
    main()
