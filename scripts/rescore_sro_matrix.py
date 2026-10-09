"""Reproduce and rescore the complete historical HEA prediction matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import load_validation_rows, audit_parent_group_leakage
from alloy_xrd_bench.sro_revision import file_sha256
from alloy_xrd_bench.sro_reassessment import align_predictions, assess_revision, paired_interval

MODELS = ["composition-ridge", "peak-xgboost", "resnet1d", "transformer", "physics-transformer"]


def expected_runs(config):
    return [(model, seed, split) for split in config["split_columns"] for model in MODELS
            for seed in ([2027] if model == "composition-ridge" else config["seeds"])]


def verify_downloads(directory, manifest_path=None):
    entries = {}
    for line in (manifest_path or directory / "remote_files.sha256").read_text(encoding="utf-8").splitlines():
        digest, remote = line.split(maxsplit=1)
        name = Path(remote.strip()).name
        if name in entries or len(digest) != 64:
            raise ValueError("Invalid or aliased download manifest")
        entries[name] = digest
    for name, digest in entries.items():
        path = directory / name
        if not path.exists():
            path = directory / "predictions" / name
        if file_sha256(path) != digest:
            raise ValueError(f"Download hash mismatch: {name}")
    return entries


def stem(model, seed, split):
    prefix = model.replace("-", "_") if model in ("composition-ridge", "peak-xgboost") else model
    return f"{prefix}_seed{seed}_{split}_test"


def verify_score(actual, expected, name, tolerance=1e-7):
    for metric in ("masked_parent_macro_mae", "rmse", "parent_count", "valid_target_count"):
        if not np.isfinite(actual[metric]) or not np.isfinite(expected[metric]) or abs(actual[metric]-expected[metric]) > tolerance:
            raise ValueError(f"Score cannot be reproduced: {name} {metric}")
    for shell in ("shell1", "shell2"):
        left, right = actual["shell_spearman"][shell], expected["shell_spearman"][shell]
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"Score cannot be reproduced: {name} {shell}")
        elif not np.isfinite(left) or not np.isfinite(right) or abs(left-right) > tolerance:
            raise ValueError(f"Score cannot be reproduced: {name} {shell}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    hashes = verify_downloads(args.historical)
    config = json.loads((args.historical / "hea_validation_v1.json").read_text(encoding="utf-8"))
    runs = expected_runs(config)
    corrected = None
    corrected_hashes = None
    if args.corrected_run:
        corrected_hashes = verify_downloads(args.corrected_run)
        corrected = json.loads((args.corrected_run / "protocol.json").read_text(encoding="utf-8"))
        if corrected["status"] != "complete" or not corrected["input_hashes_unchanged"]:
            raise ValueError("Corrected training batch is incomplete or inputs changed")
        if corrected["labels_sha256"] != file_sha256(args.revision / "labels.parquet"):
            raise ValueError("Corrected training used a different label revision")
        if corrected["source_manifest_sha256"] != file_sha256(args.dataset / "manifests/task_samples.parquet"):
            raise ValueError("Corrected training used a different source manifest")
        training_config = json.loads((args.corrected_run / "config.json").read_text(encoding="utf-8"))
        for key in ("sro_targets", "model_budgets", "elements", "split_columns"):
            if training_config[key] != config[key]:
                raise ValueError(f"Corrected training config differs: {key}")
        runs = [tuple(row) for row in corrected["runs"]]
        if not set(runs).issubset(set(expected_runs(config))):
            raise ValueError("Unexpected corrected training cases")
    protocol = dict(status="in_progress", historical_file_hashes=hashes,
                    source_manifest_sha256=file_sha256(args.dataset / "manifests/task_samples.parquet"),
                    revision_sha256=file_sha256(args.revision / "revision.json"),
                    corrected_protocol=corrected, corrected_file_hashes=corrected_hashes, reproduction_tolerance=1e-7,
                    expected_run_count=len(runs), sro_targets=config["sro_targets"])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Download hashes verified: {len(hashes)} files; loading original and revised labels", flush=True)
    old_rows = load_validation_rows(args.dataset, config)
    new_rows = load_validation_rows(args.dataset, {**config, "sro_label_revision": str(args.revision.resolve())})
    old_y, old_mask = np.stack([r.sro_values for r in old_rows]), np.stack([r.sro_mask for r in old_rows])
    new_y, new_mask = np.stack([r.sro_values for r in new_rows]), np.stack([r.sro_mask for r in new_rows])
    shells = [int(t[5]) for t in config["sro_targets"]]
    cache = {}
    for split in config["split_columns"]:
        if audit_parent_group_leakage(old_rows, split):
            raise ValueError(f"Parent split leakage: {split}")
        test = np.array([r.splits[split] == "test" for r in old_rows])
        frame = pd.DataFrame([dict(xrd_sample_id=r.xrd_sample_id, parent_group_id=r.parent_group_id)
                              for r, keep in zip(old_rows, test) if keep])
        cache[split] = test, frame
    rows, parent_tables = [], {}
    for index, (model, seed, split) in enumerate(runs, 1):
        name = stem(model, seed, split)
        test, expected = cache[split]
        frame = pd.read_parquet(args.historical / f"{name}.parquet")
        for column, value in [("model", model), ("seed", seed), ("split_column", split)]:
            if set(frame[column]) != {value}:
                raise ValueError(f"Prediction metadata mismatch: {name} {column}")
        prediction = align_predictions(expected, frame, len(shells))
        new_prediction = None
        if corrected:
            new_frame = pd.read_parquet(args.corrected_run / "predictions" / f"{name}.parquet")
            for column, value in [("model", model), ("seed", seed), ("split_column", split)]:
                if set(new_frame[column]) != {value}:
                    raise ValueError(f"Corrected prediction metadata mismatch: {name}")
            new_prediction = align_predictions(expected, new_frame, len(shells))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            summary, parents = assess_revision(old_y[test], old_mask[test], new_y[test], new_mask[test],
                prediction, expected.parent_group_id.tolist(), shells, new_prediction)
        original = json.loads((args.historical / f"{name}.metrics.json").read_text(encoding="utf-8"))["sro"]
        verify_score(summary["A"], original, name, protocol["reproduction_tolerance"])
        if corrected:
            new_metrics = json.loads((args.corrected_run / "predictions" / f"{name}.metrics.json").read_text(encoding="utf-8"))["sro"]
            verify_score(summary["C"], new_metrics, f"corrected {name}", protocol["reproduction_tolerance"])
            summary["corrected_metric_reproduced"] = True
        summary.update(model=model, seed=seed, split_column=split, old_metric_reproduced=True,
                       historical_prediction_sha256=hashes[f"{name}.parquet"])
        parents.to_csv(out / f"{name}.parents.csv")
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        parent_tables[(model, seed, split)] = parents
        for phase in ("A", "B", "C"):
            if phase in summary:
                score = summary[phase]
                rows.append(dict(model=model, seed=seed, split_column=split, phase=phase,
                                 mae=score["masked_parent_macro_mae"], rmse=score["rmse"],
                                 parent_count=score["parent_count"], valid_target_count=score["valid_target_count"],
                                 spearman_shell1=score["shell_spearman"]["shell1"], spearman_shell2=score["shell_spearman"]["shell2"]))
        print(f"{index}/{len(runs)} {name}: A={summary['A']['masked_parent_macro_mae']:.6f}, B={summary['B']['masked_parent_macro_mae']:.6f}" +
              (f", C={summary['C']['masked_parent_macro_mae']:.6f}" if "C" in summary else ""), flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)
    aggregate = scores.groupby(["split_column", "model", "phase"], sort=False).agg(
        mae_mean=("mae", "mean"), seed_sd=("mae", "std"), seed_count=("seed", "count"),
        spearman_shell1_mean=("spearman_shell1", "mean"), spearman_shell2_mean=("spearman_shell2", "mean")).reset_index()
    aggregate["seed_sd"] = aggregate.seed_sd.fillna(0.)
    aggregate.to_csv(out / "aggregate.csv", index=False)
    published = pd.read_csv(ROOT / "docs/paper/generated/hea_sro_results.csv")
    for record in aggregate[aggregate.phase == "A"].itertuples(index=False):
        if record.seed_count != (1 if record.model == "composition-ridge" else len(config["seeds"])):
            continue
        prior = published[(published.model == record.model) & (published.split_column == record.split_column)].iloc[0]
        if abs(record.mae_mean - prior.parent_macro_mae_mean) > protocol["reproduction_tolerance"]:
            raise ValueError("Recovered matrix disagrees with paper aggregate")
    comparisons = []
    for (split, model), part in scores.groupby(["split_column", "model"]):
        seeds = sorted(part.seed.unique())
        table = pd.concat([parent_tables[(model, seed, split)] for seed in seeds], keys=seeds, names=["seed"])
        mean_errors = table.groupby(level="parent_group_id").mean()
        for left, right in [("B", "A"), ("C", "B")]:
            if left not in mean_errors:
                continue
            pairs = mean_errors.dropna(subset=[left, right])
            comparisons.append(dict(split_column=split, model=model, contrast=f"{left}-{right}", seed_count=len(seeds),
                                    **paired_interval(pairs[left]-pairs[right])))
    (out / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    protocol.update(status="verified", completed_run_count=len(runs), comparison_policy="Average error across seeds per parent before paired bootstrap; seed SD reported separately")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"VERIFIED: {len(runs)} historical runs; original score and ID checks passed", flush=True)


if __name__ == "__main__":
    main()
