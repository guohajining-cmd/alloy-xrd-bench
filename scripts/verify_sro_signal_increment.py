"""Independent corrected-label rescoring and parent/seed uncertainty audit."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import load_validation_rows
from alloy_xrd_bench.sro_revision import file_sha256
from scripts.rescore_sro_matrix import stem, verify_downloads


def require_spectral_matrix(cases, config):
    models=("peak-xgboost", "resnet1d", "transformer", "physics-transformer")
    expected={(split,model,seed) for split in config["split_columns"] for model in models for seed in config["seeds"]}
    if set(cases)!=expected:
        raise ValueError(f"Incomplete spectral matrix: missing={len(expected-set(cases))}, extra={len(set(cases)-expected)}")


def score_prediction(expected, frame, truth, mask):
    required = {"xrd_sample_id", "parent_group_id", "sro_predictions"}
    if not required.issubset(frame) or frame.xrd_sample_id.isna().any() or frame.parent_group_id.isna().any():
        raise ValueError("Missing prediction identity")
    if frame.xrd_sample_id.duplicated().any() or expected.xrd_sample_id.duplicated().any():
        raise ValueError("Duplicate sample ID")
    if set(frame.xrd_sample_id) != set(expected.xrd_sample_id):
        raise ValueError("Incomplete or extra test membership")
    ordered = frame.set_index("xrd_sample_id").loc[expected.xrd_sample_id]
    if not np.array_equal(ordered.parent_group_id.to_numpy(), expected.parent_group_id.to_numpy()):
        raise ValueError("Parent mismatch")
    values = np.stack(ordered.sro_predictions).astype(np.float64)
    if values.shape != truth.shape or mask.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError("Invalid prediction array")
    if not np.isfinite(truth[mask]).all():
        raise ValueError("Nonfinite truth")
    valid = mask.sum(axis=1)
    errors = np.zeros(len(mask), dtype=float)
    eligible = valid > 0
    errors[eligible] = np.sum(np.where(mask, np.abs(values-truth), 0), axis=1)[eligible] / valid[eligible]
    series = pd.Series(errors[eligible], index=expected.parent_group_id.to_numpy()[eligible])
    parent = series.groupby(level=0, sort=True).mean()
    rmse = float(np.sqrt(np.square(values[mask]-truth[mask]).mean()))
    return parent, dict(mae=float(parent.mean()), rmse=rmse, parent_count=len(parent),
                        valid_target_count=int(mask.sum()), sample_count=len(mask),
                        eligible_sample_count=int(eligible.sum()))


def matched_arrays(candidate, reference):
    if candidate.index.has_duplicates or reference.index.has_duplicates:
        raise ValueError("Duplicate parent")
    if set(candidate.index) != set(reference.index):
        raise ValueError("Unmatched parent membership")
    c = candidate.sort_index().to_numpy(dtype=float).T
    r = reference.sort_index().to_numpy(dtype=float).T
    if c.shape[0] != r.shape[0] and r.shape[0] != 1 and c.shape[0] != 1:
        raise ValueError("Unmatched fitted seed count")
    if c.shape[0] == r.shape[0] and list(candidate.columns) != list(reference.columns):
        raise ValueError("Unmatched seed identities")
    if not c.size or not np.isfinite(c).all() or not np.isfinite(r).all() or (c < 0).any() or (r < 0).any():
        raise ValueError("Invalid parent errors")
    count = max(c.shape[0], r.shape[0])
    return np.broadcast_to(c, (count, c.shape[1])), np.broadcast_to(r, (count, r.shape[1]))


def bootstrap_contrast(candidate, reference, repeats=4000, seed=20261002):
    if repeats < 1:
        raise ValueError("Positive bootstrap repetitions required")
    c, r = matched_arrays(candidate, reference)
    rng = np.random.default_rng(seed)
    seed_count, parent_count = c.shape
    mean_c, mean_r = c.mean(axis=0), r.mean(axis=0)
    conditional, crossed = np.empty(repeats), np.empty(repeats)
    relative, relative_crossed = np.empty(repeats), np.empty(repeats)
    if np.any(mean_r < 0) or mean_r.mean() <= 0:
        raise ValueError("Zero reference error")
    # Each draw shares parent indices between methods; seed draws apply globally.
    for i in range(repeats):
        parents = rng.integers(0, parent_count, parent_count)
        seeds = rng.integers(0, seed_count, seed_count)
        c0, r0 = mean_c[parents].mean(), mean_r[parents].mean()
        c1 = c[seeds].mean(axis=0)[parents].mean()
        r1 = r[seeds].mean(axis=0)[parents].mean()
        if r0 <= 0 or r1 <= 0:
            raise ValueError("Zero resampled reference error")
        conditional[i], crossed[i] = c0-r0, c1-r1
        relative[i], relative_crossed[i] = 100*(1-c0/r0), 100*(1-c1/r1)
    per_seed = (c-r).mean(axis=1)
    return dict(candidate_mae=float(c.mean()), reference_mae=float(r.mean()),
                delta_mae=float((c-r).mean()), delta_ci95_parent=np.quantile(conditional, [.025, .975]).tolist(),
                delta_ci95_seed_parent=np.quantile(crossed, [.025, .975]).tolist(),
                relative_reduction_percent=float(100*(1-c.mean()/r.mean())),
                relative_ci95_parent=np.quantile(relative, [.025, .975]).tolist(),
                relative_ci95_seed_parent=np.quantile(relative_crossed, [.025, .975]).tolist(),
                delta_ci98_75_parent=np.quantile(conditional, [.00625, .99375]).tolist(),
                delta_ci98_75_seed_parent=np.quantile(crossed, [.00625, .99375]).tolist(),
                four_split_interval_policy="98.75% percentile intervals apply a Bonferroni allocation over four frozen splits for this one contrast, not over all exploratory model comparisons; bootstrap approximation, not preregistered confirmation.",
                per_seed_delta=per_seed.tolist(), seed_delta_sd=float(per_seed.std(ddof=1)) if seed_count > 1 else 0.,
                parent_count=parent_count, fitted_seed_count=seed_count, repeats=repeats, bootstrap_seed=seed,
                direction="Candidate minus reference; negative favors candidate",
                uncertainty="Parent-only interval conditions on fitted seeds. Crossed seed/parent interval resamples the observed seed IDs and held-out parent IDs; five seeds do not characterize all possible training randomness. Exploratory pointwise intervals, no multiplicity correction.")


def check_score(score, reported):
    for key, other in [("mae", "masked_parent_macro_mae"), ("rmse", "rmse"),
                       ("parent_count", "parent_count"), ("valid_target_count", "valid_target_count")]:
        if not np.isfinite(reported[other]) or abs(score[key]-reported[other]) > 1e-7:
            raise ValueError(f"Independent scoring mismatch: {key}")


def reproduce_ridge(x_train, truth, mask, x_test, alpha):
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    prediction = np.zeros((len(x_test), truth.shape[1]), dtype=float)
    with threadpool_limits(limits=2):
        for column in range(truth.shape[1]):
            keep = mask[:, column]
            if keep.any():
                estimator = Ridge(alpha=alpha).fit(x_train[keep].astype(float), truth[keep, column].astype(float))
                prediction[:, column] = estimator.predict(x_test.astype(float))
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--require-full-matrix", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT / "data") or out.is_relative_to(args.dataset.resolve()):
        raise ValueError("Use fresh non-data output")
    config = json.loads((args.runs[0] / "config.json").read_text(encoding="utf-8"))
    config["sro_label_revision"] = str(args.revision.resolve())
    manifest_hash = file_sha256(args.dataset / "manifests/task_samples.parquet")
    labels_hash = file_sha256(args.revision / "labels.parquet")
    input_hashes = {str(Path(__file__).resolve()): file_sha256(Path(__file__)),
                    str((args.dataset / "manifests/task_samples.parquet").resolve()): manifest_hash,
                    str((args.revision / "labels.parquet").resolve()): labels_hash}

    def bind(path):
        input_hashes[str(path.resolve())] = file_sha256(path)
        return path

    rows = load_validation_rows(args.dataset, config)
    truth, mask = np.stack([r.sro_values for r in rows]), np.stack([r.sro_mask for r in rows])
    split_cache = {}
    for split in config["split_columns"]:
        if pd.DataFrame([(r.parent_group_id, r.splits[split]) for r in rows], columns=["p", "s"]).groupby("p").s.nunique().max() > 1:
            raise ValueError("Parent leakage")
        test = np.array([r.splits[split] == "test" for r in rows])
        frame = pd.DataFrame([dict(xrd_sample_id=r.xrd_sample_id, parent_group_id=r.parent_group_id)
                              for r, keep in zip(rows, test) if keep])
        split_cache[split] = frame, truth[test], mask[test]
    errors, records = {}, []
    for directory in args.runs:
        verify_downloads(directory)
        protocol = json.loads(bind(directory / "protocol.json").read_text())
        if protocol["status"] != "complete" or not protocol["input_hashes_unchanged"]:
            raise ValueError("Unverified training batch")
        if protocol["source_manifest_sha256"] != manifest_hash or protocol["labels_sha256"] != labels_hash:
            raise ValueError("Training data or labels differ")
        run_config = json.loads(bind(directory / "config.json").read_text())
        for key in ["sro_targets", "elements", "model_budgets", "split_columns", "seeds"]:
            if config[key] != run_config[key]:
                raise ValueError(f"Training contract differs: {key}")
        for model, seed, split in protocol["runs"]:
            key = (split, model, seed)
            if key in errors:
                raise ValueError("Duplicate training case")
            name = stem(model, seed, split)
            frame = pd.read_parquet(bind(directory / "predictions" / f"{name}.parquet"))
            for column, value in [("model", model), ("seed", seed), ("split_column", split)]:
                if set(frame[column]) != {value}:
                    raise ValueError("Prediction metadata mismatch")
            expected, y, m = split_cache[split]
            errors[key], score = score_prediction(expected, frame, y, m)
            reported = json.loads(bind(directory / "predictions" / f"{name}.metrics.json").read_text())["sro"]
            check_score(score, reported)
            records.append(dict(split=split, model=model, seed=seed, **score))
    if args.require_full_matrix:
        require_spectral_matrix(errors, config)
    included_splits = sorted({key[0] for key in errors})
    if args.controls:
        control_plan = json.loads(bind(args.controls / "protocol.json").read_text())
        control_splits = {split for split, _ in control_plan["planned_runs"]}
        if not control_splits.issubset(config["split_columns"]):
            raise ValueError("Unexpected control split")
        included_splits = sorted(set(included_splits) | control_splits)
    ridge_protocol = json.loads(bind(args.ridge / "C/protocol.json").read_text())
    if ridge_protocol["source_manifest_sha256"] != manifest_hash or ridge_protocol.get("labels_sha256", labels_hash) != labels_hash or ridge_protocol["sro_targets"] != config["sro_targets"]:
        raise ValueError("Ridge data/label/target contract differs")
    ridge_config_path = bind(args.ridge / "C/config.json")
    if file_sha256(ridge_config_path) != ridge_protocol["config_sha256"]:
        raise ValueError("Ridge config hash mismatch")
    ridge_config = json.loads(ridge_config_path.read_text())
    for field in ["elements", "sro_targets", "split_columns"]:
        if ridge_config[field] != config[field]:
            raise ValueError("Ridge feature/target/split contract differs")
    composition = np.stack([r.composition for r in rows])
    ridge_reproduction = {}
    for split in included_splits:
        expected, y, m = split_cache[split]
        frame = pd.read_parquet(bind(args.ridge / "C" / f"{split}.parquet"))
        errors[(split, "composition-ridge", 2027)], score = score_prediction(expected, frame, y, m)
        check_score(score, json.loads(bind(args.ridge / "C" / f"{split}.metrics.json").read_text())["C"])
        train = np.array([r.splits[split] == "train" for r in rows])
        test = np.array([r.splits[split] == "test" for r in rows])
        reproduced = reproduce_ridge(composition[train], truth[train], mask[train], composition[test],
                                     float(ridge_config["model_budgets"]["composition_ridge"]["alpha"]))
        saved = np.stack(frame.set_index("xrd_sample_id").loc[expected.xrd_sample_id].sro_predictions)
        maximum_difference = float(np.max(np.abs(reproduced-saved)))
        if maximum_difference > 1e-7:
            raise ValueError(f"Ridge corrected-training reconstruction mismatch: {split} {maximum_difference}")
        ridge_reproduction[split] = maximum_difference
        records.append(dict(split=split, model="composition-ridge", seed=2027, **score))
    if args.controls:
        controls = json.loads(bind(args.controls / "protocol.json").read_text())
        if controls["status"] != "complete" or not controls["input_hashes_unchanged"]:
            raise ValueError("Incomplete controls")
        if controls["source_manifest_sha256"] != manifest_hash or controls["labels_sha256"] != labels_hash or controls["config"]["sro_targets"] != config["sro_targets"]:
            raise ValueError("Control target contract differs")
        for relative, digest in controls["output_hashes"].items():
            if file_sha256(bind(args.controls / relative)) != digest:
                raise ValueError("Control artifact hash mismatch")
        for split, feature in controls["planned_runs"]:
            if split not in included_splits:
                continue
            frame = pd.read_parquet(args.controls / "predictions" / f"{split}_{feature}.parquet")
            expected, y, m = split_cache[split]
            model, seed = f"control-xgboost-{feature}", controls["seed"]
            errors[(split, model, seed)], score = score_prediction(expected, frame, y, m)
            check_score(score, json.loads((args.controls / f"{split}_{feature}.metrics.json").read_text())["sro"])
            records.append(dict(split=split, model=model, seed=seed, **score))
    out.mkdir(parents=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "independent_scores.csv", index=False)
    comparisons = []
    matrices = {}
    for (split, model), part in table.groupby(["split", "model"]):
        seeds = sorted(part.seed.tolist())
        if model not in ["composition-ridge", "control-xgboost-C", "control-xgboost-X", "control-xgboost-CX"] and seeds != sorted(config["seeds"]):
            raise ValueError("Incomplete spectral seed matrix")
        matrix = pd.DataFrame({seed: errors[(split, model, seed)] for seed in seeds})
        if matrix.isna().any().any():
            raise ValueError("Parent mismatch across fitted seeds")
        matrix.to_csv(out / f"{split}_{model}.parents.csv", index_label="parent_group_id")
        matrices[(split, model)] = matrix
    for split in included_splits:
        models = sorted(model for case, model in matrices if case == split)
        pairs = [(model, "composition-ridge") for model in models if model != "composition-ridge"]
        if args.controls:
            pairs += [(model, "control-xgboost-C") for model in models if model not in ["composition-ridge", "control-xgboost-C"]]
        if "physics-transformer" in models and "transformer" in models:
            pairs.append(("physics-transformer", "transformer"))
        for candidate, reference in pairs:
            result = bootstrap_contrast(matrices[(split, candidate)], matrices[(split, reference)], args.bootstrap)
            comparisons.append(dict(split=split, candidate=candidate, reference=reference, **result))
            print(f"{split} {candidate} - {reference}: {result['delta_mae']:.8f}; parent CI {result['delta_ci95_parent']}; seed/parent CI {result['delta_ci95_seed_parent']}", flush=True)
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")
    if any(file_sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("Inputs changed during independent verification")
    protocol = dict(status="verified", input_hashes=input_hashes, verified_cases=len(records),
                    full_spectral_matrix_required=args.require_full_matrix,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                    ridge_refit_max_prediction_difference=ridge_reproduction,
                    case_counts=table.groupby(["split", "model"]).size().rename("count").reset_index().to_dict(orient="records"),
                    included_splits=included_splits, comparisons=len(comparisons),
                    evaluation="All frozen test sample identities, corrected masks; per-sample masked MAE then equal-weight parent means. Parent and crossed seed/parent bootstrap reported separately.",
                    scope="Clean simulated spectra; frozen original SRO-OOD membership. No causal/experimental SRO identifiability claim. Same-budget XGBoost controls are post-review exploratory diagnostics.")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"VERIFIED {len(records)} prediction artifacts and {len(comparisons)} contrasts", flush=True)


if __name__ == "__main__":
    main()
