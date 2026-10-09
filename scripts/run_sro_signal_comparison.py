"""Paired C, X, and C+X SRO diagnostics on frozen HEA parent splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import read_grid_row, vectorize_sro
from scripts.build_corrected_sro_manifest import DATA_ROOT, sha256
from scripts.run_sro_signal_pilot import fit_grouped_ridge, masked_parent_mae, paired_bootstrap


SPLITS = ("id_split", "composition_ood_split", "chemistry_ood_split", "sro_ood_split")
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)


def select_sample(frame: pd.DataFrame, split_column: str, seed: int, counts: dict[str, int]) -> pd.DataFrame:
    if split_column not in SPLITS:
        raise ValueError(f"unsupported split column {split_column!r}")
    if (frame.groupby("parent_group_id")[split_column].nunique() > 1).any():
        raise ValueError("parent group crosses split boundaries")
    eligible = frame[
        (frame.fidelity == "source_relaxed_dft")
        & frame.parent_lattice.isin(("fcc", "bcc"))
        & (frame.nelements > 1)
    ].copy()
    eligible["selection_hash"] = eligible.parent_group_id.map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    selected = []
    for split in ("train", "validation", "test"):
        subset = eligible[eligible[split_column] == split].sort_values(
            ["selection_hash", "structure_state_id"]
        ).drop_duplicates("parent_group_id").head(counts[split])
        if len(subset) != counts[split]:
            raise ValueError(f"not enough {split_column} {split} parents")
        selected.append(subset)
    result = pd.concat(selected, ignore_index=True)
    if result.parent_group_id.duplicated().any() or result.structure_state_id.duplicated().any():
        raise ValueError("selected sample has duplicate parent or structure")
    return result


def load_frame(sidecar_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_path = DATA_ROOT / "manifests/task_samples.parquet"
    structures_path = DATA_ROOT / "manifests/structures.parquet"
    composition_path = DATA_ROOT / "manifests/composition_only_features.parquet"
    paths = (manifest_path, structures_path, composition_path, sidecar_path)
    manifest = pd.read_parquet(manifest_path)
    structures = pd.read_parquet(structures_path, columns=["dataset_structure_id", "nelements"])
    labels = pd.read_parquet(sidecar_path, columns=[
        "structure_state_id", "corrected_shell1_json", "corrected_shell2_json"
    ])
    config = json.loads((ROOT / "configs/hea_validation_v1.json").read_text(encoding="utf-8"))
    fraction_columns = [f"fraction_{element}" for element in config["elements"]]
    composition = pd.read_parquet(composition_path, columns=["structure_state_id", *fraction_columns])
    frame = manifest.merge(
        structures, left_on="parent_structure_id", right_on="dataset_structure_id", validate="many_to_one"
    ).merge(labels, on="structure_state_id", validate="one_to_one").merge(
        composition, on="structure_state_id", validate="one_to_one"
    )
    if len(frame) != len(manifest):
        raise ValueError("corrected labels or composition features do not cover every manifest row")
    return frame, {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def evaluate_split(
    selected: pd.DataFrame, split_column: str, out: Path, source_hashes: dict[str, str],
    alpha_grid: tuple[float, ...], bootstrap_repeats: int, seed: int,
    spectra_cache: dict[str, np.ndarray],
) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    selected[["structure_state_id", "parent_group_id", split_column, "selection_hash"]].to_csv(
        out / "selection.csv", index=False
    )
    config = json.loads((ROOT / "configs/hea_validation_v1.json").read_text(encoding="utf-8"))
    targets, elements = config["sro_targets"], config["elements"]
    label_pairs = [vectorize_sro(row.corrected_shell1_json, row.corrected_shell2_json, targets)
                   for row in selected.itertuples(index=False)]
    truth = np.stack([pair[0] for pair in label_pairs]).astype(float)
    mask = np.stack([pair[1] for pair in label_pairs])
    train = (selected[split_column] == "train").to_numpy()
    validation = (selected[split_column] == "validation").to_numpy()
    test = (selected[split_column] == "test").to_numpy()
    eligible_targets = mask[train].sum(axis=0) >= 20
    truth, mask = truth[:, eligible_targets], mask[:, eligible_targets]
    valid_rows = mask.any(axis=1)
    validation &= valid_rows
    test &= valid_rows
    if not eligible_targets.any() or not validation.any() or not test.any():
        raise ValueError(f"{split_column}: insufficient label support")
    spectra = []
    root = DATA_ROOT.resolve()
    for index, row in enumerate(selected.itertuples(index=False), 1):
        sample_id = row.xrd_sample_id
        if sample_id not in spectra_cache:
            path = (root / row.grid_shard_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("grid shard path escapes dataset root")
            spectra_cache[sample_id] = read_grid_row(path, int(row.grid_row_index), sample_id)
        spectra.append(spectra_cache[sample_id])
        if index % 500 == 0:
            print(f"{split_column}: loaded {index}/{len(selected)} spectra", flush=True)
    xrd = np.stack(spectra).astype(float)
    composition = selected[[f"fraction_{element}" for element in elements]].to_numpy(dtype=float)
    features = {"composition": composition, "full_xrd": xrd,
                "composition_xrd": np.c_[composition, xrd]}
    losses = selected.loc[test, ["structure_state_id", "parent_group_id"]].copy()
    predictions = {}
    models = {}
    with threadpool_limits(limits=2):
        for name, raw in features.items():
            scaler = StandardScaler().fit(raw[train])
            x = scaler.transform(raw)
            validation_scores = []
            for alpha in alpha_grid:
                candidate = fit_grouped_ridge(x[train], truth[train], mask[train], x[validation], alpha)
                score = float(masked_parent_mae(truth[validation], candidate, mask[validation]).mean())
                validation_scores.append(score)
                print(f"{split_column} {name}: alpha={alpha:g}, validation MAE={score:.6f}", flush=True)
            best = int(np.argmin(validation_scores))
            prediction = fit_grouped_ridge(x[train], truth[train], mask[train], x[test], alpha_grid[best])
            per_parent = masked_parent_mae(truth[test], prediction, mask[test])
            losses[name] = per_parent
            predictions[f"pred_{name}"] = prediction
            models[name] = {"selected_alpha": alpha_grid[best], "validation_mae_grid": validation_scores,
                            "test_parent_macro_mae": float(per_parent.mean())}
    comparisons = {}
    for reference, candidate in (("composition", "full_xrd"), ("composition", "composition_xrd")):
        comparisons[f"{candidate}_minus_{reference}"] = paired_bootstrap(
            losses[reference].to_numpy(), losses[candidate].to_numpy(), seed, bootstrap_repeats
        )
    losses.to_csv(out / "test_parent_errors.csv", index=False)
    np.savez_compressed(out / "test_predictions.npz", y_true=truth[test], mask=mask[test],
                        structure_state_id=selected.loc[test, "structure_state_id"].to_numpy(dtype=str), **predictions)
    protocol = {
        "qualification": "Exploratory corrected-label clean-spectrum comparison; not a replacement full-data leaderboard",
        "split_column": split_column, "selection_seed": seed,
        "sample_counts": selected[split_column].value_counts().to_dict(),
        "selection_sha256": sha256(out / "selection.csv"), "source_hashes": source_hashes,
        "targets": [target for target, keep in zip(targets, eligible_targets) if keep],
        "train_minimum_target_support": 20,
        "excluded_validation_parents_no_target": int(((selected[split_column] == "validation") & ~valid_rows).sum()),
        "excluded_test_parents_no_target": int(((selected[split_column] == "test") & ~valid_rows).sum()),
        "features": {"composition": elements, "full_xrd": "4001 clean intensity bins",
                     "composition_xrd": "composition plus 4001 clean intensity bins"},
        "estimator": "Ridge with per-target valid training rows; train-only feature standardization",
        "alpha_grid": list(alpha_grid), "selection_metric": "validation parent-macro masked MAE",
        "test_protocol": "same parent/test targets for all methods; parent-paired bootstrap conditional on fitted models",
        "limitations": ["Corrected labels are a sidecar; published labels and models are unchanged.",
                        "Frozen SRO OOD groups reflect the original sparse labels.",
                        "No explicit diffuse scattering or experimental SRO labels are tested."],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = {"models": models, "paired_comparisons": comparisons,
              "evaluated_targets": int(eligible_targets.sum()),
              "valid_test_entries": int(mask[test].sum()), "test_parents": int(test.sum())}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{split_column}: {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=250)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out == ROOT / "data" or ROOT / "data" in out.parents:
        raise ValueError("output must not overwrite released data")
    if out.exists():
        raise ValueError("use a fresh output directory")
    if any(value <= 0 for value in (*args.alpha_grid, args.train, args.validation, args.test, args.bootstrap_repeats)):
        raise ValueError("counts and regularization strengths must be positive")
    frame, source_hashes = load_frame(args.label_manifest.resolve())
    out.mkdir(parents=True)
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    spectra_cache: dict[str, np.ndarray] = {}
    results = {}
    for split in args.splits:
        selected = select_sample(frame, split, args.seed, counts)
        results[split] = evaluate_split(selected, split, out / split, source_hashes,
                                        tuple(args.alpha_grid), args.bootstrap_repeats, args.seed, spectra_cache)
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
