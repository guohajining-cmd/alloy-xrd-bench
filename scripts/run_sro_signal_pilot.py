"""Matched Ridge diagnostics with corrected labels and untouched parent splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import read_grid_row, vectorize_sro
from scripts.build_corrected_sro_manifest import sha256


def masked_parent_mae(truth, prediction, mask):
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every scored parent needs at least one eligible label")
    if not np.isfinite(prediction[mask]).all():
        raise ValueError("Nonfinite scored predictions")
    return np.where(mask, np.abs(prediction - truth), 0).sum(axis=1) / counts


def paired_bootstrap(reference, candidate, seed, repeats=2000):
    delta = candidate - reference
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(axis=1)
    return {"delta_mae": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "parents": len(delta), "bootstrap_repeats": repeats,
            "direction": "candidate minus reference; negative favors candidate"}


def fit_grouped_ridge(x_train, y_train, mask_train, x_eval, alpha):
    prediction = np.full((len(x_eval), y_train.shape[1]), np.nan)
    groups = defaultdict(list)
    for column in range(y_train.shape[1]):
        groups[mask_train[:, column].tobytes()].append(column)
    for columns in groups.values():
        available = mask_train[:, columns[0]]
        if not available.any():
            continue
        model = Ridge(alpha=alpha, solver="cholesky")
        model.fit(x_train[available], y_train[np.ix_(available, columns)])
        prediction[:, columns] = model.predict(x_eval).reshape(len(x_eval), len(columns))
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[.01, .1, 1., 10., 100., 1000., 10000., 100000., 1000000.])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if ROOT / "data" == out or ROOT / "data" in out.parents:
        raise ValueError("Do not overwrite released data")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("Use a new output directory to preserve results")
    audit = json.loads((args.audit_dir / "summary.json").read_text())
    if audit["errors"]:
        raise ValueError("Label audit must pass before model fitting")
    source = args.audit_dir / "corrected_sample.parquet"
    frame = pd.read_parquet(source)
    frame = frame[frame.pilot].reset_index(drop=True)
    if frame.parent_group_id.duplicated().any():
        raise ValueError("Pilot requires exactly one canonical view per parent")
    if (frame.groupby("structure_sha256").id_split.nunique() > 1).any():
        raise ValueError("Exact structure hash leakage across pilot splits")
    config = json.loads((ROOT / "configs/hea_validation_v1.json").read_text())
    targets, elements = config["sro_targets"], config["elements"]
    labels = [vectorize_sro(row.corrected_shell1_json, row.corrected_shell2_json, targets) for row in frame.itertuples()]
    y, mask = np.stack([x[0] for x in labels]), np.stack([x[1] for x in labels])
    train = (frame.id_split == "train").to_numpy()
    validation = (frame.id_split == "validation").to_numpy()
    test = (frame.id_split == "test").to_numpy()
    eligible = mask[train].sum(axis=0) >= 20
    # Eligibility depends only on training support and is identical for all methods.
    y, mask = y[:, eligible].astype(float), mask[:, eligible]
    eligible_rows = mask.sum(axis=1) > 0
    validation &= eligible_rows
    test &= eligible_rows
    alphas = args.alpha_grid
    protocol = {"sample_sha256": sha256(source), "script_sha256": sha256(Path(__file__)), "seed": 20260930,
                "split_counts": frame.id_split.value_counts().to_dict(),
                "train_minimum_target_support": 20,
                "evaluated_targets": [t for t, keep in zip(targets, eligible) if keep],
                "excluded_targets": [t for t, keep in zip(targets, eligible) if not keep],
                "excluded_parent_rows_no_eligible_target": int((~eligible_rows).sum()),
                "features": {"composition": elements,
                             "composition_lattice_scale": elements + ["cuberoot(volume_per_atom)"],
                "full_xrd": "all 4001 clean intensity bins, no PCA",
                "composition_xrd": "composition + all 4001 clean intensity bins",
                             "composition_lattice_scale_xrd": "composition + cubic density scale + all bins"},
                "lattice_caveat": "Volume-per-atom cube root is an oracle structural scale, not complete lattice metrics or an experimentally estimated lattice parameter.",
                "estimator": "Ridge with intercept, target-specific valid training rows; identical training-only feature standardization",
                "alpha_grid": alphas, "selection_metric": "validation parent-macro masked MAE, one global alpha per feature set",
                "test_protocol": "No test-based model selection; bootstrap conditional on this trained pilot, not retraining uncertainty",
                "qualification": "Exploratory small ID-split diagnostic; not replacement leaderboard or diffuse-scattering validation"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    root = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    spectra = []
    for index, row in enumerate(frame.itertuples()):
        spectra.append(read_grid_row(root / row.grid_shard_path, int(row.grid_row_index), row.xrd_sample_id))
        if (index + 1) % 250 == 0:
            print(f"Loaded {index + 1}/{len(frame)} spectra", flush=True)
    spectra = np.stack(spectra).astype(float)
    composition = np.array([[json.loads(s)[e] for e in elements] for s in frame.composition_json])
    lattice = np.cbrt(frame.volume_per_atom.to_numpy())[:, None]
    features = {"composition": composition, "composition_lattice_scale": np.c_[composition, lattice],
                "full_xrd": spectra, "composition_xrd": np.c_[composition, spectra],
                "composition_lattice_scale_xrd": np.c_[composition, lattice, spectra]}
    report, per_parent, arrays = {}, frame.loc[test, ["structure_state_id", "parent_group_id"]].copy(), {}
    with threadpool_limits(limits=2):
        for name, x in features.items():
            scaler = StandardScaler().fit(x[train])
            x = scaler.transform(x)
            validation_scores = []
            for alpha in alphas:
                pred = fit_grouped_ridge(x[train], y[train], mask[train], x[validation], alpha)
                loss = float(masked_parent_mae(y[validation], pred, mask[validation]).mean())
                validation_scores.append(loss)
                print(f"{name}: alpha={alpha}, validation MAE={loss:.6f}", flush=True)
            best_index = int(np.argmin(validation_scores))
            best_alpha = alphas[best_index]
            pred = fit_grouped_ridge(x[train], y[train], mask[train], x[test], best_alpha)
            losses = masked_parent_mae(y[test], pred, mask[test])
            per_parent[name] = losses
            arrays[f"pred_{name}"] = pred
            report[name] = {"selected_alpha": best_alpha, "validation_mae_grid": validation_scores,
                            "test_parent_macro_mae": float(losses.mean()), "test_parents": int(test.sum())}
    comparisons = {}
    for reference, candidate in [("composition", "composition_lattice_scale"), ("composition", "full_xrd"),
                                 ("composition", "composition_xrd"),
                                 ("composition_lattice_scale", "composition_lattice_scale_xrd")]:
        comparisons[f"{candidate}_minus_{reference}"] = paired_bootstrap(
            per_parent[reference].to_numpy(), per_parent[candidate].to_numpy(), 20260930)
    per_parent.to_csv(out / "test_parent_errors.csv", index=False)
    np.savez_compressed(out / "test_predictions.npz", y_true=y[test], mask=mask[test],
                        structure_state_id=frame.loc[test, "structure_state_id"].to_numpy(dtype=str), **arrays)
    result = {"models": report, "paired_comparisons": comparisons,
              "evaluated_targets": int(eligible.sum()), "valid_test_entries": int(mask[test].sum())}
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
