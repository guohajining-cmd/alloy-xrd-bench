"""Summarize matched SRO phases with phase-specific Ridge references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rescore_sro_matrix import stem
from alloy_xrd_bench.sro_revision import file_sha256


def compare_to_ridge(candidate, ridge, iterations=2000, seed=2027):
    if not candidate.index.is_unique or not ridge.index.is_unique or set(candidate.index) != set(ridge.index):
        raise ValueError("Candidate and Ridge must have identical unique parent IDs")
    candidate = candidate.sort_index().to_numpy(dtype=float)
    ridge = ridge.sort_index().to_numpy(dtype=float)
    if not len(ridge) or not np.isfinite(candidate).all() or not np.isfinite(ridge).all() or ridge.mean() <= 0 or np.any(candidate < 0) or np.any(ridge < 0):
        raise ValueError("Invalid paired errors or zero Ridge reference")
    rng = np.random.default_rng(seed)
    deltas, percentages = np.empty(iterations), np.empty(iterations)
    for index in range(iterations):
        selected = rng.integers(0, len(ridge), len(ridge))
        c, r = candidate[selected].mean(), ridge[selected].mean()
        if r <= 0:
            raise ValueError("Zero bootstrap Ridge mean")
        deltas[index], percentages[index] = c-r, 100*(1-c/r)
    relative = float(100*(1-candidate.mean()/ridge.mean()))
    return dict(parent_count=len(ridge), model_mae=float(candidate.mean()), ridge_mae=float(ridge.mean()),
                delta_mae=float((candidate-ridge).mean()), delta_mae_ci95=np.quantile(deltas, [.025, .975]).tolist(),
                relative_improvement_percent=relative, relative_improvement_ci95=np.quantile(percentages, [.025, .975]).tolist(),
                S_SRO=max(0., relative), bootstrap_improvement_fraction=float((deltas < 0).mean()),
                iterations=iterations, seed=seed,
                interpretation="Paired-parent bootstrap conditional on these fitted seeds; improvement fraction is not a p-value or posterior probability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-ab", type=Path, required=True)
    parser.add_argument("--corrected-abc", type=Path, action="append")
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    sources = [(args.historical_ab, ["A", "B"])]
    if args.corrected_abc:
        sources.extend((directory, ["C"]) for directory in args.corrected_abc)
    records, comparisons, inputs = [], [], {}
    for directory, phases in sources:
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if protocol["status"] != "verified":
            raise ValueError("Matrix is not verified")
        inputs[str(directory)] = file_sha256(directory / "protocol.json")
        scores = pd.read_csv(directory / "scores.csv")
        scores = scores[scores.phase.isin(phases)]
        for (split, model, phase), part in scores.groupby(["split_column", "model", "phase"], sort=False):
            if phase == "C":
                reference = pd.read_csv(args.ridge / "C" / f"{split}.parent_errors.csv").set_index("parent_group_id")["C"]
            else:
                reference = pd.read_csv(args.historical_ab / f"{stem('composition-ridge', 2027, split)}.parents.csv").set_index("parent_group_id")[phase]
            errors = []
            for seed in sorted(part.seed.unique()):
                errors.append(pd.read_csv(directory / f"{stem(model, seed, split)}.parents.csv").set_index("parent_group_id")[phase])
            if any(set(s.index) != set(reference.index) for s in errors):
                raise ValueError("Different parent membership across seeds")
            mean_parent_error = pd.concat(errors, axis=1).mean(axis=1)
            comparison = compare_to_ridge(mean_parent_error, reference)
            comparison.update(model=model, split_column=split, phase=phase, seed_count=len(part))
            comparisons.append(comparison)
            records.append(dict(model=model, split_column=split, phase=phase, seed_count=len(part),
                                mae_mean=float(part.mae.mean()), seed_sd=float(part.mae.std(ddof=1)) if len(part)>1 else 0.,
                                rmse_mean=float(part.rmse.mean()), shell1_spearman_mean=float(part.spearman_shell1.mean()),
                                shell2_spearman_mean=float(part.spearman_shell2.mean()), S_SRO=comparison["S_SRO"],
                                relative_improvement_percent=comparison["relative_improvement_percent"],
                                parent_count=comparison["parent_count"]))
    if args.corrected_abc:
        summaries = json.loads((args.ridge / "C/summary.json").read_text(encoding="utf-8"))
        for split in sorted({r["split_column"] for r in records if r["phase"] == "C"}):
            value = summaries[split]["C"]
            records.append(dict(model="composition-ridge", split_column=split, phase="C", seed_count=1,
                                mae_mean=value["masked_parent_macro_mae"], seed_sd=0., rmse_mean=value["rmse"],
                                shell1_spearman_mean=value["shell_spearman"]["shell1"],
                                shell2_spearman_mean=value["shell_spearman"]["shell2"], S_SRO=0.,
                                relative_improvement_percent=0., parent_count=value["parent_count"]))
    table = pd.DataFrame(records)
    if table.duplicated(["model", "split_column", "phase"]).any():
        raise ValueError("Duplicate model/split/phase across supplied matrices")
    table.to_csv(out / "leaderboard.csv", index=False)
    (out / "ridge_comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")
    metadata = dict(status="verified", inputs=inputs, score_policy="S_SRO clips negative relative improvements at zero; unclipped percentages also retained",
                    reference_policy="A versus old Ridge; B versus rescored old Ridge; C versus corrected-retrained Ridge",
                    uncertainty_policy="Seed mean/SD separate from paired-parent bootstrap; no correction for multiple exploratory comparisons",
                    evidence_boundary="Only the listed completed phases and splits; no experimental SRO observability claim")
    (out / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table[table.split_column.eq("id_split")].to_string(index=False))


if __name__ == "__main__":
    main()
