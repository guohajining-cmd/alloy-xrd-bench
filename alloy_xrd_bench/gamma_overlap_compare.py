"""Paired parent-level bootstrap comparisons for Gamma-Decomp baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compare_parent_bootstrap(
    reference_score_dir: str | Path,
    candidate_score_dir: str | Path,
    output_dir: str | Path,
    *,
    metric: str = "component_f1",
    replicates: int = 2000,
    seed: int = 2027,
    minimum_improvement: float = 0.05,
) -> dict[str, Any]:
    reference = pd.read_parquet(Path(reference_score_dir) / "per_parent_pair.parquet")
    candidate = pd.read_parquet(Path(candidate_score_dir) / "per_parent_pair.parquet")
    for name, frame in (("reference", reference), ("candidate", candidate)):
        if frame["parent_pair_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate parent_pair_id values")
        if metric not in frame:
            raise ValueError(f"{name} does not contain metric {metric!r}")
    reference = reference[["parent_pair_id", metric]].rename(columns={metric: "reference"})
    candidate = candidate[["parent_pair_id", metric]].rename(columns={metric: "candidate"})
    paired = reference.merge(candidate, on="parent_pair_id", how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("reference and candidate parent groups do not match exactly")
    paired = paired.drop(columns="_merge").sort_values("parent_pair_id").reset_index(drop=True)
    if not np.isfinite(paired[["reference", "candidate"]].to_numpy(dtype=float)).all():
        raise ValueError("bootstrap metric values must be finite")
    differences = paired["candidate"].to_numpy(dtype=float) - paired["reference"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        sampled = rng.integers(0, len(differences), size=len(differences))
        bootstrap[index] = float(differences[sampled].mean())
    ci = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    improvement = float(differences.mean())
    report = {
        "schema_version": "gamma-decomp-parent-bootstrap-v1",
        "metric": metric,
        "parent_group_count": len(paired),
        "reference_mean": float(paired["reference"].mean()),
        "candidate_mean": float(paired["candidate"].mean()),
        "absolute_improvement": improvement,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_ci_95": ci,
        "minimum_improvement": float(minimum_improvement),
        "five_point_gate_passed": bool(
            improvement >= float(minimum_improvement) and ci[0] > 0.0
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"replicate": np.arange(len(bootstrap)), "improvement": bootstrap}).to_parquet(
        output / "bootstrap_samples.parquet", index=False
    )
    paired.to_parquet(output / "paired_parent_metrics.parquet", index=False)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return report
