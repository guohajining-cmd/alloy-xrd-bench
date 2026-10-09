"""Decompose the frozen HEA identifiability floor by nuisance factor and magnitude.

Read-only: re-renders the released observation parameters with individual
factors switched off or scaled, then recomputes the released ratio definition
(pair clean L1 / parent 95th-percentile observation-vs-clean L1). The released
frozen condition is re-rendered and must match the stored arrays exactly.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_hea_rebuttal_evidence import distribution, load_grids, sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FWHM = 0.1  # Gaussian FWHM of the released clean spectra.
FACTORS = ("zero_shift", "instrument_fwhm", "microstrain", "crystallite_size",
           "texture", "background", "poisson", "gaussian")
GROUPS = {
    "all": FACTORS,
    "peak_profile": ("zero_shift", "instrument_fwhm", "microstrain", "crystallite_size", "texture"),
    "broadening": ("instrument_fwhm", "microstrain", "crystallite_size"),
    "stochastic": ("poisson", "gaussian"),
    "all_except_background": tuple(f for f in FACTORS if f != "background"),
    "all_except_background_zero_shift": tuple(f for f in FACTORS if f not in {"background", "zero_shift"}),
}
SCALES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


def _load_renderer():
    """Replay the legacy release with the versioned renderer, without pymatgen."""
    source = ROOT / "alloy_xrd_bench/occupancy_expansion.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"render_observation", "_texture_factor", "_validate_renderer_version"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(selected) != len(names):
        raise ValueError("Renderer functions not found")
    constant_names = {"OBSERVATION_RENDERER_VERSION", "LEGACY_OBSERVATION_RENDERER_VERSION"}
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in constant_names for t in n.targets)]
    if len(constants) != len(constant_names):
        raise ValueError("Renderer versions not found")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *constants, *selected], type_ignores=[])
    namespace = {"math": math, "json": json, "np": np, "pd": pd}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

    def replay(two_theta, peaks, params):
        versioned = dict(params)
        versioned.setdefault("renderer_version", namespace["LEGACY_OBSERVATION_RENDERER_VERSION"])
        return namespace["render_observation"](two_theta, peaks, versioned)

    return replay


def condition_params(params: dict, active: tuple[str, ...], scale: float = 1.0) -> dict:
    """Keep active factors (magnitude scaled toward neutral), neutralize the rest."""
    out = dict(params)
    on = set(active)
    s = scale
    out["zero_shift_deg"] = params["zero_shift_deg"] * s if "zero_shift" in on else 0.0
    out["instrument_fwhm_deg"] = (CLEAN_FWHM + (params["instrument_fwhm_deg"] - CLEAN_FWHM) * s
                                  if "instrument_fwhm" in on else CLEAN_FWHM)
    out["microstrain_fraction"] = params["microstrain_fraction"] * s if "microstrain" in on else 0.0
    # Scherrer width is proportional to 1/size, so scaling the width by s divides size by s.
    out["crystallite_size_nm"] = (params["crystallite_size_nm"] / s
                                  if "crystallite_size" in on and s > 0 else None)
    out["march_dollase_r"] = params["march_dollase_r"] ** s if "texture" in on else 1.0
    out["background_fraction"] = params["background_fraction"] * s if "background" in on else 0.0
    # Poisson relative noise scales as 1/sqrt(poisson_scale), so noise*s means scale/s^2.
    out["poisson_scale"] = params["poisson_scale"] / s**2 if "poisson" in on and s > 0 else 0.0
    out["gaussian_noise_fraction"] = params["gaussian_noise_fraction"] * s if "gaussian" in on else 0.0
    return out


def conditions() -> list[tuple[str, tuple[str, ...], float]]:
    items = [("neutral", (), 1.0)]
    items += [(f"only_{f}", (f,), 1.0) for f in FACTORS]
    items += [(f"group_{g}", members, 1.0) for g, members in GROUPS.items()]
    items += [(f"scale_all_{s:g}", FACTORS, s) for s in SCALES if s != 1.0]
    items += [(f"scale_stochastic_{s:g}", GROUPS["stochastic"], s) for s in SCALES if s != 1.0]
    return items


def _worker(task):
    peak_path, sample_ids, params_list, clean, axis, stored = task
    render = _load_renderer()
    peaks = pd.read_parquet(peak_path)
    by_sample = {k: v for k, v in peaks.groupby("xrd_sample_id", sort=False)}
    conds = conditions()
    l1 = np.empty((len(sample_ids), len(conds)))
    reproduction_error = 0.0
    for i, (sample, params) in enumerate(zip(sample_ids, params_list)):
        table = by_sample[sample]
        truth = clean[i].astype(np.float64)
        frozen = render(axis, table, params)
        reproduction_error = max(reproduction_error, float(np.max(np.abs(frozen - stored[i]))))
        for j, (_, active, scale) in enumerate(conds):
            view = render(axis, table, condition_params(params, active, scale)).astype(np.float64)
            l1[i, j] = np.mean(np.abs(view - truth)) / 100
    return l1, reproduction_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT / "data" in output.parents:
        raise ValueError("Audit outputs must be outside source data")
    source = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    release = source.with_name("benchmark_v1_1")
    pairs = pd.read_parquet(release / "manifests/occupancy_pairs_calibrated.parquet")
    tasks = pd.read_parquet(source / "manifests/task_samples.parquet").set_index("structure_state_id", drop=False)
    observations = pd.read_parquet(release / "xrd/runs/hea_observation_v1_1/xrd_samples.parquet")

    state_ids = sorted(set(observations["structure_state_id"]))
    lookup = {s: i for i, s in enumerate(state_ids)}
    clean, _ = load_grids(source, tasks.loc[state_ids].reset_index(drop=True))
    stored, _ = load_grids(release, observations)
    with h5py.File(release / observations["grid_shard_path"].iloc[0], "r") as handle:
        axis = np.asarray(handle["two_theta"], dtype=float)

    work = []
    for peak_rel, frame in observations.groupby("peak_shard_path", sort=True):
        idx = frame.index.to_numpy()
        work.append((idx, (source / peak_rel,
                           frame["source_xrd_sample_id"].tolist(),
                           [json.loads(p) for p in frame["perturbation_json"]],
                           clean[[lookup[s] for s in frame["structure_state_id"]]],
                           axis, stored[idx])))
    conds = conditions()
    noise = np.empty((len(observations), len(conds)))
    reproduction = 0.0
    print(f"Rendering {len(observations)} views x {len(conds) + 1} conditions on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, ((idx, _), (l1, err)) in enumerate(zip(work, pool.map(_worker, [t for _, t in work])), 1):
            noise[idx] = l1
            reproduction = max(reproduction, err)
            print(f"  shard {done}/{len(work)}", flush=True)

    parent_idx = np.asarray([lookup[s] for s in pairs["parent_structure_id"]])
    cand_idx = np.asarray([lookup[s] for s in pairs["candidate_structure_state_id"]])
    pair_l1 = np.mean(np.abs(clean[parent_idx].astype(np.float64) - clean[cand_idx].astype(np.float64)), axis=1) / 100
    released_ratio = pairs["identifiability_ratio"].to_numpy()

    results, ratios = [], {}
    groups = observations["parent_group_id"].to_numpy()
    for j, (name, active, scale) in enumerate(conds):
        floor = pd.Series(noise[:, j]).groupby(groups).quantile(0.95)
        f = floor.loc[pairs["parent_group_id"]].to_numpy()
        ratio = np.divide(pair_l1, f, out=np.full_like(pair_l1, np.inf), where=f > 0)
        ratios[name] = ratio
        results.append({
            "condition": name, "active_factors": list(active), "scale": scale,
            "view_noise_l1": distribution(noise[:, j]),
            "parent_floor_q95": distribution(floor),
            "zero_floor_parents": int((floor <= 0).sum()),
            "ratio": distribution(ratio[np.isfinite(ratio)]),
            "pairs_ratio_ge_1": int((ratio >= 1).sum()),
            "pairs_ratio_ge_1_by_split": {k: int(v) for k, v in
                                          pd.Series(ratio >= 1).groupby(pairs["id_split"].to_numpy()).sum().items()},
        })
    neutral_floor_max = results[0]["view_noise_l1"]["quantiles"].get("1", float("nan"))
    frozen_all = next(r for r in results if r["condition"] == "group_all")
    frozen_ratio_error = float(np.max(np.abs(ratios["group_all"] - released_ratio)))
    checks = {
        "frozen_render_reproduces_stored_arrays": reproduction <= 1e-4,
        "neutral_render_matches_clean": neutral_floor_max <= 1e-5,
        "group_all_reproduces_released_ratio": frozen_ratio_error <= 1e-5,
        "group_all_reproduces_zero_of_4000": frozen_all["pairs_ratio_ge_1"] == int((released_ratio >= 1).sum()),
    }
    report = {
        "audit_scope": "new sensitivity analysis on released parameters; not a released benchmark result",
        "python": sys.executable, "python_version": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
        "script_sha256": sha256(Path(__file__)),
        "renderer_sha256": sha256(ROOT / "alloy_xrd_bench/occupancy_expansion.py"),
        "definitions": {
            "ratio": "clean pair mean-L1/100 divided by parent 95th-percentile of observation-vs-own-clean mean-L1/100 over 25 frozen views",
            "neutral": f"zero shift 0, instrument FWHM {CLEAN_FWHM} (clean), no strain/size/texture/background/noise",
            "scale_s": "each factor's deviation from neutral multiplied by s; Poisson scale divided by s^2; Scherrer size divided by s; March-Dollase r -> r^s",
            "seeds": "released per-view seeds and sampled parameters reused unchanged",
        },
        "checks": checks, "all_checks_pass": all(checks.values()),
        "frozen_reproduction_max_error": reproduction,
        "group_all_ratio_max_error_vs_released": frozen_ratio_error,
        "pair_l1": distribution(pair_l1),
        "conditions": results,
        "limits": [
            "Uses the released forward model (periodic reflections + Gaussian profiles); no diffuse scattering.",
            "The distance criterion is unchanged; an estimator that models nuisances could do better than this criterion suggests.",
            "Scaled conditions are hypothetical magnitudes, not calibrated experimental noise.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "decomposition.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    detail = pairs[["pair_id", "parent_group_id", "id_split"]].copy()
    detail["pair_l1"] = pair_l1
    for name, ratio in ratios.items():
        detail[f"ratio_{name}"] = ratio
    detail.to_csv(output / "pair_ratios.csv", index=False)
    summary = pd.DataFrame([{
        "condition": r["condition"], "floor_median": r["parent_floor_q95"]["quantiles"].get("0.5"),
        "ratio_max": r["ratio"]["quantiles"].get("1"), "pairs_ratio_ge_1": r["pairs_ratio_ge_1"],
    } for r in results])
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
