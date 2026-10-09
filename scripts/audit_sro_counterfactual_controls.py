"""Verify fixed-cell controls and independently relabel a frozen pair sample."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.occupancy_pipeline import _sro
from scripts.audit_sro_independent import ase_sro, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if ROOT / "data" == out or ROOT / "data" in out.parents:
        raise ValueError("Do not overwrite source data")
    out.mkdir(parents=True, exist_ok=True)
    root = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    source = root / "derived/occupancy/hea_sro_cf_pilot_v2/structures.parquet"
    frame = pd.read_parquet(source)
    sources = pd.read_parquet(root / "manifests/structures.parquet").set_index("dataset_structure_id")
    parents = sorted(frame.parent_structure_id.unique())
    rows = sources.loc[parents, "source_row"].tolist()
    payload = pd.read_parquet(root / "xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet",
                              filters=[("source_row", "in", rows)]).set_index("source_row")
    parent_structures = {p: Structure.from_dict(ast.literal_eval(payload.loc[sources.loc[p, "source_row"], "structure_as_dict"])) for p in parents}
    selected = set(sorted(frame.structure_state_id, key=lambda s: hashlib.sha256(f"20260930:{s}".encode()).hexdigest())[:100])
    (out / "protocol.json").write_text(json.dumps({"source_sha256": sha256(source), "sample_ids": sorted(selected),
                                                   "selection": "Lowest SHA256(20260930:state_id); 100 of 4000 pairs",
                                                   "script_sha256": sha256(__file__)}, indent=2), encoding="utf-8")
    details, label_checks = [], []
    for index, row in enumerate(frame.itertuples()):
        parent = parent_structures[row.parent_structure_id]
        child = Structure.from_dict(json.loads(row.structure_json))
        same_composition = Counter(s.specie.symbol for s in parent) == Counter(s.specie.symbol for s in child)
        cell_delta = float(np.max(np.abs(parent.lattice.matrix - child.lattice.matrix)))
        position_delta = child.frac_coords - parent.frac_coords
        position_delta -= np.round(position_delta)
        max_position_delta = float(np.max(np.abs(position_delta)))
        details.append({"structure_state_id": row.structure_state_id, "parent_structure_id": row.parent_structure_id,
                        "same_composition": same_composition, "max_cell_delta_A": cell_delta,
                        "max_fractional_position_delta": max_position_delta})
        if row.structure_state_id in selected:
            p_labels, c_labels = _sro(parent, row.parent_lattice), _sro(child, row.parent_lattice)
            p_ref, _ = ase_sro(parent, row.parent_lattice)
            c_ref, _ = ase_sro(child, row.parent_lattice)
            max_error, mask_errors = 0., 0
            for actual, ref in [(p_labels, p_ref), (c_labels, c_ref)]:
                for a, b in zip(actual, ref):
                    mask_errors += int(set(a) != set(b))
                    max_error = max(max_error, max((abs(a[k] - b[k]) for k in a.keys() & b.keys()), default=0.))
            changes = [abs(a[k] - b[k]) for a, b in zip(p_labels, c_labels) for k in a.keys() & b.keys()]
            label_checks.append({"structure_state_id": row.structure_state_id, "max_ase_error": max_error,
                                 "mask_errors": mask_errors, "sro_pair_mean_abs_change": float(np.mean(changes)),
                                 "sro_pair_max_abs_change": max(changes)})
        if (index + 1) % 1000 == 0:
            print(f"Verified fixed-cell geometry {index + 1}/{len(frame)}", flush=True)
    details, checks = pd.DataFrame(details), pd.DataFrame(label_checks)
    details.to_csv(out / "geometry_checks.csv", index=False)
    checks.to_csv(out / "independent_pair_labels.csv", index=False)
    previous = ROOT / "output/rebuttal_2026_09_28/hea_audit/pair_audit.csv"
    pairs = pd.read_csv(previous)
    if set(pairs.candidate_structure_state_id) != set(frame.structure_state_id):
        raise ValueError("Previous spectral audit covers a different set")
    report = {"pairs": len(details), "composition_mismatch": int((~details.same_composition).sum()),
              "max_cell_delta_A": float(details.max_cell_delta_A.max()),
              "max_fractional_position_delta": float(details.max_fractional_position_delta.max()),
              "label_pair_sample": len(checks), "max_ase_error": float(checks.max_ase_error.max()),
              "label_mask_errors": int(checks.mask_errors.sum()),
              "pairs_with_nonzero_sro_change_in_sample": int((checks.sro_pair_max_abs_change > 1e-10).sum()),
              "median_sample_sro_mean_abs_change": float(checks.sro_pair_mean_abs_change.median()),
              "prior_spectral_audit_sha256": sha256(previous),
              "clean_mean_abs_spectral_difference_quantiles": pairs.recomputed_pair_l1.quantile([0, .5, .95, 1]).to_dict(),
              "synthetic_nuisance_ratio_quantiles": pairs.recomputed_ratio.quantile([0, .5, .95, 1]).to_dict(),
              "caveat": "Spectral distances reuse the prior independently reconstructed clean-spectrum audit. Noise ratios use the synthetic nuisance protocol, not experimental noise. These controls do not include relaxation or explicit diffuse scattering."}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["composition_mismatch"] or report["max_cell_delta_A"] > 1e-10 or report["max_fractional_position_delta"] > 1e-10 or report["label_mask_errors"] or report["max_ase_error"] > 1e-10:
        raise SystemExit("Control audit failed")


if __name__ == "__main__":
    main()
