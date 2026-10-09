"""Read-only, commit-pinned comparison on published CoCrNi atomistic snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from ase.io import read
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.occupancy_pipeline import _sro
from scripts.audit_sro_independent import ase_sro, sha256

COMMIT = "92c9843eb36b2c4f640ab8917a0953342a085f02"
BASE = f"https://raw.githubusercontent.com/yifan-henry-cao/cSRO_lattice/{COMMIT}/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if ROOT / "data" == out or ROOT / "data" in out.parents:
        raise ValueError("Keep external data out of released benchmark")
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def fetch(path):
        target = out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            response = requests.get(BASE + path, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
        manifest.append({"url": BASE + path, "relative_path": path, "sha256": sha256(target)})
        return target

    fetch("LICENSE")
    fetch("README.md")
    fetch("plotting_codes/manuscript/auxiliary/08_WC_vs_T.py")
    avg = np.load(fetch("plotting_codes/manuscript/auxiliary/data/WC_params/WC_avg.npy"), allow_pickle=False)
    std = np.load(fetch("plotting_codes/manuscript/auxiliary/data/WC_params/WC_std.npy"), allow_pickle=False)
    pairs = ["Cr->Cr", "Cr->Co", "Cr->Ni", "Co->Co", "Co->Ni", "Ni->Ni"]
    rows = []
    for initial in ("RSS", "ordered"):
        for temperature in (400, 1000, 1600):
            path = f"simulations/MC_structures/{initial}_relaxation_20_1_{temperature}K.dump"
            # LAMMPS dump types are one-based; repository potential types are zero-based.
            atoms = read(fetch(path), format="lammps-dump-text", specorder=["Cr", "Co", "Ni"], index=-1)
            structure = Structure(atoms.cell.array, atoms.get_chemical_symbols(), atoms.get_scaled_positions())
            production = _sro(structure, "fcc")
            reference, diagnostics = ase_sro(structure, "fcc")
            mask_match = all(set(a) == set(b) for a, b in zip(production, reference))
            error = max(abs(a[k] - b[k]) for a, b in zip(production, reference) for k in a.keys() & b.keys())
            idx = (temperature - 300) // 100
            row = {"file": path, "initial_condition": initial, "temperature_K": temperature,
                   "natoms": len(structure), "composition": structure.composition.as_dict(),
                   "max_ase_error": error, "mask_match": mask_match, "coordination": diagnostics,
                   "first_shell": production[0], "second_shell": production[1],
                   "reported_ensemble_first_shell_six_pairs": dict(zip(pairs, avg[0, idx].tolist())),
                   "reported_ensemble_std_six_pairs": dict(zip(pairs, std[0, idx].tolist())),
                   "single_snapshot_minus_reported_ensemble": {p: production[0][p] - float(avg[0, idx, j]) for j, p in enumerate(pairs)},
                   "first_shell_abs_sum_full_matrix": sum(abs(v) for v in production[0].values())}
            rows.append(row)
            (out / "snapshots.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"{path}: N={len(structure)}, ASE max error={error:g}, masks match={mask_match}", flush=True)
    (out / "source_manifest.json").write_text(json.dumps({"commit": COMMIT, "files": manifest}, indent=2), encoding="utf-8")
    summary = {"snapshots": len(rows), "max_ase_error": max(r["max_ase_error"] for r in rows),
               "all_masks_match": all(r["mask_match"] for r in rows), "reference": "10.1038/s41467-026-75622-4",
               "commit": COMMIT, "selection": "Run 1 from both initial conditions at 400, 1000, 1600 K, fixed before inspecting results",
               "validated_scope": "Cross-implementation numerical agreement on external simulated structures, first and second shells",
               "not_validated": ["Exact reproduction of authors' ensemble averages: the averaging membership and snapshot-to-statistic mapping are not established",
                                 "Experimental SRO ground truth or XRD inversion", "Thermal lattice-response prediction by our unrelaxed counterfactual generator"],
               "reported_ensemble_shape": list(avg.shape), "atom_type_mapping": {"1": "Cr", "2": "Co", "3": "Ni"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["max_ase_error"] > 1e-10 or not summary["all_masks_match"]:
        raise SystemExit("External structure arithmetic comparison failed")


if __name__ == "__main__":
    main()
