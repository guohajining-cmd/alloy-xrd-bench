"""Independent ASE label audit and a frozen, non-destructive rebuttal sample."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.occupancy_pipeline import _sro


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implied_zero_pairs(values, elements):
    # A stored outgoing edge proves the central species has a nonempty shell.
    centers = {key.split("->")[0] for key in values}
    return sorted(f"{a}->{b}" for a in centers for b in elements if f"{a}->{b}" not in values)


def ase_sro(structure, parent):
    """Independently count periodic ASE edges using the published radial rule.

    Shares the scientific shell convention, not pymatgen's neighbor algorithm.
    Probabilities are pooled over directed edges, not averaged per atom.
    """
    if parent not in ("fcc", "bcc") or len(structure.composition.elements) < 2:
        return ({}, {}), []
    symbols = np.array([site.specie.symbol for site in structure])
    atoms = Atoms(symbols.tolist(), scaled_positions=structure.frac_coords,
                  cell=structure.lattice.matrix, pbc=True)
    volume_per_atom = atoms.get_volume() / len(atoms)
    if parent == "fcc":
        lattice = np.cbrt(4 * volume_per_atom)
        first, second = lattice / np.sqrt(2), lattice
    else:
        lattice = np.cbrt(2 * volume_per_atom)
        first, second = lattice * np.sqrt(3) / 2, lattice
    boundary, outer = (first + second) / 2, second * 1.15
    i, j, distance = neighbor_list("ijd", atoms, outer + 1e-8)
    elements, counts = np.unique(symbols, return_counts=True)
    fractions = dict(zip(elements, counts / len(atoms)))
    output, diagnostics = [], []
    for lower, upper in [(0, boundary), (boundary, outer)]:
        selected = (distance > lower + 1e-8) & (distance <= upper)
        centers, neighbors = symbols[i[selected]], symbols[j[selected]]
        coord = np.bincount(i[selected], minlength=len(atoms))
        table = {}
        for element in elements:
            adjacent = neighbors[centers == element]
            if not len(adjacent):
                continue
            for other in elements:
                table[f"{element}->{other}"] = float(1 - np.count_nonzero(adjacent == other) / len(adjacent) / fractions[other])
        output.append(table)
        diagnostics.append({"coordination_min": int(coord.min()), "coordination_max": int(coord.max()),
                            "coordination_mean": float(coord.mean()), "zero_neighbor_atoms": int((coord == 0).sum())})
    return tuple(output), diagnostics


def stored_scope(structures, tasks):
    element_map = structures.set_index("dataset_structure_id")["elements_json"].to_dict()
    details = []
    for row in tasks.itertuples(index=False):
        elements = json.loads(element_map[row.parent_structure_id])
        if row.parent_lattice not in ("fcc", "bcc") or len(elements) < 2:
            continue
        for shell in (1, 2):
            values = json.loads(getattr(row, f"sro_shell{shell}_json"))
            missing = implied_zero_pairs(values, elements)
            undefined_centers = set(elements) - {k.split("->")[0] for k in values}
            if missing or undefined_centers:
                details.append({"structure_state_id": row.structure_state_id, "fidelity": row.fidelity,
                                "id_split": row.id_split, "shell": shell, "added_keys": len(missing),
                                "undefined_center_keys": len(undefined_centers) * len(elements)})
    frame = pd.DataFrame(details)
    summary = {}
    for fidelity, part in frame.groupby("fidelity"):
        summary[fidelity] = {"affected_states": int(part.loc[part.added_keys > 0, "structure_state_id"].nunique()),
                             "inferred_zero_keys": int(part.added_keys.sum()),
                             "keys_left_undefined": int(part.undefined_center_keys.sum())}
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=250)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if ROOT / "data" == out or ROOT / "data" in out.parents:
        raise ValueError("Source data must not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.csv").exists():
        raise ValueError("Use a fresh output directory to preserve the frozen selection")
    root = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    structures_path = root / "manifests/structures.parquet"
    tasks_path = root / "manifests/task_samples.parquet"
    structures, tasks = pd.read_parquet(structures_path), pd.read_parquet(tasks_path)
    full_scope, scope_summary = stored_scope(structures, tasks)
    full_scope.to_csv(out / "full_stored_scope.csv", index=False)
    canonical = tasks[tasks.fidelity == "source_relaxed_dft"].merge(
        structures[["dataset_structure_id", "source_row", "nelements", "natoms", "duplicate_of"]],
        left_on="structure_state_id", right_on="dataset_structure_id", validate="one_to_one")
    eligible = canonical[(canonical.nelements > 1) & canonical.parent_lattice.isin(["fcc", "bcc"])].copy()
    eligible["selection_hash"] = eligible.parent_group_id.map(
        lambda value: hashlib.sha256(f"{args.seed}:{value}".encode()).hexdigest())
    selected = []
    for split in ("train", "validation", "test"):
        subset = eligible[eligible.id_split == split].sort_values(["selection_hash", "structure_state_id"])
        selected.append(subset.drop_duplicates("parent_group_id").head(getattr(args, split)))
    pilot = pd.concat(selected).copy()
    pilot["pilot"] = True
    rare = eligible[eligible.structure_family.isin(["b2_like", "l12_like"]) &
                    ~eligible.structure_state_id.isin(pilot.structure_state_id)].copy()
    rare["pilot"] = False
    selection = pd.concat([pilot, rare], ignore_index=True)
    if selection.groupby("parent_group_id").id_split.nunique().max() != 1:
        raise ValueError("Parent split leakage")
    selection.to_csv(out / "selection.csv", index=False)
    protocol = {"seed": args.seed, "split_counts": pilot.id_split.value_counts().to_dict(),
                "selection": "Lowest SHA256(seed:parent_group_id) per original ID split, one canonical view per parent; all rare prototypes added for audit only",
                "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [structures_path, tasks_path, ROOT / "alloy_xrd_bench/occupancy_pipeline.py"]},
                "selection_sha256": sha256(out / "selection.csv"), "full_stored_scope": scope_summary,
                "scope_caveat": "Full scope inferred from existing outgoing-pair keys; coordinate recomputation is sampled."}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    payload_path = root / "xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet"
    payload = pd.read_parquet(payload_path, filters=[("source_row", "in", selection.source_row.tolist())]).set_index("source_row")
    vocab = json.loads((ROOT / "configs/hea_validation_v1.json").read_text())["elements"]
    rows, errors = [], []
    for index, record in enumerate(selection.to_dict("records")):
        structure = Structure.from_dict(ast.literal_eval(payload.loc[record["source_row"], "structure_as_dict"]))
        actual = _sro(structure, record["parent_lattice"])
        reference, diagnostics = ase_sro(structure, record["parent_lattice"])
        max_error, added, old_changed, masks_different = 0., 0, 0, False
        for shell, (a, b) in enumerate(zip(actual, reference), 1):
            masks_different |= set(a) != set(b)
            max_error = max(max_error, max((abs(a[k] - b[k]) for k in a.keys() & b.keys()), default=0.))
            old = json.loads(record[f"sro_shell{shell}_json"])
            old_changed += sum(k not in a or abs(v - a[k]) > 1e-10 for k, v in old.items())
            added += len(a.keys() - old.keys())
            record[f"corrected_shell{shell}_json"] = json.dumps(a, sort_keys=True)
            record[f"shell{shell}_diagnostics_json"] = json.dumps(diagnostics[shell - 1])
        record.update({"max_ase_error": max_error, "mask_disagreement": masks_different,
                       "old_values_changed": old_changed, "added_keys": added,
                       "volume_per_atom": structure.volume / len(structure),
                       "composition_json": json.dumps({el: structure.composition.get_atomic_fraction(el) for el in vocab})})
        if masks_different or max_error > 1e-10 or old_changed:
            errors.append({k: record[k] for k in ["structure_state_id", "max_ase_error", "mask_disagreement", "old_values_changed"]})
        rows.append(record)
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(selection)}; discrepancy rows={len(errors)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_parquet(out / "corrected_sample.parquet", index=False)
    report = {"sample_size": len(results), "pilot_size": len(pilot), "errors": errors,
              "max_ase_error": float(results.max_ase_error.max()), "sample_added_keys": int(results.added_keys.sum()),
              "sample_affected_states": int((results.added_keys > 0).sum()),
              "old_values_changed": int(results.old_values_changed.sum()),
              "families": results.structure_family.value_counts().to_dict(),
              "parent_leakage_count": 0, "natoms_min": int(results.natoms.min()), "natoms_max": int(results.natoms.max()),
              "full_stored_scope": scope_summary,
              "limitations": "Agreement validates arithmetic and masks under the shared radial-shell convention, not shell physical adequacy, thermodynamic realism, or experimental observability."}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit("Independent audit failed; do not use this sample for scientific claims")


if __name__ == "__main__":
    main()
