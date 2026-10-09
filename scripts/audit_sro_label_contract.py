"""Check ideal SRO fixtures and stored label completeness without editing data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def prototype_audit(root: Path) -> dict:
    from pymatgen.core import Lattice, Structure

    source = root / "alloy_xrd_bench/occupancy_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_sro", "_warren_cowley", "_site_element"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise ValueError("Expected SRO functions were not found")
    # Execute exact source functions without importing unrelated HDF5/Arrow modules.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace = {"math": math, "Counter": Counter, "Structure": Structure}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    fixtures = [
        ("B2_FeAl", "bcc", Structure(Lattice.cubic(3), ["Fe", "Al"], [[0, 0, 0], [.5, .5, .5]]),
         [{"Fe->Fe": 1., "Al->Al": 1., "Fe->Al": -1., "Al->Fe": -1.},
          {"Fe->Fe": -1., "Al->Al": -1., "Fe->Al": 1., "Al->Fe": 1.}]),
        ("L12_Cu3Au", "fcc", Structure(Lattice.cubic(4), ["Au", "Cu", "Cu", "Cu"],
                                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
         [{"Au->Au": 1., "Au->Cu": -1/3, "Cu->Au": -1/3, "Cu->Cu": 1/9},
          {"Au->Au": -3., "Au->Cu": 1., "Cu->Au": 1., "Cu->Cu": -1/3}]),
    ]
    results = []
    for name, parent, structure, expected in fixtures:
        actual = namespace["_sro"](structure, parent)
        for index, (wanted, observed) in enumerate(zip(expected, actual), 1):
            results.append({
                "fixture": name, "shell": index, "expected": wanted, "actual": observed,
                "missing_defined_pairs": sorted(set(wanted) - set(observed)),
                "max_error_on_present_pairs": max(abs(observed[k] - wanted[k]) for k in observed),
            })
    return {"scope": "Exact production SRO functions isolated with AST; analytical fixtures, not a full pipeline run",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fixtures": results}


def stored_audit(root: Path) -> tuple[dict, list[dict]]:
    import pandas as pd

    directory = root / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1/manifests"
    path = directory / "structures.parquet"
    frame = pd.read_parquet(path)
    tasks = pd.read_parquet(directory / "task_samples.parquet").set_index("structure_state_id")
    rows = []
    binding_errors = 0
    eligible = 0
    for record in frame.to_dict("records"):
        if record["parent_lattice"] not in {"fcc", "bcc"}:
            continue
        elements = json.loads(record["elements_json"])
        if len(elements) < 2:
            continue
        eligible += 1
        expected = {f"{a}->{b}" for a in elements for b in elements}
        for shell in (1, 2):
            column = f"sro_shell{shell}_json"
            values = json.loads(record[column])
            task_values = json.loads(tasks.loc[record["dataset_structure_id"], column])
            binding_errors += int(values != task_values)
            missing = sorted(expected - set(values))
            if missing:
                rows.append({"structure_state_id": record["dataset_structure_id"],
                             "family": record["structure_family"], "shell": shell,
                             "nonempty_shell_dictionary": bool(values),
                             "missing_pairs": "|".join(missing), "missing_count": len(missing)})
    affected = {r["structure_state_id"] for r in rows}
    return {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "structure_count": len(frame), "multielement_fcc_bcc_rows": eligible,
        "rows_requiring_missing_pair_review": len(affected),
        "shell_rows_requiring_review": len(rows),
        "missing_key_count": sum(r["missing_count"] for r in rows),
        "canonical_to_task_label_mismatches": binding_errors,
        "affected_structure_counts_by_family": dict(Counter(
            frame.set_index("dataset_structure_id").loc[sorted(affected), "structure_family"])),
        "caveat": "Missing keys are audit flags, not all independently validated errors. Verify shell availability before completing zero-count pairs."
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prototype", "stored"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if root / "data" == output or root / "data" in output.parents:
        raise ValueError("Do not write audit output into source data")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "prototype":
        result = prototype_audit(root)
    else:
        import pandas as pd
        result, rows = stored_audit(root)
        pd.DataFrame(rows).to_csv(output / "missing_pair_flags.csv", index=False)
    (output / f"{args.mode}_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
