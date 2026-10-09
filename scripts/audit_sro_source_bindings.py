"""Trace the frozen audit sample back to original CSV rows and source hashes."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.contract import canonical_structure_hash
from scripts.audit_sro_independent import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    if ROOT / "data" == out or ROOT / "data" in out.parents:
        raise ValueError("Do not overwrite source data")
    sample = pd.read_parquet(out / "corrected_sample.parquet").set_index("source_row")
    benchmark = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    inventory = json.loads((benchmark / "reports/inventory_summary.json").read_text())
    source = Path(inventory["source_csv"])
    checksum = sha256(source)
    if checksum != inventory["source_sha256"]:
        raise ValueError("Original CSV hash does not match released inventory")
    payload_path = benchmark / "xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet"
    payload = pd.read_parquet(payload_path, filters=[("source_row", "in", sample.index.tolist())]).set_index("source_row")
    errors, hash_warnings, matched = [], [], 0
    for chunk in pd.read_csv(source, usecols=["fid", "structure_as_dict"], chunksize=2000):
        common = chunk.index.intersection(sample.index)
        for source_row in common:
            row, wanted = chunk.loc[source_row], sample.loc[source_row]
            document = ast.literal_eval(row.structure_as_dict)
            cached = ast.literal_eval(payload.loc[source_row, "structure_as_dict"])
            structure = Structure.from_dict(document)
            actual = canonical_structure_hash(structure.as_dict())
            if actual != wanted.structure_sha256:
                hash_warnings.append({"source_row": int(source_row), "runtime_hash": actual,
                                      "stored_hash": wanted.structure_sha256})
            if document != cached or "occstr_" + str(row.fid) != wanted.structure_state_id:
                errors.append({"source_row": int(source_row), "payload_equal": document == cached,
                               "fid": str(row.fid), "structure_state_id": wanted.structure_state_id})
            matched += 1
    old_protocol = json.loads((out / "protocol.json").read_text())
    source_manifest_changes = [name for name, digest in old_protocol["source_hashes"].items()
                               if name.endswith(".parquet") and sha256(ROOT / name) != digest]
    nonideal, empty = Counter(), Counter()
    for row in sample.to_dict("records"):
        for shell in (1, 2):
            diagnostics = json.loads(row[f"shell{shell}_diagnostics_json"])
            expected = (12 if row["parent_lattice"] == "fcc" else 8) if shell == 1 else 6
            nonideal[f"shell{shell}"] += int(diagnostics["coordination_min"] != expected or diagnostics["coordination_max"] != expected)
            empty[f"shell{shell}"] += int(diagnostics["zero_neighbor_atoms"] > 0)
    result = {"source_csv_sha256": checksum, "inventory_hash_match": True, "sample_rows": len(sample),
              "matched_original_csv_rows": matched, "structure_id_or_raw_payload_errors": errors,
              "reconstructed_stored_hash_mismatch_count": len(hash_warnings),
              "hash_warning": "Stored serialized-Structure hashes are not all reproducible in this runtime. Raw source-to-cache equality is checked independently. Do not interpret this as a passed historical hash audit; historical serialization environment remains to be reconciled.",
              "source_manifest_changes": source_manifest_changes,
              "sample_structures_with_nonideal_coordination": dict(nonideal),
              "sample_structures_with_zero_neighbor_atoms": dict(empty),
              "interpretation": "Nonideal coordination is a flag for shell-definition sensitivity, not evidence of arithmetic disagreement. Full source CSV checksum verified; per-structure bindings checked only on the frozen sample."}
    (out / "source_bindings_raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "runtime_hash_mismatches.json").write_text(json.dumps(hash_warnings, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors or source_manifest_changes or matched != len(sample):
        raise SystemExit("Source binding audit failed")


if __name__ == "__main__":
    main()
