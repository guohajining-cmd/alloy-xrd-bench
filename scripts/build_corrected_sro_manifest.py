"""Write a versioned SRO-label sidecar without changing the released manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def complete_shell(raw: str, elements: list[str]) -> tuple[str, int, int]:
    values = json.loads(raw or "{}")
    if not isinstance(values, dict):
        raise ValueError("SRO shell must be a JSON object")
    allowed = set(elements)
    centers = set()
    for pair, value in values.items():
        parts = pair.split("->")
        if len(parts) != 2 or not set(parts) <= allowed or not math.isfinite(float(value)):
            raise ValueError(f"invalid SRO pair or value: {pair!r}")
        centers.add(parts[0])
    added = 0
    for center in centers:
        for other in elements:
            pair = f"{center}->{other}"
            if pair not in values:
                values[pair] = 1.0
                added += 1
    undefined = len(allowed - centers) * len(allowed)
    return json.dumps(values, sort_keys=True), added, undefined


def corrected_labels(tasks: pd.DataFrame, element_map: dict[str, list[str]]) -> pd.DataFrame:
    if tasks.structure_state_id.duplicated().any():
        raise ValueError("duplicate structure_state_id")
    rows = []
    for row in tasks.itertuples(index=False):
        elements = element_map.get(row.parent_structure_id)
        if elements is None:
            raise ValueError(f"missing parent structure {row.parent_structure_id!r}")
        corrected = []
        added_counts = []
        undefined_counts = []
        for raw in (row.sro_shell1_json, row.sro_shell2_json):
            if row.parent_lattice in {"fcc", "bcc"} and len(elements) > 1:
                value, added, undefined = complete_shell(raw, elements)
            else:
                value, added, undefined = raw, 0, 0
            corrected.append(value)
            added_counts.append(added)
            undefined_counts.append(undefined)
        rows.append({
            "structure_state_id": row.structure_state_id,
            "parent_group_id": row.parent_group_id,
            "fidelity": row.fidelity,
            "corrected_shell1_json": corrected[0],
            "corrected_shell2_json": corrected[1],
            "added_shell1_keys": added_counts[0],
            "added_shell2_keys": added_counts[1],
            "undefined_shell1_keys": undefined_counts[0],
            "undefined_shell2_keys": undefined_counts[1],
        })
    return pd.DataFrame(rows)


def validate_sample(sidecar: pd.DataFrame, sample_path: Path) -> int:
    sample = pd.read_parquet(sample_path).set_index("structure_state_id")
    corrected = sidecar.set_index("structure_state_id")
    if not sample.index.is_unique or not sample.index.isin(corrected.index).all():
        raise ValueError("independent sample IDs do not map one-to-one to corrected labels")
    for row in sample.itertuples():
        stored = corrected.loc[row.Index]
        for shell in (1, 2):
            expected = json.loads(getattr(row, f"corrected_shell{shell}_json"))
            actual = json.loads(stored[f"corrected_shell{shell}_json"])
            if expected.keys() != actual.keys() or any(
                not math.isclose(float(expected[key]), float(actual[key]), rel_tol=0, abs_tol=1e-10)
                for key in expected
            ):
                raise ValueError(f"independent label mismatch: {row.Index}, shell {shell}")
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT / "data" or ROOT / "data" in out.parents:
        raise ValueError("output must not overwrite released data")
    if out.exists():
        raise ValueError("use a fresh output directory")
    structures_path = DATA_ROOT / "manifests/structures.parquet"
    tasks_path = DATA_ROOT / "manifests/task_samples.parquet"
    audit_dir = args.sample_audit_dir.resolve()
    sample_path = audit_dir / "corrected_sample.parquet"
    audit_report = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if audit_report["errors"]:
        raise ValueError("independent sample audit has errors")
    structures = pd.read_parquet(structures_path, columns=["dataset_structure_id", "elements_json"])
    if structures.dataset_structure_id.duplicated().any():
        raise ValueError("duplicate parent structure ID")
    element_map = dict(zip(structures.dataset_structure_id, structures.elements_json.map(json.loads)))
    tasks = pd.read_parquet(tasks_path)
    sidecar = corrected_labels(tasks, element_map)
    checked = validate_sample(sidecar, sample_path)
    if len(sidecar) != len(tasks) or not sidecar.structure_state_id.equals(tasks.structure_state_id):
        raise ValueError("sidecar row alignment failed")
    by_fidelity = {}
    for fidelity, group in sidecar.groupby("fidelity", sort=True):
        added = group.added_shell1_keys + group.added_shell2_keys
        by_fidelity[fidelity] = {
            "states": len(group),
            "affected_states": int((added > 0).sum()),
            "added_keys": int(added.sum()),
            "undefined_keys": int((group.undefined_shell1_keys + group.undefined_shell2_keys).sum()),
        }
    report = {
        "qualification": "Full stored-label completion under the frozen shell rule; independent coordinate recomputation is sampled",
        "rows": len(sidecar),
        "independent_sample_checked": checked,
        "by_fidelity": by_fidelity,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (structures_path, tasks_path, sample_path)},
        "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "Only absent pairs for centers with a recorded outgoing shell are filled with alpha=1.",
            "Centers without any shell neighbors stay undefined.",
            "Frozen SRO OOD splits were built using the old sparse labels and are retained for paired isolation.",
            "This sidecar does not modify the released dataset or its published scores.",
        ],
    }
    out.mkdir(parents=True)
    sidecar.to_parquet(out / "corrected_sro_labels.parquet", index=False)
    report["sidecar_sha256"] = sha256(out / "corrected_sro_labels.parquet")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
