"""Measure runtime sensitivity of label-derived SRO-OOD, without reassigning it."""

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.sro_revision import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == args.dataset.resolve() or out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    path = args.dataset / "manifests/task_samples.parquet"
    meta = json.loads((args.revision / "revision.json").read_text(encoding="utf-8"))
    if meta["status"] != "verified" or file_sha256(path) != meta["source_manifest_sha256"]:
        raise ValueError("Source manifest binding failed")
    if file_sha256(args.revision / "labels.parquet") != meta["labels_sha256"]:
        raise ValueError("Labels hash mismatch")
    tasks = pd.read_parquet(path)
    labels = pd.read_parquet(args.revision / "labels.parquet").set_index("structure_state_id")
    tasks["corrected"] = labels.loc[tasks.structure_state_id, "sro_shell1_json"].to_numpy()
    groups = tasks.groupby("parent_group_id", sort=False).first()
    thresholds = {}
    for name, column in [("old", "sro_shell1_json"), ("corrected", "corrected")]:
        groups[f"{name}_norm"] = groups[column].map(lambda s: float(np.linalg.norm(list(json.loads(s).values()))))
        low, high = groups[f"{name}_norm"].quantile([.15, .85]).tolist()
        thresholds[name] = [low, high]
        holdout = (groups[f"{name}_norm"] <= low) | (groups[f"{name}_norm"] >= high)
        groups[f"{name}_rule_split"] = np.where(holdout, "test", np.where(groups.id_split == "validation", "validation", "train"))
    mismatch = groups.old_rule_split != groups.sro_ood_split
    near_boundary = np.minimum(np.abs(groups.old_norm-thresholds["old"][0]), np.abs(groups.old_norm-thresholds["old"][1])) <= 1e-12
    report = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  source_manifest_sha256=file_sha256(path), labels_sha256=meta["labels_sha256"],
                  parent_count=len(groups), thresholds=thresholds,
                  old_rule_disagrees_with_stored=int(mismatch.sum()),
                  old_rule_mismatches_within_1e12_of_boundary=int((mismatch & near_boundary).sum()),
                  corrected_rule_disagrees_with_stored=int((groups.corrected_rule_split != groups.sro_ood_split).sum()),
                  corrected_rule_disagrees_with_recomputed_old=int((groups.corrected_rule_split != groups.old_rule_split).sum()),
                  applied_to_benchmark=False,
                  qualification="Diagnostic only; preserve frozen split files for A/B/C. Numeric cutoff ties require a declared policy before a future split release.")
    groups[["sro_ood_split", "old_norm", "corrected_norm", "old_rule_split", "corrected_rule_split"]].to_csv(out / "parents.csv")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
