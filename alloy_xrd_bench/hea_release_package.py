"""Freeze HEA benchmark v1.1 manifests, provenance, licenses, and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import pandas as pd


DEFAULT_MANIFESTS = (
    "structures.parquet",
    "xrd_samples.parquet",
    "task_samples.parquet",
    "split_assignments.parquet",
    "occupancy_pairs.parquet",
    "composition_only_features.parquet",
)
FROZEN_V1 = Path("/home/GJM/data/hea/benchmark_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_output(source: Path, output: Path) -> None:
    if output == source or source in output.parents:
        raise ValueError(f"immutable source dataset cannot be an output: {output}")
    normalized = output.as_posix().rstrip("/")
    if normalized == FROZEN_V1.as_posix() or normalized.startswith(FROZEN_V1.as_posix() + "/"):
        raise ValueError(f"immutable HEA v1 path cannot be written: {output}")


def freeze_release_package(
    source: Path,
    output: Path,
    config_path: Path,
    *,
    source_license: str,
    required_manifests: Sequence[str] = DEFAULT_MANIFESTS,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    config_path = config_path.resolve()
    _guard_output(source, output)
    if not source_license.strip():
        raise ValueError("source license must be explicitly declared")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("release_scope") != "sro_core":
        raise ValueError("release config must freeze release_scope=sro_core")
    if bool(config.get("experimental_noise_calibrated", False)):
        raise ValueError("v1.1 cannot claim experimental noise calibration")

    manifests = output / "manifests"
    release = output / "release"
    manifests.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    missing_manifests = []
    for name in required_manifests:
        source_path = source / "manifests" / name
        if not source_path.exists():
            missing_manifests.append(name)
            continue
        shutil.copy2(source_path, manifests / name)

    task_path = manifests / "task_samples.parquet"
    if task_path.exists():
        tasks = pd.read_parquet(task_path)
        labels = pd.DataFrame(
            {
                "structure_state_id": tasks["structure_state_id"].astype(str),
                "parent_group_id": tasks["parent_group_id"].astype(str),
                "automated_family_label": (
                    tasks["structure_family"].astype(str)
                    if "structure_family" in tasks
                    else "unknown"
                ),
                "ground_truth_status": "automated_unadjudicated",
                "release_gate": False,
            }
        )
        labels.to_parquet(manifests / "automated_family_labels.parquet", index=False)

    snapshot_path = release / "hea_benchmark_v1_1.config.snapshot.json"
    snapshot_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "schema_version": "sitexrd-hea-source-provenance-v1.1",
        "release_scope": "sro_core",
        "source_dataset_root": str(source),
        "output_dataset_root": str(output),
        "source_dataset_access": "read-only",
        "source_license": source_license,
        "source_dataset": config.get("source_dataset", {}),
        "experimental_noise_calibrated": False,
        "family_ground_truth_status": "automated_unadjudicated_diagnostic",
        "occupancy_status": "simulated_identifiability_diagnostic",
        "required_manifests": list(required_manifests),
        "missing_manifests": missing_manifests,
    }
    (release / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        [{"component": "hea_source_dataset", "license": source_license, "status": "declared"}]
    ).to_csv(release / "license_audit.csv", index=False)

    artifact_roots = [manifests, release]
    if (output / "xrd").exists():
        artifact_roots.append(output / "xrd")
    if (output / "reports").exists():
        artifact_roots.append(output / "reports")
    files = sorted(
        path for parent in artifact_roots for path in parent.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "file_manifest.parquet"}
    )
    file_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "source": "v1_manifest" if manifests in path.parents else "v1_1_generated",
        }
        for path in files
    ]
    pd.DataFrame(file_rows).to_parquet(release / "file_manifest.parquet", index=False)
    checksum_path = release / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in file_rows), encoding="utf-8"
    )
    checksum_errors = sum(
        not (output / row["path"]).exists() or _sha256(output / row["path"]) != row["sha256"]
        for row in file_rows
    )
    license_errors = 0 if source_license.strip() else 1
    status = "pass" if not (missing_manifests or checksum_errors or license_errors) else "fail"
    return {
        "schema_version": "sitexrd-hea-release-package-v1.1",
        "status": status,
        "release_scope": "sro_core",
        "source_dataset_immutable": True,
        "manifest_count": len(required_manifests) - len(missing_manifests),
        "missing_manifests": missing_manifests,
        "file_count": len(file_rows),
        "checksum_errors": int(checksum_errors),
        "license_errors": license_errors,
        "checksum_path": str(checksum_path),
        "file_manifest_path": str(release / "file_manifest.parquet"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hea_benchmark_v1_1.json"))
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = freeze_release_package(
        args.source, args.output, args.config, source_license=args.source_license
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
