"""Generate metadata documents for Alloy-XRD-Bench releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


EXCLUDED_FILE_OBJECTS = {
    "checksums.sha256",
    "croissant.jsonld",
    "file_manifest.parquet",
    "publication_record.yml",
}


def write_croissant_metadata(
    *,
    release_root: str | Path,
) -> dict[str, Any]:
    """Write a release-specific Croissant-style JSON-LD metadata file."""

    root = Path(release_root).resolve()
    summary_path = root / "release_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    file_rows = _scan_files(root)
    payload = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "Dataset",
        "name": "Alloy-XRD-Bench",
        "description": (
            "A two-line simulated alloy XRD benchmark with per-potential "
            "structure provenance, private-test hiding, and file-level checksums."
        ),
        "version": summary.get("release_version", "working"),
        "license": "See release manifests and per-potential license audit.",
        "keywords": [
            "XRD",
            "alloys",
            "machine-learning interatomic potentials",
            "provenance",
            "benchmark",
        ],
        "measurementTechnique": "simulated_xrd_from_relaxed_structure",
        "variableMeasured": [
            "two_theta",
            "intensity",
            "peak_position",
            "peak_intensity",
            "relaxed_structure_id",
            "split",
        ],
        "releaseSummary": summary,
        "fileObjects": [_file_object(row) for row in file_rows],
        "recordSets": _record_sets(),
    }
    (root / "croissant.jsonld").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"file_objects": len(payload["fileObjects"]), "path": str(root / "croissant.jsonld")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Alloy-XRD-Bench metadata artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    croissant = subparsers.add_parser("croissant", help="Generate release-specific croissant.jsonld")
    croissant.add_argument("release_root")

    args = parser.parse_args(argv)
    if args.command == "croissant":
        report = write_croissant_metadata(release_root=args.release_root)
        print(f"file_objects: {report['file_objects']}")
        print(f"path: {report['path']}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _file_object(row: dict[str, Any]) -> dict[str, Any]:
    path = str(row["path"])
    return {
        "@id": path,
        "@type": "cr:FileObject",
        "name": path,
        "contentUrl": path,
        "encodingFormat": _encoding_format(path),
        "contentSize": int(row.get("size_bytes", 0)),
        "sha256": row.get("sha256", ""),
    }


def _encoding_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".parquet": "application/vnd.apache.parquet",
        ".json": "application/json",
        ".jsonld": "application/ld+json",
        ".csv": "text/csv",
        ".yml": "application/x-yaml",
        ".yaml": "application/x-yaml",
        ".md": "text/markdown",
        ".cff": "text/plain",
        ".cif": "chemical/x-cif",
        ".sha256": "text/plain",
    }.get(suffix, "application/octet-stream")


def _record_sets() -> list[dict[str, Any]]:
    return [
        {"@id": "manifests/potentials_catalog.yml", "@type": "cr:RecordSet", "name": "MLIP registry"},
        {"@id": "manifests/initial_structures.parquet", "@type": "cr:RecordSet", "name": "Initial structures"},
        {"@id": "manifests/relaxations.parquet", "@type": "cr:RecordSet", "name": "Relaxation runs"},
        {"@id": "manifests/relaxed_structures.parquet", "@type": "cr:RecordSet", "name": "Relaxed structures"},
        {"@id": "manifests/xrd_samples.parquet", "@type": "cr:RecordSet", "name": "Individual XRD samples"},
        {"@id": "manifests/xrd_averages.parquet", "@type": "cr:RecordSet", "name": "Averaged XRD samples"},
        {"@id": "manifests/private_inputs.parquet", "@type": "cr:RecordSet", "name": "Private test inputs"},
        {"@id": "manifests/provenance_edges.parquet", "@type": "cr:RecordSet", "name": "Provenance graph"},
    ]


def _scan_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED_FILE_OBJECTS:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
