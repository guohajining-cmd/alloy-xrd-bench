"""Package Alloy-XRD-Bench release directories with file-level checksums."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def package_release(
    *,
    release_root: str | Path,
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write file manifest/checksums and optionally a zip archive."""

    root = Path(release_root).resolve()
    if archive_path is None:
        archive = root.with_suffix(".zip")
    else:
        archive = Path(archive_path).resolve()

    manifest_path = root / "file_manifest.parquet"
    checksums_path = root / "checksums.sha256"
    sidecar_paths = {root / "publication_record.yml"}
    rows = _collect_file_rows(
        root,
        exclude={manifest_path.resolve(), checksums_path.resolve(), archive, *sidecar_paths},
    )
    pd.DataFrame(rows).to_parquet(manifest_path)
    rows = _collect_file_rows(root, exclude={checksums_path.resolve(), archive, *sidecar_paths})
    checksums_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    rows = _collect_file_rows(root, exclude={archive, *sidecar_paths})
    pd.DataFrame(rows).to_parquet(manifest_path)
    checksums_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for row in rows:
            zip_file.write(root / row["path"], row["path"])

    return {
        "release_root": str(root),
        "archive": str(archive),
        "file_count": len(rows),
        "manifest_path": str(manifest_path),
        "checksums_path": str(checksums_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package an Alloy-XRD-Bench release directory.")
    parser.add_argument("release_root", help="Release directory to package")
    parser.add_argument("--archive", help="Zip archive path")
    args = parser.parse_args(argv)

    report = package_release(release_root=args.release_root, archive_path=args.archive)
    print(f"file_count: {report['file_count']}")
    print(f"checksums: {report['checksums_path']}")
    print(f"manifest: {report['manifest_path']}")
    print(f"archive: {report['archive']}")
    return 0


def _collect_file_rows(root: Path, *, exclude: set[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if resolved in exclude:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
