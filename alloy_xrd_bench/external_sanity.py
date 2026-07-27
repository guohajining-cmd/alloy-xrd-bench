from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


GAMMA_SOURCE_DOI = "10.17863/CAM.87406"
GAMMA_ARTICLE_DOI = "10.1017/dce.2022.31"
GAMMA_SOURCE_URL = (
    "https://api.repository.cam.ac.uk/server/api/core/bitstreams/"
    "22010f28-2195-41e0-8e5c-ee90b6354a4f/content"
)
GAMMA_SOURCE_SHA256 = "e58445b9c2769ac3991e97fa6216b05e6332845ff6eadaeaf2b6e4545c377465"
GAMMA_SOURCE_TITLE = (
    'Research data supporting "Design of a Ni-based Superalloy for Laser Repair '
    'Applications using Probabilistic Neural Network Identification"'
)
GAMMA_SOURCE_CREATORS = [
    "Markanday, Jonathon",
    "Conduit, Gareth",
    "Conduit, Bryce",
    "Purstl, Julia",
    "Christofidou, Katerina",
    "Chechik, Lova",
    "Baxter, Gavin",
    "Heason, Christopher",
    "Stone, Howard",
]
GAMMA_FILES = {
    "IN718_HTppt_32_95_400_exported.xy": ("gamma_exp_in718_htppt", "IN718"),
    "IN718Mk3_HTppt_32_95_400_exported.xy": ("gamma_exp_am718r_htppt", "AM718R"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bruker_export_xy(path: str | Path) -> tuple[str, np.ndarray]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Gamma sanity pattern is empty: {path}")
    header = lines[0]
    values = np.loadtxt(lines[1:], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected two-column angle-intensity data: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Pattern contains non-finite values: {path}")
    if (values[:, 1] < 0).any():
        raise ValueError(f"Pattern contains negative intensity: {path}")
    if not (np.diff(values[:, 0]) > 0).all():
        raise ValueError(f"Two-theta grid is not strictly increasing: {path}")
    if 'Anode: "Cu"' not in header:
        raise ValueError(f"Pattern does not declare a Cu source: {path}")
    return header, values


def admit_gamma_sanity(
    source_zip: str | Path,
    external_root: str | Path = "data/external_sanity",
) -> dict:
    source_zip = Path(source_zip)
    external_root = Path(external_root)
    if _sha256(source_zip) != GAMMA_SOURCE_SHA256:
        raise ValueError("Gamma sanity source ZIP does not match the frozen SHA256")

    root = external_root / "gamma_v1"
    source_dir = root / "source"
    raw_dir = root / "raw"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archived_source = source_dir / "XRD Data.zip"
    shutil.copyfile(source_zip, archived_source)

    rows = []
    with zipfile.ZipFile(source_zip) as archive:
        if set(archive.namelist()) != set(GAMMA_FILES):
            raise ValueError("Gamma sanity ZIP members do not match the frozen source contract")
        for member, (sample_id, alloy) in GAMMA_FILES.items():
            raw_path = raw_dir / member
            raw_path.write_bytes(archive.read(member))
            header, values = read_bruker_export_xy(raw_path)
            file_hash = _sha256(raw_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "track": "gamma_decomp",
                    "composition": alloy,
                    "treatment": "precipitation heat-treated",
                    "source_doi": GAMMA_SOURCE_DOI,
                    "source_title": GAMMA_SOURCE_TITLE,
                    "source_creators_json": json.dumps(GAMMA_SOURCE_CREATORS),
                    "article_doi": GAMMA_ARTICLE_DOI,
                    "license": "CC-BY-ND-4.0",
                    "source_raw_path": raw_path.as_posix(),
                    "pattern_path": raw_path.as_posix(),
                    "pattern_format": "bruker_export_xy",
                    "point_count": int(len(values)),
                    "two_theta_min": float(values[0, 0]),
                    "two_theta_max": float(values[-1, 0]),
                    "step_deg": float(np.median(np.diff(values[:, 0]))),
                    "radiation": "Cu Kalpha",
                    "exact_latent_truth_available": False,
                    "allowed_use": "phase_trend_reconstruction_and_domain_gap_only",
                    "source_bundle_sha256": GAMMA_SOURCE_SHA256,
                    "source_raw_sha256": file_hash,
                    "pattern_sha256": file_hash,
                    "source_url": GAMMA_SOURCE_URL,
                    "source_header": header,
                }
            )

    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    manifest.to_parquet(root / "manifest.parquet", index=False)
    report = {
        "schema_version": "sitexrd-external-sanity-v1",
        "status": "pass",
        "track": "gamma_decomp",
        "sample_count": int(len(manifest)),
        "source_doi": GAMMA_SOURCE_DOI,
        "source_title": GAMMA_SOURCE_TITLE,
        "source_creators": GAMMA_SOURCE_CREATORS,
        "article_doi": GAMMA_ARTICLE_DOI,
        "license": "CC-BY-ND-4.0",
        "source_bundle_sha256": GAMMA_SOURCE_SHA256,
        "source_files_redistributed_unmodified": True,
        "derived_pattern_files_distributed": False,
        "exact_latent_truth_claimed": False,
        "allowed_claims": [
            "gamma matrix peak trend",
            "whole-pattern reconstruction trend",
            "simulation-to-experiment domain gap",
        ],
        "prohibited_claims": [
            "experimental component-level accuracy",
            "gamma-prime versus gamma-double-prime component accuracy",
            "experimental sublattice occupancy accuracy",
        ],
        "source_limitations": (
            "The associated article states that gamma-prime and gamma-double-prime "
            "superlattice reflections are not distinguishable in these patterns."
        ),
    }
    (root / "sanity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_combined_status(external_root)
    return report


def write_combined_status(external_root: str | Path) -> dict:
    external_root = Path(external_root)
    tracks = {}
    for track, folder in (("hea_sro", "hea_v1"), ("gamma_decomp", "gamma_v1")):
        report_path = external_root / folder / "sanity_report.json"
        if not report_path.exists():
            tracks[track] = {"status": "missing", "sample_count": 0}
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        tracks[track] = {
            "status": report.get("status", "invalid"),
            "sample_count": int(report.get("sample_count", 0)),
            "report": f"{folder}/sanity_report.json",
        }
    status = {
        "schema_version": "sitexrd-external-sanity-status-v1",
        "status": "pass" if all(v["status"] == "pass" for v in tracks.values()) else "partial",
        "exact_latent_truth_claimed": False,
        "tracks": tracks,
    }
    (external_root / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit traceable external XRD sanity data")
    parser.add_argument("source_zip")
    parser.add_argument("--external-root", default="data/external_sanity")
    args = parser.parse_args()
    report = admit_gamma_sanity(args.source_zip, args.external_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
