"""Generate a minimal valid Alloy-XRD-Bench workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alloy_xrd_bench.contract import canonical_structure_hash


POTENTIAL_ID = "pot_mace_mp_0"


def make_minimal_workspace(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    paths = {
        "registry": root_path / "data" / "registry",
        "shared_manifests": root_path / "data" / "shared" / "manifests",
        "shared_structures": root_path / "data" / "shared" / "structures_initial",
        "potential": root_path / "data" / "potentials" / POTENTIAL_ID / "potential",
        "potential_manifests": root_path
        / "data"
        / "potentials"
        / POTENTIAL_ID
        / "generated"
        / "manifests",
        "relax_logs": root_path
        / "data"
        / "potentials"
        / POTENTIAL_ID
        / "generated"
        / "relaxation_logs",
        "relaxed": root_path
        / "data"
        / "potentials"
        / POTENTIAL_ID
        / "generated"
        / "structures_relaxed",
        "xrd_grid": root_path
        / "data"
        / "potentials"
        / POTENTIAL_ID
        / "generated"
        / "xrd"
        / "grid",
        "xrd_peaks": root_path
        / "data"
        / "potentials"
        / POTENTIAL_ID
        / "generated"
        / "xrd"
        / "peaks",
        "audits": root_path / "data" / "potentials" / POTENTIAL_ID / "audits",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    initial_doc = {"@module": "pymatgen.core.structure", "example": "initial"}
    relaxed_doc = {"@module": "pymatgen.core.structure", "example": "relaxed"}
    initial_bytes = json.dumps(initial_doc, sort_keys=True, indent=2).encode("utf-8")
    relaxed_bytes = json.dumps(relaxed_doc, sort_keys=True, indent=2).encode("utf-8")
    cif_bytes = b"# example CIF exchange artifact\n"
    (paths["shared_structures"] / "init_CrCoNi_fcc_000001.json").write_bytes(initial_bytes)
    (paths["relaxed"] / "rstr_CrCoNi_fcc_000001.json").write_bytes(relaxed_bytes)
    (paths["relaxed"] / "rstr_CrCoNi_fcc_000001.cif").write_bytes(cif_bytes)
    (paths["relax_logs"] / "relax_CrCoNi_fcc_000001.jsonl").write_text(
        '{"step":1,"fmax":0.03}\n',
        encoding="utf-8",
    )
    (paths["potential"] / "metadata.yml").write_text(
        yaml.safe_dump(
            {
                "potential_id": POTENTIAL_ID,
                "note": "Example metadata-only potential entry; no MLIP file is bundled.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (paths["potential"] / "checksums.sha256").write_text(
        f"{'a' * 64}  external:https://example.org/mace\n",
        encoding="utf-8",
    )
    (paths["potential"] / "INSTALL.md").write_text(
        "\n".join(
            [
                f"# {POTENTIAL_ID} Installation",
                "",
                "This example potential is metadata-only and does not bundle an MLIP file.",
                "",
                "- Source: https://example.org/mace",
                "- DOI: 10.0000/example",
                "- Expected SHA-256: " + "a" * 64,
                "",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"two_theta": [20.0, 20.02, 20.04], "intensity": [0.0, 100.0, 0.0]}).to_parquet(
        paths["xrd_grid"] / "xrd_CrCoNi_fcc_000001.parquet"
    )
    pd.DataFrame({"two_theta": [20.02], "intensity": [100.0]}).to_parquet(
        paths["xrd_peaks"] / "xrd_CrCoNi_fcc_000001.parquet"
    )

    _write_catalog(paths["registry"])
    _write_initial_structures(paths["shared_manifests"], initial_doc, initial_bytes)
    _write_relaxations(paths["potential_manifests"])
    _write_relaxed_structures(paths["potential_manifests"], relaxed_doc, relaxed_bytes, cif_bytes)
    _write_xrd_samples(paths["potential_manifests"])
    _write_provenance_edges(paths["potential_manifests"])
    return root_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a minimal valid Alloy-XRD-Bench workspace.")
    parser.add_argument("--root", default="data/examples/minimal_workspace", help="Workspace root to create")
    args = parser.parse_args(argv)

    root = make_minimal_workspace(args.root)
    print(root)
    return 0


def _write_catalog(registry_dir: Path) -> None:
    catalog = {
        "potentials": [
            {
                "potential_id": POTENTIAL_ID,
                "family": "MACE",
                "engine": "mace",
                "elements": ["Cr", "Co", "Ni"],
                "phases": ["fcc"],
                "source_url": "https://example.org/mace",
                "doi": "10.0000/example",
                "license": "MIT",
                "file_sha256": "a" * 64,
                "redistribution_policy": "metadata_only",
                "citation_keys": ["mace"],
                "potential_dir": f"data/potentials/{POTENTIAL_ID}/potential",
                "generated_data_dir": f"data/potentials/{POTENTIAL_ID}/generated",
            }
        ]
    }
    (registry_dir / "potentials_catalog.yml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )


def _write_initial_structures(manifest_dir: Path, initial_doc: dict[str, object], initial_bytes: bytes) -> None:
    pd.DataFrame(
        [
            {
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "alloy_system": "CrCoNi",
                "phase": "fcc",
                "composition": {"Cr": 0.33, "Co": 0.33, "Ni": 0.34},
                "supercell": "4x4x4",
                "generation_method": "sqs_like",
                "seed": 7,
                "structure_sha256": canonical_structure_hash(initial_doc),
                "artifact_file_sha256": hashlib.sha256(initial_bytes).hexdigest(),
                "artifact_path": "data/shared/structures_initial/init_CrCoNi_fcc_000001.json",
            }
        ]
    ).to_parquet(manifest_dir / "initial_structures.parquet")


def _write_relaxations(manifest_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "potential_id": POTENTIAL_ID,
                "engine": "mace",
                "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "converged": True,
                "fmax": 0.03,
                "stress": {"xx": 0.0, "yy": 0.0, "zz": 0.0},
                "volume_change": 0.04,
                "steps": 45,
                "log_path": (
                    f"data/potentials/{POTENTIAL_ID}/generated/relaxation_logs/"
                    "relax_CrCoNi_fcc_000001.jsonl"
                ),
            }
        ]
    ).to_parquet(manifest_dir / "relaxations.parquet")


def _write_relaxed_structures(
    manifest_dir: Path,
    relaxed_doc: dict[str, object],
    relaxed_bytes: bytes,
    cif_bytes: bytes,
) -> None:
    pd.DataFrame(
        [
            {
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "structure_sha256": canonical_structure_hash(relaxed_doc),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": (
                    f"data/potentials/{POTENTIAL_ID}/generated/structures_relaxed/"
                    "rstr_CrCoNi_fcc_000001.json"
                ),
                "artifact_path_cif": (
                    f"data/potentials/{POTENTIAL_ID}/generated/structures_relaxed/"
                    "rstr_CrCoNi_fcc_000001.cif"
                ),
                "quality_tier": "pass",
            }
        ]
    ).to_parquet(manifest_dir / "relaxed_structures.parquet")


def _write_xrd_samples(manifest_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "xrd_sample_id": "xrd_CrCoNi_fcc_000001",
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "xrd_config_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "two_theta_min": 20.0,
                "two_theta_max": 100.0,
                "step": 0.02,
                "wavelength": "Cu Kalpha",
                "broadening": "gaussian_fwhm_0p1",
                "grid_path": (
                    f"data/potentials/{POTENTIAL_ID}/generated/xrd/grid/"
                    "xrd_CrCoNi_fcc_000001.parquet"
                ),
                "peak_path": (
                    f"data/potentials/{POTENTIAL_ID}/generated/xrd/peaks/"
                    "xrd_CrCoNi_fcc_000001.parquet"
                ),
                "split": "train",
            }
        ]
    ).to_parquet(manifest_dir / "xrd_samples.parquet")


def _write_provenance_edges(manifest_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "source_id": "init_CrCoNi_fcc_000001",
                "source_type": "initial_structure",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "input_to",
            },
            {
                "source_id": POTENTIAL_ID,
                "source_type": "potential_file",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "used_by",
            },
            {
                "source_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "source_type": "relax_config",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "configured_by",
            },
            {
                "source_id": "relax_CrCoNi_fcc_000001",
                "source_type": "relaxation_run",
                "target_id": "rstr_CrCoNi_fcc_000001",
                "target_type": "relaxed_structure",
                "relation_type": "produced",
            },
            {
                "source_id": "rstr_CrCoNi_fcc_000001",
                "source_type": "relaxed_structure",
                "target_id": "xrd_CrCoNi_fcc_000001",
                "target_type": "xrd_sample",
                "relation_type": "simulated_as",
            },
            {
                "source_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "source_type": "xrd_config",
                "target_id": "xrd_CrCoNi_fcc_000001",
                "target_type": "xrd_sample",
                "relation_type": "configured_by",
            },
        ]
    ).to_parquet(manifest_dir / "provenance_edges.parquet")


if __name__ == "__main__":
    raise SystemExit(main())
