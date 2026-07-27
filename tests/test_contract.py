import json
import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from alloy_xrd_bench.contract import (
    DEFAULT_RELAX_CONFIG,
    DEFAULT_XRD_CONFIG,
    MANIFEST_SCHEMAS,
    canonical_structure_hash,
    potential_workspace_paths,
    validate_artifact_hashes,
    validate_manifest_records,
    validate_potential_scoped_artifacts,
    validate_potential_storage_layout,
    validate_relaxation_quality,
    validate_provenance_graph,
    validate_release_potentials,
    validate_split_isolation,
    validate_structure_json_hashes,
    validate_xrd_geometry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkContractTests(unittest.TestCase):
    def test_manifest_schemas_include_public_contract_fields(self):
        expected_fields = {
            "potentials_catalog": {
                "potential_id",
                "family",
                "engine",
                "elements",
                "phases",
                "source_url",
                "doi",
                "license",
                "file_sha256",
                "redistribution_policy",
                "citation_keys",
                "potential_dir",
                "generated_data_dir",
            },
            "initial_structures": {
                "initial_structure_id",
                "alloy_system",
                "phase",
                "composition",
                "supercell",
                "generation_method",
                "seed",
                "structure_sha256",
                "artifact_path",
            },
            "relaxations": {
                "relaxation_run_id",
                "initial_structure_id",
                "potential_id",
                "engine",
                "relax_config_id",
                "converged",
                "fmax",
                "stress",
                "volume_change",
                "steps",
                "log_path",
            },
            "relaxed_structures": {
                "relaxed_structure_id",
                "relaxation_run_id",
                "structure_sha256",
                "artifact_path_json",
                "artifact_path_cif",
                "quality_tier",
            },
            "xrd_samples": {
                "xrd_sample_id",
                "relaxed_structure_id",
                "xrd_config_id",
                "two_theta_min",
                "two_theta_max",
                "step",
                "wavelength",
                "broadening",
                "grid_path",
                "peak_path",
                "split",
            },
            "provenance_edges": {
                "source_id",
                "source_type",
                "target_id",
                "target_type",
                "relation_type",
            },
        }

        for manifest_name, field_names in expected_fields.items():
            with self.subTest(manifest_name=manifest_name):
                self.assertIn(manifest_name, MANIFEST_SCHEMAS)
                actual_names = {field.name for field in MANIFEST_SCHEMAS[manifest_name].fields}
                self.assertTrue(field_names <= actual_names)

    def test_schema_json_files_match_runtime_contract(self):
        for manifest_name, schema in MANIFEST_SCHEMAS.items():
            with self.subTest(manifest_name=manifest_name):
                schema_path = REPO_ROOT / "schemas" / f"{manifest_name}.schema.json"
                self.assertTrue(schema_path.exists(), f"missing {schema_path}")
                payload = json.loads(schema_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest_name, payload["name"])
                self.assertEqual(
                    [field.name for field in schema.fields],
                    [field["name"] for field in payload["fields"]],
                )

    def test_default_configs_encode_phase_one_thresholds(self):
        self.assertEqual(
            {
                "xrd_config_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "radiation": "Cu Kalpha",
                "two_theta_min": 20.0,
                "two_theta_max": 100.0,
                "step": 0.02,
                "broadening": {"kind": "gaussian", "fwhm_deg": 0.1},
                "intensity_normalization": {"method": "max", "max_intensity": 100.0},
            },
            DEFAULT_XRD_CONFIG,
        )
        self.assertEqual(
            {
                "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "fmax_eV_per_A": 0.05,
                "max_steps": 1000,
                "abnormal_volume_change_threshold": 0.20,
            },
            DEFAULT_RELAX_CONFIG,
        )

    def test_canonical_structure_hash_is_order_stable(self):
        structure_a = {
            "@module": "pymatgen.core.structure",
            "lattice": {"b": 2.0, "a": 1.0},
            "sites": [
                {
                    "abc": [0.0, 0.0, 0.0],
                    "species": [{"element": "Ni", "occu": 1.0}],
                }
            ],
        }
        structure_b = {
            "sites": [
                {
                    "species": [{"occu": 1.0, "element": "Ni"}],
                    "abc": [0.0, 0.0, 0.0],
                }
            ],
            "lattice": {"a": 1.0, "b": 2.0},
            "@module": "pymatgen.core.structure",
        }

        digest = canonical_structure_hash(structure_a)

        self.assertEqual(digest, canonical_structure_hash(structure_b))
        self.assertEqual(64, len(digest))

    def test_manifest_records_reject_missing_required_fields_and_duplicate_ids(self):
        records = [
            {
                "xrd_sample_id": "xrd_CrCoNi_fcc_000001",
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "xrd_config_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "two_theta_min": 20.0,
                "two_theta_max": 100.0,
                "step": 0.02,
                "wavelength": "Cu Kalpha",
                "broadening": "gaussian_fwhm_0p1",
                "grid_path": "xrd/grid/xrd_CrCoNi_fcc_000001.parquet",
                "peak_path": "xrd/peaks/xrd_CrCoNi_fcc_000001.parquet",
                "split": "train",
            },
            {
                "xrd_sample_id": "xrd_CrCoNi_fcc_000001",
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000002",
            },
        ]

        issues = validate_manifest_records("xrd_samples", records)
        codes = {issue.code for issue in issues}

        self.assertIn("duplicate_id", codes)
        self.assertIn("missing_required_field", codes)

    def test_provenance_graph_requires_unique_xrd_to_structure_chain(self):
        manifests = valid_minimal_manifests()

        self.assertEqual([], validate_provenance_graph(manifests))

        broken = deepcopy(manifests)
        broken["relaxations"][0]["potential_id"] = "pot_missing"
        issues = validate_provenance_graph(broken)

        self.assertEqual(["unknown_potential"], [issue.code for issue in issues])

    def test_provenance_graph_requires_config_edges(self):
        manifests = valid_minimal_manifests()
        manifests["provenance_edges"] = [
            edge
            for edge in manifests["provenance_edges"]
            if edge["source_type"] not in {"relax_config", "xrd_config"}
        ]

        issues = validate_provenance_graph(manifests)

        self.assertEqual(
            ["missing_provenance_edge", "missing_provenance_edge"],
            [issue.code for issue in issues],
        )

    def test_private_split_cannot_share_relaxed_structure_with_train(self):
        xrd_samples = [
            {
                "xrd_sample_id": "xrd_train",
                "relaxed_structure_id": "rstr_shared",
                "split": "train",
            },
            {
                "xrd_sample_id": "xrd_private",
                "relaxed_structure_id": "rstr_shared",
                "split": "private_test",
            },
        ]

        issues = validate_split_isolation(xrd_samples)

        self.assertEqual(["private_relaxed_structure_leak"], [issue.code for issue in issues])

    def test_release_policy_blocks_metadata_only_potential_files(self):
        potentials = [
            {
                "potential_id": "pot_metadata_only",
                "redistribution_policy": "metadata_only",
            },
            {
                "potential_id": "pot_packaged",
                "redistribution_policy": "redistributable",
            },
        ]
        release_paths = [
            "release/potentials/pot_metadata_only/model.pt",
            "release/potentials/pot_packaged/model.pt",
        ]

        issues = validate_release_potentials(potentials, release_paths)

        self.assertEqual(["non_redistributable_potential_packaged"], [issue.code for issue in issues])

    def test_potential_workspace_paths_are_isolated_by_potential_id(self):
        self.assertEqual(
            {
                "root": "data/potentials/pot_mace_mp_0",
                "potential_dir": "data/potentials/pot_mace_mp_0/potential",
                "generated_data_dir": "data/potentials/pot_mace_mp_0/generated",
                "manifests_dir": "data/potentials/pot_mace_mp_0/generated/manifests",
                "relaxation_logs_dir": "data/potentials/pot_mace_mp_0/generated/relaxation_logs",
                "relaxed_structures_dir": "data/potentials/pot_mace_mp_0/generated/structures_relaxed",
                "xrd_grid_dir": "data/potentials/pot_mace_mp_0/generated/xrd/grid",
                "xrd_peaks_dir": "data/potentials/pot_mace_mp_0/generated/xrd/peaks",
                "xrd_average_grid_dir": "data/potentials/pot_mace_mp_0/generated/xrd/averages/grid",
                "xrd_average_peaks_dir": "data/potentials/pot_mace_mp_0/generated/xrd/averages/peaks",
                "audits_dir": "data/potentials/pot_mace_mp_0/audits",
            },
            potential_workspace_paths("pot_mace_mp_0"),
        )

    def test_potential_catalog_records_declare_their_storage_dirs(self):
        potentials = valid_minimal_manifests()["potentials_catalog"]

        self.assertEqual([], validate_potential_storage_layout(potentials))

        broken = deepcopy(potentials)
        broken[0]["generated_data_dir"] = "data/generated/pot_mace_mp_0"

        issues = validate_potential_storage_layout(broken)

        self.assertEqual(["invalid_generated_data_dir"], [issue.code for issue in issues])

    def test_generated_artifacts_must_live_under_their_potential_folder(self):
        manifests = valid_minimal_manifests()

        self.assertEqual([], validate_potential_scoped_artifacts(manifests))

        broken = deepcopy(manifests)
        broken["xrd_samples"][0]["grid_path"] = "data/potentials/pot_other/generated/xrd/grid/xrd.parquet"

        issues = validate_potential_scoped_artifacts(broken)

        self.assertEqual(["artifact_outside_potential_scope"], [issue.code for issue in issues])

    def test_artifact_hashes_match_manifest_records(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            artifact = root / "structures" / "relaxed.json"
            artifact.parent.mkdir(parents=True)
            payload = b'{"sites":[]}'
            artifact.write_bytes(payload)

            records = [
                {
                    "relaxed_structure_id": "rstr_hash_ok",
                    "artifact_path_json": "structures/relaxed.json",
                    "structure_sha256": hashlib.sha256(payload).hexdigest(),
                },
                {
                    "relaxed_structure_id": "rstr_hash_bad",
                    "artifact_path_json": "structures/relaxed.json",
                    "structure_sha256": "0" * 64,
                },
                {
                    "relaxed_structure_id": "rstr_missing",
                    "artifact_path_json": "structures/missing.json",
                    "structure_sha256": "1" * 64,
                },
            ]

            issues = validate_artifact_hashes(
                records,
                hash_field="structure_sha256",
                path_field="artifact_path_json",
                root=root,
                id_field="relaxed_structure_id",
            )

        self.assertEqual(["artifact_hash_mismatch", "missing_artifact"], [issue.code for issue in issues])

    def test_structure_json_hashes_separate_canonical_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            artifact = root / "structure.json"
            structure = {"sites": [], "lattice": {"a": 1.0}}
            artifact.write_text(json.dumps(structure, indent=2), encoding="utf-8")
            good = {
                "initial_structure_id": "init_good",
                "artifact_path": "structure.json",
                "structure_sha256": canonical_structure_hash(structure),
                "artifact_file_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            bad = dict(good)
            bad["initial_structure_id"] = "init_bad"
            bad["structure_sha256"] = "0" * 64
            bad["artifact_file_sha256"] = "1" * 64

            issues = validate_structure_json_hashes(
                [good, bad],
                root=root,
                path_field="artifact_path",
                id_field="initial_structure_id",
                file_hash_field="artifact_file_sha256",
            )

        self.assertEqual(
            ["structure_hash_mismatch", "artifact_hash_mismatch"],
            [issue.code for issue in issues],
        )

    def test_relaxation_quality_flags_threshold_violations(self):
        records = [
            {
                "relaxation_run_id": "relax_bad",
                "converged": True,
                "fmax": 0.06,
                "volume_change": 0.21,
                "steps": 1001,
            }
        ]

        issues = validate_relaxation_quality(records)

        self.assertEqual(
            [
                "relaxation_fmax_above_default",
                "abnormal_volume_change",
                "relaxation_steps_above_default",
            ],
            [issue.code for issue in issues],
        )

    def test_xrd_geometry_flags_empty_or_out_of_range_grids(self):
        records = [
            {
                "xrd_sample_id": "xrd_bad",
                "two_theta_min": 100.0,
                "two_theta_max": 20.0,
                "step": 0.0,
            }
        ]

        issues = validate_xrd_geometry(records)

        self.assertEqual(
            ["invalid_two_theta_range", "invalid_two_theta_step"],
            [issue.code for issue in issues],
        )


def valid_minimal_manifests():
    return {
        "potentials_catalog": [
            {
                "potential_id": "pot_mace_mp_0",
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
                "potential_dir": "data/potentials/pot_mace_mp_0/potential",
                "generated_data_dir": "data/potentials/pot_mace_mp_0/generated",
            }
        ],
        "initial_structures": [
            {
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "alloy_system": "CrCoNi",
                "phase": "fcc",
                "composition": {"Cr": 0.33, "Co": 0.33, "Ni": 0.34},
                "supercell": "4x4x4",
                "generation_method": "sqs_like",
                "seed": 7,
                "structure_sha256": "b" * 64,
                "artifact_path": "data/shared/structures_initial/init_CrCoNi_fcc_000001.json",
            }
        ],
        "relaxations": [
            {
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "potential_id": "pot_mace_mp_0",
                "engine": "mace",
                "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "converged": True,
                "fmax": 0.03,
                "stress": {"xx": 0.0, "yy": 0.0, "zz": 0.0},
                "volume_change": 0.04,
                "steps": 45,
                "log_path": (
                    "data/potentials/pot_mace_mp_0/generated/relaxation_logs/"
                    "relax_CrCoNi_fcc_000001.jsonl"
                ),
            }
        ],
        "relaxed_structures": [
            {
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "structure_sha256": "c" * 64,
                "artifact_path_json": (
                    "data/potentials/pot_mace_mp_0/generated/structures_relaxed/"
                    "rstr_CrCoNi_fcc_000001.json"
                ),
                "artifact_path_cif": (
                    "data/potentials/pot_mace_mp_0/generated/structures_relaxed/"
                    "rstr_CrCoNi_fcc_000001.cif"
                ),
                "quality_tier": "pass",
            }
        ],
        "xrd_samples": [
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
                    "data/potentials/pot_mace_mp_0/generated/xrd/grid/"
                    "xrd_CrCoNi_fcc_000001.parquet"
                ),
                "peak_path": (
                    "data/potentials/pot_mace_mp_0/generated/xrd/peaks/"
                    "xrd_CrCoNi_fcc_000001.parquet"
                ),
                "split": "train",
            }
        ],
        "provenance_edges": [
            {
                "source_id": "init_CrCoNi_fcc_000001",
                "source_type": "initial_structure",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "input_to",
            },
            {
                "source_id": "pot_mace_mp_0",
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
        ],
    }
