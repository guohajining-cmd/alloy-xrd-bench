import json
import subprocess
import sys
import tempfile
import types
import unittest
import hashlib
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import yaml

from alloy_xrd_bench.mlip import LoadedInitial, MaceAdapter, MatglAdapter, NepAdapter, download_candidate


class MlipAdmissionCliTests(unittest.TestCase):
    def test_project_registry_tracks_exact_specialized_mlip_candidates(self):
        payload = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in payload["candidates"]}

        expected = {
            "pot_crconi_mtp_csro": ("target_specific_mlip", "mtp_mlp", {"Cr", "Co", "Ni"}),
            "pot_crconi_ace_sheriff_motif_2025": (
                "target_specific_mlip",
                "pace_lammps",
                {"Cr", "Co", "Ni"},
            ),
            "pot_crconi_nep_wu_2026": (
                "target_specific_mlip",
                "nep",
                {"Cr", "Co", "Ni"},
            ),
            "pot_cocrfemnni_mtp_cantor_alloy_mlip": (
                "target_specific_mlip",
                "mtp_lammps",
                {"Co", "Cr", "Fe", "Mn", "Ni"},
            ),
            "pot_cocrfeni_cocrni_mtp_nitol_2026": (
                "target_specific_mlip",
                "mtp_lammps",
                {"Co", "Cr", "Fe", "Ni"},
            ),
            "pot_monbtavw_tabgap_byggmastar": (
                "target_specific_mlip",
                "tabgap_lammps",
                {"Mo", "Nb", "Ta", "V", "W"},
            ),
            "pot_monbtavw_mtp_wang_2024": (
                "target_specific_mlip",
                "mtp_lammps",
                {"Mo", "Nb", "Ta", "V", "W"},
            ),
            "pot_monbtaw_mtp_ufsf_2025": (
                "refractory_alloy_specific_mlip",
                "mtp_lammps",
                {"Mo", "Nb", "Ta", "W"},
            ),
            "pot_nbmotaw_snap_mvl_2021": (
                "refractory_alloy_specific_mlip",
                "snap_lammps",
                {"Mo", "Nb", "Ta", "W"},
            ),
            "pot_monbtaw_mtp_mvl_2021": (
                "refractory_alloy_specific_mlip",
                "mtp_lammps",
                {"Mo", "Nb", "Ta", "W"},
            ),
            "pot_monbtavw_dpa2_lam_jiang_2026": (
                "refractory_alloy_specific_mlip",
                "deepmd_lammps",
                {"V", "Nb", "Mo", "Ta", "W"},
            ),
            "pot_monbtavw_dpa2_lam_all_jiang_2026": (
                "refractory_alloy_specific_mlip",
                "deepmd_lammps",
                {"V", "Nb", "Mo", "Ta", "W"},
            ),
            "pot_monbtavw_dp_lam_jiang_2026": (
                "refractory_alloy_specific_mlip",
                "deepmd_lammps",
                {"V", "Nb", "Mo", "Ta", "W"},
            ),
            "pot_monbtavw_dp_lam_all_jiang_2026": (
                "refractory_alloy_specific_mlip",
                "deepmd_lammps",
                {"V", "Nb", "Mo", "Ta", "W"},
            ),
            "pot_nicofecraltita_dpa3_lam_jiang_2026_lead": (
                "gamma_prime_specific_candidate",
                "deepmd_lammps",
                {"Ti", "Ta", "Al", "Cr", "Fe", "Ni", "Co"},
            ),
            "pot_nialti_mace_fitsnap_song_2026_lead": (
                "gamma_prime_specific_candidate",
                "mace_or_fitsnap_lammps",
                {"Ni", "Al", "Ti"},
            ),
            "pot_nibased_gamma_prime_mou_2023_lead": (
                "gamma_prime_specific_candidate",
                "deepmd_or_nequip_or_mtp",
                {"Ni", "Al"},
            ),
            "pot_nicoal_nep_qin_2025_lead": (
                "gamma_prime_specific_candidate",
                "gpumd",
                {"Ni", "Co", "Al"},
            ),
            "pot_nialta_eam_fan_2016_classical_lead": (
                "classical_fallback",
                "eam_lammps",
                {"Ni", "Al", "Ta"},
            ),
            "pot_nicrfealp_mtp_ning_2026_lead": (
                "nickel_alloy_specific_candidate",
                "mtp_lammps",
                {"Ni", "Cr", "Fe", "Al", "P"},
            ),
            "pot_nialti_meam_kim_2017_classical_fallback": (
                "classical_fallback",
                "meam_lammps",
                {"Ni", "Al", "Ti"},
            ),
            "pot_nialcoti_meam_huang_2024_classical_fallback": (
                "classical_fallback",
                "meam_lammps",
                {"Ni", "Al", "Co", "Ti"},
            ),
            "pot_wtacrv_tabgap_byggmastar_2025": (
                "target_specific_mlip",
                "tabgap_lammps",
                {"W", "Ta", "Cr", "V"},
            ),
            "pot_nial_mtp_wang_2025_candidate": (
                "target_specific_mlip",
                "mtp_mlp",
                {"Al", "Ni"},
            ),
            "pot_nial_hdnnp_sandberg_2024": (
                "target_specific_mlip",
                "n2p2_lammps",
                {"Al", "Ni"},
            ),
            "pot_ni3al_mtp_darus_yield_anomaly_2024": (
                "gamma_prime_specific_mlip",
                "mtp_mlp",
                {"Al", "Ni"},
            ),
            "pot_ni3al_mtp_darus_spin_fluctuations_2023": (
                "gamma_prime_specific_mlip",
                "mtp_lammps",
                {"Al", "Ni"},
            ),
            "pot_altiv_mtp_nitol_2025": (
                "supporting_specialized_mlip_partial_coverage",
                "mtp_lammps",
                {"Al", "Ti", "V"},
            ),
            "pot_tial_polymlp_seko_2022p3_openkim": (
                "supporting_specialized_mlip_partial_coverage",
                "openkim_polymlp",
                {"Al", "Ti"},
            ),
            "pot_tial_mtp_mvl_2023": (
                "supporting_specialized_mlip_partial_coverage",
                "mtp_lammps",
                {"Al", "Ti"},
            ),
            "pot_titavw_ace_sheriff_motif_2025": (
                "target_specific_mlip",
                "pace_lammps",
                {"Ti", "Ta", "V", "W"},
            ),
            "pot_titavw_ace_sheriff_chemical_mixing_2025": (
                "target_specific_mlip",
                "pace_lammps",
                {"Ti", "Ta", "V", "W"},
            ),
            "pot_titavw_milady_wrobel_2025": (
                "target_specific_mlip",
                "milady_lammps",
                {"Ti", "Ta", "V", "W"},
            ),
            "pot_rhea9_nep_byggmastar_2026": (
                "refractory_alloy_specific_mlip",
                "gpumd",
                {"Cr", "Hf", "Mo", "Nb", "Ta", "Ti", "V", "W", "Zr"},
            ),
            "pot_rhea9_tabgap_byggmastar_2026": (
                "refractory_alloy_specific_mlip",
                "tabgap_lammps",
                {"Cr", "Hf", "Mo", "Nb", "Ta", "Ti", "V", "W", "Zr"},
            ),
            "pot_unep_v1_16metals_without_zbl": (
                "metal_domain_baseline",
                "gpumd",
                {"Ag", "Al", "Au", "Cr", "Cu", "Mg", "Mo", "Ni", "Pb", "Pd", "Pt", "Ta", "Ti", "V", "W", "Zr"},
            ),
            "pot_unep_v1_16metals_with_zbl": (
                "metal_domain_baseline",
                "gpumd",
                {"Ag", "Al", "Au", "Cr", "Cu", "Mg", "Mo", "Ni", "Pb", "Pd", "Pt", "Ta", "Ti", "V", "W", "Zr"},
            ),
            "pot_alloy53_dpa1_vasp_dai_2026_aissquare": (
                "metal_domain_baseline",
                "deepmd_lammps",
                {"Al", "Ti", "Ni", "Ta", "Mo", "Nb", "W", "Cr", "Co", "Fe", "Mn", "V"},
            ),
            "pot_alloy53_dpa1_2_multitask_wen_2024_aissquare": (
                "metal_domain_baseline",
                "deepmd_lammps",
                {"Al", "Ti", "Ni", "Ta", "Mo", "Nb", "W", "Cr", "Co", "Fe", "Mn", "V"},
            ),
            "pot_nep89_gpumd_2025": (
                "general_baseline",
                "gpumd",
                {"Al", "Ti", "Ni", "Ta", "Mo", "Nb", "W", "Cr", "Co", "Fe", "Mn", "V"},
            ),
            "pot_alcocrfeni_lightpfp_2025_lead": (
                "hea_specific_candidate",
                "commercial_lightpfp",
                {"Al", "Co", "Cr", "Fe", "Ni"},
            ),
            "pot_heaml_pointff_wu_2024_lead": (
                "hea_specific_candidate",
                "pointff_lammps",
                {"Cr", "Fe", "Co", "Ni", "Pd"},
            ),
        }

        for candidate_id, (target_role, engine, elements) in expected.items():
            with self.subTest(candidate_id=candidate_id):
                row = candidates[candidate_id]
                self.assertEqual(target_role, row["target_role"])
                self.assertEqual(engine, row["engine"])
                self.assertTrue(elements.issubset(set(row["elements"])))
                self.assertIn(row["acceptance_status"], {"candidate", "blocked", "accepted", "research_needed"})

    def test_openkim_target_species_audit_keeps_baseline_separate_from_specialized_mlip(self):
        audit = json.loads(
            Path("audits/mlip_openkim_target_species_query_2026-07-04.json").read_text(encoding="utf-8")
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        exact_queries = {tuple(row["species"]): row for row in audit["queries"]}
        nialtita_models = set(exact_queries[("Ni", "Al", "Ti", "Ta")]["models"])
        reinforced = set(audit["conclusion"]["blocked_candidates_reinforced"])

        self.assertFalse(audit["conclusion"]["dedicated_gamma_prime_mlip_found"])
        self.assertIn("TorchML_MACE_BatatiaBennerChiang_2023_MP0a_medium__MO_568776921807_000", nialtita_models)
        self.assertEqual(
            {
                "pot_nialti_mlip_candidate",
                "pot_nialta_mlip_candidate",
                "pot_nialtita_mlip_candidate",
            },
            reinforced,
        )
        for candidate_id in reinforced:
            with self.subTest(candidate_id=candidate_id):
                row = candidates[candidate_id]
                self.assertEqual("research_needed", row["acceptance_status"])
                self.assertIn("OpenKIM", row["rejection_reason"])
                self.assertIn("general baseline", row["rejection_reason"])

    def test_aissquare_alloy53_audit_tracks_auth_download_blocker(self):
        audit = json.loads(Path("audits/mlip_aissquare_alloy53_dpa_2026_audit.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        query_names = {
            item["name"]
            for query in audit["aissquare_api"]["queries"]
            for item in query["items"]
        }
        detail_statuses = {row["http_status"] for row in audit["aissquare_api"]["anonymous_detail_access"]}

        self.assertTrue(audit["classification"]["not_gamma_prime_specific"])
        self.assertIn("Alloy53_VASP", query_names)
        self.assertIn("DPA-1&2-53-alloy-multitask-400w[EN]", query_names)
        self.assertEqual({401}, detail_statuses)
        for candidate_id in (
            "pot_alloy53_dpa1_vasp_dai_2026_aissquare",
            "pot_alloy53_dpa1_2_multitask_wen_2024_aissquare",
        ):
            with self.subTest(candidate_id=candidate_id):
                row = candidates[candidate_id]
                self.assertEqual("research_needed", row["acceptance_status"])
                self.assertEqual("aissquare_model_entry_auth_required", row["artifact_kind"])
                self.assertIn("gamma/gamma-prime", row["rejection_reason"])

    def test_lam_dpa_dpa3_audit_records_config_but_no_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_specialized_search_lam_dpa_2026-07-04.json").read_text(encoding="utf-8")
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nicofecraltita_dpa3_lam_jiang_2026_lead"
        ]

        tree = audit["source_repository"]["github_tree_inspection"]
        finding = {
            row["candidate_id"]: row
            for row in audit["gamma_prime_findings"]
        }["pot_nicofecraltita_dpa3_lam_jiang_2026_lead"]
        config = finding["public_config_files"][0]

        self.assertFalse(tree["dpa3_fcc_l12_model_file_found"])
        self.assertEqual("training_config_only_no_checkpoint", candidate["artifact_kind"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual(
            "4d3060738a7a5d2ac3f28994bbe60d69481d4e60c6f1368397db10ccd3849b4e",
            config["sha256"],
        )
        self.assertEqual({"Ti", "Ta", "Al", "Cr", "Fe", "Ni", "Co"}, set(config["type_map"]))
        self.assertIn("no trained FCC+L12 checkpoint", candidate["rejection_reason"])

    def test_round29_rechecks_lam_dpa_gamma_prime_lead_and_engine_blockers(self):
        audit = json.loads(
            Path("audits/mlip_lam_dpa_gamma_prime_checkpoint_and_engine_probe_round29_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertEqual("pot_nicofecraltita_dpa3_lam_jiang_2026_lead", summary["existing_candidate_rechecked"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["new_checkpoint_found"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertFalse(summary["engine_smoke_possible_now"])

        tree = audit["github_tree_probe"]
        self.assertEqual(39, tree["total_blobs"])
        self.assertEqual(21, tree["model_like_paths_count"])
        self.assertIn("complex-structures/dpa3-FCC+L12/input_dpa3ft.json", tree["complex_structure_paths"])
        self.assertEqual([], tree["fcc_l12_checkpoint_paths"])
        self.assertTrue(tree["fcc_l12_config_only"])

        config = audit["fcc_l12_config_evidence"]
        self.assertEqual({"Ti", "Ta", "Al", "Cr", "Fe", "Ni", "Co"}, set(config["type_map"]))
        self.assertEqual("dpa3", config["descriptor_type"])
        self.assertFalse(config["trained_model_path_declared"])

        engine = audit["local_engine_probe"]
        self.assertEqual("NVIDIA GeForce RTX 4060", engine["nvidia_gpu"])
        for key in ["gpumd_on_path", "nep_on_path", "lmp_on_path", "deepmd_dp_on_path", "nvcc_on_path"]:
            with self.subTest(key=key):
                self.assertFalse(engine[key])

        candidate = candidates["pot_nicofecraltita_dpa3_lam_jiang_2026_lead"]
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertIn("Round-29", candidate["rejection_reason"])
        self.assertIn("no trained checkpoint", candidates["pot_nialtita_mlip_candidate"]["rejection_reason"])

    def test_round30_deepmd_lammps_adapter_readiness_keeps_candidates_unaccepted(self):
        audit = json.loads(
            Path("audits/mlip_deepmd_lammps_adapter_readiness_round30_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["adapter_added"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertEqual(3, summary["candidate_status_counts"]["accepted"])
        self.assertTrue(summary["deepmd_candidates_remain_blocked_or_research_needed"])

        adapter = audit["adapter_readiness"]
        self.assertEqual("deepmd_lammps", adapter["adapter_name"])
        self.assertIn("deepmd_lammps", adapter["engines_mapped"])
        self.assertIn("dpa_lammps", adapter["engines_mapped"])
        self.assertIn("*.pth", adapter["model_globs"])
        self.assertIn("*dpa*", adapter["model_globs"])

        runtime = audit["runtime_status"]
        for key in [
            "python_module_deepmd_installed",
            "python_module_lammps_installed",
            "lmp_on_path",
            "deepmd_dp_on_path",
            "gpumd_on_path",
            "nep_on_path",
        ]:
            with self.subTest(key=key):
                self.assertFalse(runtime[key])
        self.assertFalse(runtime["pip_dry_run"]["install_performed"])
        self.assertEqual("resolved", runtime["pip_dry_run"]["resolution_status"])

        for candidate_id in audit["candidate_ids_unblocked_at_adapter_layer"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertNotEqual("accepted", candidates[candidate_id]["acceptance_status"])
                self.assertEqual("deepmd_lammps", candidates[candidate_id]["engine"])

        counts = audit["formal_manifest_counts_before_and_after"]
        self.assertEqual(123, counts["relaxations"])
        self.assertEqual(123, counts["relaxed_structures"])
        self.assertEqual(14, counts["xrd_samples"])

    def test_round31_records_gamma_prime_search_and_deepmd_runtime_gates(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_search_and_deepmd_runtime_round31_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertTrue(summary["deepmd_lammps_packages_installed"])
        self.assertFalse(summary["deepmd_pair_style_available"])
        self.assertFalse(summary["deepmd_pytorch_model_show_ok"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertFalse(summary["accepted_potential_created"])

        hf_queries = audit["public_index_search"]["hugging_face"]["queries"]
        self.assertTrue(any(row["query"] == "NiAlTi" and row["target_model_hits"] == 0 for row in hf_queries))
        datacite_queries = audit["public_index_search"]["datacite"]["queries"]
        self.assertTrue(any(row.get("known_existing_hit") == "10.18419/darus-3579" for row in datacite_queries))

        runtime = audit["deepmd_runtime_probe"]
        self.assertEqual("3.1.3", runtime["installed_packages"]["deepmd-kit"])
        self.assertEqual("2025.7.22.4.0", runtime["installed_packages"]["lammps"])
        self.assertTrue(runtime["python_imports"]["deepmd"]["ok"])
        self.assertTrue(runtime["python_imports"]["lammps"]["ok"])
        self.assertEqual("unrecognized_pair_style", runtime["lammps_pair_style_probe"]["error_code"])
        self.assertEqual("pytorch_cxx11_abi_mismatch", runtime["deepmd_pytorch_probe"]["error_code"])
        style_probe = runtime["lammps_optional_pair_style_probe"]
        self.assertIn("snap", style_probe["available_or_parse_reached"])
        self.assertIn("meam", style_probe["available_or_parse_reached"])
        blocked_styles = {row["pair_style"] for row in style_probe["not_enabled"]}
        self.assertEqual({"pace", "hdnnp", "quip"}, blocked_styles)

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertNotEqual("accepted", candidates[candidate_id]["acceptance_status"])
                self.assertIn("Round-31", candidates[candidate_id]["rejection_reason"])
        self.assertIn("pair_style deepmd", candidates["pot_monbtavw_dpa2_lam_jiang_2026"]["rejection_reason"])
        self.assertIn("ML-PACE", candidates["pot_crconi_ace_sheriff_motif_2025"]["rejection_reason"])
        self.assertIn("ML-HDNNP", candidates["pot_nial_hdnnp_sandberg_2024"]["rejection_reason"])
        self.assertIn("no new", candidates["pot_nialtita_mlip_candidate"]["rejection_reason"])

    def test_round32_snap_lammps_supporting_candidate_smoke_is_partial_coverage_only(self):
        audit = json.loads(
            Path("audits/mlip_snap_lammps_supporting_candidate_round32_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["adapter_added"])
        self.assertEqual("snap_lammps", summary["adapter_name"])
        self.assertTrue(summary["installed_lammps_snap_available"])
        self.assertFalse(summary["original_param_file_run0_passed"])
        self.assertTrue(summary["compat_param_file_run0_passed"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertFalse(summary["target_coverage_complete_for_mo_nb_ta_v_w"])

        scope = audit["candidate_scope"]
        self.assertEqual({"Nb", "Mo", "Ta", "W"}, set(scope["elements"]))
        self.assertEqual(["V"], scope["missing_target_elements_for_mo_nb_ta_v_w"])
        self.assertEqual("unknown_snap_parameter_diagonalstyle", audit["original_parameter_file_probe"]["error_code"])
        self.assertEqual("pass", audit["compat_parameter_file_probe"]["status"])
        self.assertEqual(26.272449, audit["compat_parameter_file_probe"]["thermo_observation"]["potential_energy_eV"])

        candidate = candidates["pot_nbmotaw_snap_mvl_2021"]
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertIn("Round-32", candidate["rejection_reason"])
        self.assertIn("lacks V", candidate["rejection_reason"])
        self.assertIn("temporary compatibility", candidates["pot_snap_hea_candidate"]["rejection_reason"])

    def test_round33_nep_mdapy_runtime_records_reference_inference_without_acceptance(self):
        audit = json.loads(
            Path("audits/mlip_nep_mdapy_runtime_round33_2026-07-04.json").read_text(encoding="utf-8")
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["mdapy_nep4ase_available"])
        self.assertTrue(summary["adapter_support_added"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertFalse(summary["dedicated_gamma_prime_checkpoint_found"])

        probes = {row["candidate_id"]: row for row in audit["mdapy_nep4ase_single_point_probes"]}
        self.assertEqual("pass", probes["pot_rhea9_nep_byggmastar_2026"]["status"])
        self.assertEqual("pass", probes["pot_unep_v1_16metals_with_zbl"]["status"])
        self.assertEqual("pass", probes["pot_nep89_gpumd_2025"]["status"])
        self.assertIn("Ni-Al-Ti-Ta", probes["pot_unep_v1_16metals_with_zbl"]["probe_system"])

        self.assertEqual("accepted", candidates["pot_rhea9_nep_byggmastar_2026"]["acceptance_status"])
        self.assertIn("Round-38", candidates["pot_rhea9_nep_byggmastar_2026"]["rejection_reason"])
        self.assertIn("mdapy.NEP4ASE", candidates["pot_unep_v1_16metals_with_zbl"]["rejection_reason"])
        self.assertNotEqual("accepted", candidates["pot_nep89_gpumd_2025"]["acceptance_status"])

    def test_round34_records_gamma_prime_checkpoint_misses_without_accepting_leads(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round34_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])

        searches = audit["public_index_queries"]
        self.assertEqual(0, searches["hugging_face"]["target_model_hits"])
        self.assertEqual(0, searches["github_repository_search"]["target_repository_hits"])
        self.assertEqual(0, searches["datacite"]["target_dataset_hits"])
        self.assertEqual(0, searches["zenodo"]["target_model_hits"])

        lam_dpa = {
            row["candidate_id"]: row for row in audit["primary_lead_rechecks"]
        }["pot_nicofecraltita_dpa3_lam_jiang_2026_lead"]
        self.assertEqual(0, lam_dpa["github_probe"]["releases_count"])
        self.assertEqual(["main"], lam_dpa["github_probe"]["branches"])
        self.assertEqual(
            ["complex-structures/dpa3-FCC+L12/input_dpa3ft.json"],
            [row["path"] for row in lam_dpa["github_probe"]["fcc_l12_paths"]],
        )

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertEqual("research_needed", candidates[candidate_id]["acceptance_status"])
                self.assertIn("Round-34", candidates[candidate_id]["rejection_reason"])

    def test_round35_mtp_engine_and_file_integrity_keeps_candidates_blocked(self):
        audit = json.loads(
            Path("audits/mlip_mtp_engine_and_file_integrity_round35_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertEqual(17, summary["downloaded_mtp_files_checked"])
        self.assertEqual(17, summary["mtp_header_parse_pass_count"])
        self.assertEqual(16, summary["trained_coefficients_detected_count"])
        self.assertEqual(1, summary["header_only_or_untrained_template_count"])
        self.assertFalse(summary["mlp_executable_on_path"])
        self.assertFalse(summary["cxx_toolchain_on_path"])
        self.assertFalse(summary["reference_inference_passed"])
        self.assertFalse(summary["smoke_relaxation_passed"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])

        rows = {
            (row["candidate_id"], Path(row["path"]).name): row
            for row in audit["mtp_file_integrity"]
        }
        crconi = rows[("pot_crconi_mtp_csro", "MTP_TS-3_Cao_Levmax_20.mtp")]
        self.assertEqual("pass", crconi["parse_status"])
        self.assertTrue(crconi["trained_coefficients_detected"])

        header_like = rows[("pot_ni3al_mtp_darus_spin_fluctuations_2023", "24.mtp")]
        self.assertEqual("header_only_or_untrained_template", header_like["parse_status"])
        self.assertFalse(header_like["trained_coefficients_detected"])

        trained = rows[("pot_ni3al_mtp_darus_spin_fluctuations_2023", "Trained.mtp_")]
        self.assertEqual("pass", trained["parse_status"])
        self.assertTrue(trained["trained_coefficients_detected"])

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                expected_status = (
                    "accepted"
                    if candidate_id
                    in {
                        "pot_crconi_mtp_csro",
                        "pot_ni3al_mtp_darus_yield_anomaly_2024",
                    }
                    else "blocked"
                )
                self.assertEqual(expected_status, candidates[candidate_id]["acceptance_status"])
                if candidate_id == "pot_crconi_mtp_csro":
                    self.assertIn("Round-50", candidates[candidate_id]["rejection_reason"])
                elif candidate_id == "pot_ni3al_mtp_darus_yield_anomaly_2024":
                    self.assertIn("round-51", candidates[candidate_id]["rejection_reason"].lower())
                else:
                    self.assertIn("Round-35", candidates[candidate_id]["rejection_reason"])

    def test_round36_accepts_crconi_nep_after_mdapy_reference_inference(self):
        audit = json.loads(
            Path("audits/mlip_crconi_nep_wu_2026_round36_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        catalog = yaml.safe_load(Path("data/registry/potentials_catalog.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}
        potentials = {row["potential_id"]: row for row in catalog["potentials"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_target_specific_mlip_found"])
        self.assertTrue(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertTrue(summary["reference_inference_passed"])
        self.assertTrue(summary["smoke_passed"])

        candidate = candidates["pot_crconi_nep_wu_2026"]
        self.assertEqual("accepted", candidate["acceptance_status"])
        self.assertEqual("MIT", candidate["license"])
        self.assertEqual("98e4e98e84007b794506c9e0c0ef5993beec19aff27bb667850fbbec35e142b7", candidate["artifact_hashes"]["model/nep.txt"]["sha256"])
        self.assertIn("Round-36", candidate["rejection_reason"])

        potential = potentials["pot_crconi_nep_wu_2026"]
        self.assertEqual("nep", potential["engine"])
        self.assertEqual("redistributable_with_attribution", potential["redistribution_policy"])
        self.assertEqual("98e4e98e84007b794506c9e0c0ef5993beec19aff27bb667850fbbec35e142b7", potential["file_sha256"])

        verification = json.loads(
            Path("data/potentials/pot_crconi_nep_wu_2026/audits/potential_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("pass", verification["status"])
        self.assertTrue(verification["load"]["ok"])
        self.assertEqual("pass", verification["reference_inference"]["status"])
        smoke_statuses = [row["status"] for row in verification["smoke_relaxations"]]
        self.assertEqual(["pass", "skipped_unsupported_elements", "skipped_unsupported_elements"], smoke_statuses)

    def test_round38_accepts_rhea9_nep_after_target_smoke(self):
        audit = json.loads(
            Path("audits/mlip_rhea9_nep_byggmastar_2026_round38_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        catalog = yaml.safe_load(Path("data/registry/potentials_catalog.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}
        potentials = {row["potential_id"]: row for row in catalog["potentials"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_refractory_alloy_specific_mlip_accepted"])
        self.assertTrue(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertTrue(summary["reference_inference_passed"])
        self.assertTrue(summary["smoke_passed"])
        self.assertEqual(["WTaCrV", "MoNbTaVW", "TiTaVW"], summary["target_systems_smoked"])

        candidate = candidates["pot_rhea9_nep_byggmastar_2026"]
        self.assertEqual("accepted", candidate["acceptance_status"])
        self.assertEqual("CC-BY-4.0", candidate["license"])
        self.assertEqual(
            "018bc44356b48ee370f51a8748f3aeb3ef94fb2319ef43edb570cf2e5e65a3df",
            candidate["artifact_hashes"]["potential_files/nep-RHEA-large.txt"]["sha256"],
        )
        self.assertIn("Round-38", candidate["rejection_reason"])

        potential = potentials["pot_rhea9_nep_byggmastar_2026"]
        self.assertEqual("gpumd", potential["engine"])
        self.assertEqual("redistributable_with_attribution", potential["redistribution_policy"])
        self.assertEqual(
            "018bc44356b48ee370f51a8748f3aeb3ef94fb2319ef43edb570cf2e5e65a3df",
            potential["file_sha256"],
        )

        verification = json.loads(
            Path("data/potentials/pot_rhea9_nep_byggmastar_2026/audits/potential_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("pass", verification["status"])
        self.assertTrue(verification["load"]["ok"])
        self.assertTrue(all(row["status"] == "pass" for row in verification["smoke_relaxations"]))

    def test_round39_adds_heaml_pointff_lead_without_accepting_it(self):
        audit = json.loads(
            Path("audits/mlip_heaml_pointff_wu_2024_round39_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_hea_related_model_file_found"])
        self.assertFalse(summary["new_exact_benchmark_target_mlip_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertTrue(summary["torch_jit_load_passed"])

        candidate = candidates["pot_heaml_pointff_wu_2024_lead"]
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertEqual("PointFF/TorchScript", candidate["family"])
        self.assertEqual("pointff_lammps", candidate["engine"])
        self.assertEqual("license_not_found", candidate["license"])
        self.assertEqual("metadata_only", candidate["redistribution_policy"])
        self.assertEqual({"Cr", "Fe", "Co", "Ni", "Pd"}, set(candidate["elements"]))
        self.assertIn("Round-39", candidate["rejection_reason"])
        self.assertIn("no PointFF-enabled LAMMPS", candidate["rejection_reason"])
        self.assertEqual(
            "97f43b8036335fc3cc52888500b988738377424f162c699df7a4a78a5be98ff8",
            candidate["artifact_hashes"]["jit_epoch.pth"]["sha256"],
        )

        source = audit["source_evidence"]
        self.assertIsNone(source["repository_license"])
        self.assertFalse(source["repository_readme_found"])
        self.assertEqual(0, source["repository_release_count"])

        torch_load = audit["torch_jit_load"]
        self.assertEqual("pass", torch_load["status"])
        self.assertIn("Tensor x", torch_load["forward_schema"])

        lammps_input = audit["lammps_input_evidence"]["in.MC"]
        self.assertEqual("PointFF 6.0 45", lammps_input["pair_style"])
        self.assertEqual("jit_epoch_37.pth", lammps_input["pair_coeff_model"])
        self.assertEqual([1.66, 1.83, 1.88, 1.91, 2.2], lammps_input["pair_coeff_radii"])

    def test_round40_records_gamma_prime_checkpoint_misses_and_request_only_nicoal(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round40_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(60, summary["candidate_count_before"])
        self.assertEqual(60, summary["candidate_count_after"])

        nicoal = audit["primary_source_rechecks"]["ni_co_al_nep_qin_2025"]
        self.assertEqual("Data will be made available on request.", nicoal["science_direct_data_availability"])
        self.assertTrue(all(row["http_status"] == 404 for row in nicoal["common_elsevier_supplement_paths_checked"]))
        self.assertEqual({}, nicoal["crossref_relation"])

        nial = audit["primary_source_rechecks"]["ni_al_mtp_wang_2025"]
        self.assertIsNone(nial["repository_license"])
        self.assertFalse(nial["license_file_present"])
        self.assertTrue(nial["model_file_present"])
        self.assertIn("potential.mtp", nial["tree_paths"])

        nep_data = audit["public_index_queries"]["gitlab_nep_data_tree_probe"]
        self.assertEqual(773, nep_data["total_tree_rows_checked"])
        self.assertEqual(0, nep_data["target_pattern_hit_count"])

        ti_al_nb = audit["public_index_queries"]["zenodo_supporting_non_target_hit"]
        self.assertEqual("supporting_non_target_ti_al_nb_lead", ti_al_nb["classification"])
        self.assertIn("lacks Ni", ti_al_nb["not_added_to_registry_reason"])

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertIn("Round-40", candidates[candidate_id]["rejection_reason"])
                self.assertNotEqual("accepted", candidates[candidate_id]["acceptance_status"])
        self.assertIn("request", candidates["pot_nicoal_nep_qin_2025_lead"]["rejection_reason"])
        self.assertIn("no LICENSE", candidates["pot_nial_mtp_wang_2025_candidate"]["rejection_reason"])

    def test_round41_mtp_hdnnp_runtime_probe_keeps_specialized_candidates_blocked(self):
        audit = json.loads(
            Path("audits/mlip_mtp_hdnnp_runtime_resolution_round41_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertTrue(summary["downloaded_mtp_model_files_present"])
        self.assertTrue(summary["downloaded_nial_hdnnp_model_files_present"])
        self.assertFalse(summary["mtp_runtime_available"])
        self.assertFalse(summary["n2p2_runtime_available"])
        self.assertTrue(summary["python_lammps_module_present"])
        self.assertFalse(summary["python_lammps_load_ok"])

        self.assertFalse(audit["environment_probes"]["path_executables"]["mlp"])
        self.assertFalse(audit["environment_probes"]["path_executables"]["lammps"])
        self.assertTrue(audit["environment_probes"]["python_modules"]["lammps"])
        self.assertFalse(audit["environment_probes"]["lammps_python"]["load_ok"])

        maml = audit["pip_probes"]["maml_2025_4_3"]
        self.assertEqual("would_install", maml["dry_run_result"])
        self.assertTrue(maml["contains_mtp_wrapper"])
        self.assertTrue(maml["requires_external_mlp_executable"])
        self.assertTrue(maml["requires_external_lammps_executable"])

        self.assertFalse(audit["pip_probes"]["mlip_0_2_2"]["is_shapeev_mtp_runtime"])
        self.assertEqual("No matching distribution found", audit["pip_probes"]["n2p2"]["pip_index_result"])
        self.assertEqual("No matching distribution found", audit["pip_probes"]["pynnp"]["pip_index_result"])

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                if candidate_id == "pot_ni3al_mtp_darus_yield_anomaly_2024":
                    self.assertEqual("accepted", candidates[candidate_id]["acceptance_status"])
                    self.assertIn("round-51", candidates[candidate_id]["rejection_reason"].lower())
                else:
                    self.assertEqual("blocked", candidates[candidate_id]["acceptance_status"])
                    self.assertIn("Round-41", candidates[candidate_id]["rejection_reason"])

    def test_round42_records_gamma_prime_checkpoint_misses_without_acceptance(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round42_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(60, summary["candidate_count_before"])
        self.assertEqual(60, summary["candidate_count_after"])

        github_counts = [
            row["total_count"]
            for row in audit["github_repository_search"]["queries"]
            if row["status"] == 200
        ]
        self.assertTrue(github_counts)
        self.assertTrue(all(count == 0 for count in github_counts))

        hf_counts = [
            row["count"]
            for row in audit["huggingface_model_search"]["queries"]
            if row["status"] == 200
        ]
        self.assertTrue(hf_counts)
        self.assertTrue(all(count == 0 for count in hf_counts))

        self.assertEqual("not_usable_without_authentication", audit["github_code_search"]["result"])
        self.assertEqual(401, audit["github_code_search"]["http_status"])
        self.assertEqual("no_target_checkpoint", audit["zenodo_search"]["classification"])

        nialta = next(row for row in audit["datacite_search"]["queries"] if row["query"] == "NiAlTa")
        self.assertEqual("non_mlip_experimental_or_materials_data", nialta["classification"])

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertEqual("research_needed", candidates[candidate_id]["acceptance_status"])
                self.assertIn("Round-42", candidates[candidate_id]["rejection_reason"])

    def test_round43_author_repo_probe_finds_no_admittable_gamma_prime_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_author_repo_probe_round43_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertTrue(summary["new_author_profile_found"])
        self.assertTrue(summary["new_non_target_repository_found"])

        mou_repos = {
            row["full_name"]: row
            for row in audit["mou_sakidja_author_evidence"]["github_repositories_checked"]
        }
        self.assertEqual(
            "non_target_high_entropy_carbide_workflow",
            mou_repos["mariummou/HEC_GB_MCMD_MACE_scripts"]["classification"],
        )
        self.assertEqual("MIT", mou_repos["mariummou/HEC_GB_MCMD_MACE_scripts"]["license"])

        lam_dpa = next(
            row
            for row in audit["known_repository_rechecks"]
            if row["repo"] == "JiangXiaoMingSan/LAM-DPA-Train-Script"
        )
        self.assertEqual(0, lam_dpa["release_count"])
        self.assertEqual("training_config_only_no_fcc_l12_checkpoint", lam_dpa["classification"])
        self.assertIn("complex-structures/dpa3-FCC+L12/input_dpa3ft.json", lam_dpa["fcc_l12_paths"])

        light_pfp = next(row for row in audit["known_repository_rechecks"] if row["repo"] == "pfnet-research/light-pfp")
        self.assertEqual("datasets_configs_results_no_standalone_checkpoint", light_pfp["classification"])

        self.assertTrue(
            all(row["result_count"] == 0 for row in audit["github_plugin_file_search"]["queries"])
        )

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertEqual("research_needed", candidates[candidate_id]["acceptance_status"])
                self.assertIn("Round-43", candidates[candidate_id]["rejection_reason"])

    def test_round44_strong_gamma_prime_leads_are_not_admitted_without_model_files(self):
        audit = json.loads(
            Path("audits/mlip_nialti_nicoal_strong_lead_followup_round44_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(60, summary["candidate_count_before"])
        self.assertEqual(60, summary["candidate_count_after"])
        self.assertEqual(audit["formal_manifest_counts_before"], audit["formal_manifest_counts_after"])

        nialti = audit["primary_source_rechecks"]["nialti_mace_fitsnap_tms2026"]
        self.assertEqual(
            "conference_abstract_confirms_model_training_but_no_checkpoint",
            nialti["classification"],
        )
        self.assertEqual(["MACE", "FitSNAP neural-network architecture"], nialti["observed_model_families"])
        self.assertFalse(nialti["model_download_url_found"])
        self.assertFalse(nialti["license_found"])

        nicoal = audit["primary_source_rechecks"]["nicoal_nep_matdes2025"]
        self.assertEqual("paper_confirms_target_nep_but_request_only_data", nicoal["classification"])
        self.assertEqual("Data will be made available on request.", nicoal["data_availability"])
        self.assertFalse(nicoal["model_download_url_found"])

        gpumd = audit["primary_source_rechecks"]["gpumd_publications"]
        self.assertTrue(gpumd["contains_nicoal_article"])
        self.assertFalse(gpumd["contains_target_model_download"])

        github_counts = [row["total_count"] for row in audit["repository_and_model_hub_search"]["github_repository_api"]["queries"]]
        hf_counts = [row["count"] for row in audit["repository_and_model_hub_search"]["huggingface_model_api"]["queries"]]
        self.assertTrue(github_counts)
        self.assertTrue(hf_counts)
        self.assertTrue(all(count == 0 for count in github_counts))
        self.assertTrue(all(count == 0 for count in hf_counts))

        datacite_totals = [row["total"] for row in audit["repository_and_model_hub_search"]["datacite_api"]["queries"]]
        self.assertTrue(all(total == 0 for total in datacite_totals))
        self.assertEqual(
            "published_nep_repository_no_target_model",
            audit["repository_and_model_hub_search"]["gitlab_nep_data_tree"]["classification"],
        )
        self.assertEqual([], audit["repository_and_model_hub_search"]["gitlab_nep_data_tree"]["target_path_matches"])
        self.assertEqual(
            401,
            audit["repository_and_model_hub_search"]["gitlab_global_search_api"]["http_status"],
        )

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            with self.subTest(candidate_id=candidate_id):
                self.assertEqual("research_needed", candidates[candidate_id]["acceptance_status"])
                self.assertIn("Round-44", candidates[candidate_id]["rejection_reason"])

    def test_round45_accepted_nep_potentials_have_formal_relaxed_manifests(self):
        audit = json.loads(
            Path("audits/mlip_accepted_nep_formal_relaxation_round45_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertEqual(20, summary["new_formal_relaxations"])
        self.assertEqual(20, summary["new_formal_relaxed_structures"])
        self.assertEqual(0, summary["new_xrd_samples"])
        self.assertTrue(summary["all_new_relaxations_converged"])
        self.assertTrue(summary["all_new_relaxed_structures_quality_pass"])
        self.assertEqual(0, summary["validation_issue_count_after"])
        self.assertEqual(0, summary["quality_issue_count_after"])

        expected_counts = {
            "pot_crconi_nep_wu_2026": {
                "relaxations": 5,
                "systems": {"CrCoNi"},
            },
            "pot_rhea9_nep_byggmastar_2026": {
                "relaxations": 15,
                "systems": {"MoNbTaVW", "TiTaVW", "WTaCrV"},
            },
        }
        for potential_id, expected in expected_counts.items():
            with self.subTest(potential_id=potential_id):
                manifest_dir = Path("data") / "potentials" / potential_id / "generated" / "manifests"
                relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
                relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")
                systems = {
                    str(value)
                    for value in relaxations["initial_structure_id"].str.extract(r"init_([^_]+)_")[0].dropna()
                }
                self.assertEqual(expected["relaxations"], len(relaxations))
                self.assertEqual(expected["relaxations"], len(relaxed))
                self.assertEqual(expected["relaxations"], int(relaxations["converged"].sum()))
                self.assertEqual({"pass"}, set(relaxed["quality_tier"]))
                self.assertEqual(expected["systems"], systems)

        self.assertEqual(
            143,
            audit["workspace_manifest_counts"]["after"]["relaxations"],
        )
        self.assertEqual(
            143,
            audit["workspace_manifest_counts"]["after"]["relaxed_structures"],
        )
        self.assertFalse(audit["runtime_probe"]["python_lammps_load"]["ok"])

    def test_round10_adds_cocrfeni_cocrni_mtp_candidate_without_accepting_it(self):
        audit = json.loads(Path("audits/mlip_specialized_search_round10_2026-07-04.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_cocrfeni_cocrni_mtp_nitol_2026"
        ]
        files = {Path(row["path"]).name: row for row in audit["new_downloaded_candidate"]["downloaded_files"]}

        self.assertTrue(audit["summary"]["new_downloaded_target_specific_mlip_candidate_found"])
        self.assertFalse(audit["summary"]["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual("target_specific_mlip", candidate["target_role"])
        self.assertEqual("mtp_lammps", candidate["engine"])
        self.assertEqual({"Co", "Cr", "Fe", "Ni"}, set(candidate["elements"]))
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertEqual("metadata_only", candidate["redistribution_policy"])
        self.assertFalse(audit["new_downloaded_candidate"]["repository_tree_evidence"]["license_file_found"])
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertEqual(
            "b1bf52e51e4f9125cbda5780857ee9be46356f64768b60e2b5bf482d3fccda70",
            files["cocrfeni.mtp"]["sha256"],
        )
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_unresolved_candidates"])

    def test_round11_separates_existing_downloads_from_new_gamma_prime_models(self):
        audit = json.loads(Path("audits/mlip_specialized_search_round11_2026-07-04.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_downloaded_target_specific_model_found"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertTrue(summary["nial_mtp_existing_candidate_confirmed_not_missing"])
        self.assertFalse(summary["nial_mtp_github_license_file_found"])
        self.assertTrue(summary["aissquare_public_article_confirms_dataset_and_model_release"])

        nial = candidates["pot_nial_mtp_wang_2025_candidate"]
        self.assertEqual("unknown", nial["license"])
        self.assertEqual(
            "4697dfcebbae24f99a2ef66d0dcf2a8577fd8aa6143d7313ad8bc95d5d0996f1",
            nial["artifact_hashes"]["potential.mtp"]["sha256"],
        )
        cocrfeni = candidates["pot_cocrfeni_cocrni_mtp_nitol_2026"]
        self.assertEqual(
            "b1bf52e51e4f9125cbda5780857ee9be46356f64768b60e2b5bf482d3fccda70",
            cocrfeni["artifact_hashes"]["cocrfeni.mtp"]["sha256"],
        )

        nonadmissible = {
            row["doi"]: row["classification"]
            for row in audit["related_nonadmissible_public_data"]
            if "doi" in row
        }
        self.assertEqual("atomic_structure_dump_dataset_not_mlip_model", nonadmissible["10.5281/zenodo.18427033"])
        self.assertEqual(
            "open_access_crconi_simulation_article_no_model_artifact",
            nonadmissible["10.1039/D5MA01373A"],
        )
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_unresolved_candidates"])

    def test_round12_author_repository_followup_keeps_mou_lead_unadmitted(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_author_repository_followup_round12_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nibased_gamma_prime_mou_2023_lead"
        ]

        summary = audit["summary"]
        self.assertTrue(summary["thesis_and_aps_sources_confirm_dedicated_mlip_lead"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["author_github_model_file_found"])
        self.assertEqual(0, summary["huggingface_model_hits"])
        self.assertEqual(0, summary["huggingface_dataset_hits"])
        self.assertEqual(0, summary["zenodo_exact_author_hits"])
        self.assertEqual(403, summary["bearworks_pdf_probe_status"])
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertFalse(audit["registry_effect"]["candidate_registry_status_changed"])

        repos = {row["repo"]: row for row in audit["github_api"]["author_repositories_scanned"]}
        self.assertEqual([], repos["mariummou/HEC_GB_MCMD_MACE_scripts"]["model_like_paths"])
        self.assertIn("maml/apps/pes/tests/MTP/fitted.mtp", repos["mariummou/maml"]["representative_model_like_paths"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual("thesis_abstract_no_model_file", candidate["artifact_kind"])
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_candidates"])

    def test_round13_extracts_ni3al_spin_mtp_models_but_keeps_candidate_blocked(self):
        audit = json.loads(
            Path("audits/mlip_ni3al_mtp_darus_spin_fluctuations_2023_round13_audit.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_ni3al_mtp_darus_spin_fluctuations_2023"
        ]

        self.assertTrue(audit["classification"]["model_files_found"])
        self.assertFalse(audit["classification"]["admittable_now"])
        self.assertEqual("archive_extracted_model_files", candidate["artifact_kind"])
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertFalse(audit["registry_effect"]["generated_manifest_rows_created"])

        hashes = candidate["artifact_hashes"]
        self.assertEqual(
            "f3b3be363b8cb941e89c1e154048774d09d2817135ff40a198c4d0a64c3967aa",
            hashes["MTP-training-set/mtp-training/24.mtp"]["sha256"],
        )
        self.assertEqual(741332, hashes["MTP-training-set/mtp-training/24.mtp"]["bytes"])
        self.assertEqual(
            "e5ec87dd5290317989fa5ed2d67a7319adce35b8db4c7f2306ff843b0d776a87",
            hashes["MTP-training-set/mtp-training/Trained.mtp_"]["sha256"],
        )
        self.assertEqual(766402, hashes["MTP-training-set/mtp-training/Trained.mtp_"]["bytes"])
        members = {row["archive_member"] for row in audit["extracted_model_files"]}
        self.assertIn("MTP-training-set/mtp-training/24.mtp", members)
        self.assertIn("MTP-training-set/mtp-training/Trained.mtp_", members)

    def test_round14_public_checkpoint_search_records_no_new_gamma_prime_model(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round14_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertEqual(0, summary["huggingface_model_hits"])
        self.assertEqual(0, summary["huggingface_dataset_hits"])
        self.assertEqual(0, summary["datacite_exact_file_pattern_hits"])
        self.assertEqual(0, summary["aissquare_target_specific_hits"])
        self.assertFalse(summary["lam_dpa3_fcc_l12_checkpoint_found"])

        self.assertEqual(131, audit["aissquare_public_model_list"]["items_scanned"])
        aissquare_names = {row["name"] for row in audit["aissquare_public_model_list"]["matched_items"]}
        self.assertEqual({"DPA3-Omol-Large", "DPA3-SPICE-MACE-OFF"}, aissquare_names)
        self.assertEqual(0, audit["aissquare_public_model_list"]["target_specific_hits"])

        lam = audit["lam_dpa_repository_followup"]
        self.assertEqual(51, lam["blob_count"])
        self.assertEqual(["complex-structures/dpa3-FCC+L12/input_dpa3ft.json"], lam["dpa3_fcc_l12_paths"])
        self.assertFalse(lam["dpa3_fcc_l12_checkpoint_found"])
        self.assertEqual(
            "magnesium_battery_dataset_not_Ni_Al_Ti_Ta_Co_gamma_prime_mlip",
            audit["materials_cloud_api"]["false_positive_record"]["classification"],
        )
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_candidates"])

    def test_round15_finds_exact_monbtavw_mtp_without_accepting_it(self):
        audit = json.loads(Path("audits/mlip_specialized_search_round15_2026-07-04.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_downloaded_target_specific_mlip_candidates_found"])
        self.assertTrue(summary["new_exact_benchmark_system_model_found"])
        self.assertIn("pot_monbtavw_mtp_wang_2024", summary["new_exact_benchmark_system_model_ids"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["generated_manifest_rows_created"])

        monbtavw = candidates["pot_monbtavw_mtp_wang_2024"]
        self.assertEqual("target_specific_mlip", monbtavw["target_role"])
        self.assertEqual("mtp_lammps", monbtavw["engine"])
        self.assertEqual({"Mo", "Nb", "Ta", "V", "W"}, set(monbtavw["elements"]))
        self.assertEqual("blocked", monbtavw["acceptance_status"])
        self.assertEqual("unknown", monbtavw["license"])
        self.assertEqual("metadata_only", monbtavw["redistribution_policy"])
        self.assertEqual(
            "80075e111fa2d6676adcd57842dc014e556518a8efba319ea3cc4e573f822083",
            monbtavw["artifact_hashes"]["potential/fitted.mtp"]["sha256"],
        )

        self.assertEqual(
            "official_nep_data_repository_contains_many_nep_txt_models_but_no_CrCoNi_CoCrFeMnNi_MoNbTaVW_NiAlTi_NiAlTa_or_NiAlTiTa_target_model",
            audit["negative_or_non_target_checks"]["gpumd_nep_data"]["classification"],
        )
        self.assertIn("pot_tial_mtp_mvl_2023", audit["registry_effect"]["added_candidate_ids"])

    def test_round16_downloads_lam_dpa_dp_and_finetune_without_accepting_it(self):
        audit = json.loads(Path("audits/mlip_specialized_search_round16_2026-07-04.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_downloaded_target_specific_mlip_candidates_found"])
        self.assertTrue(summary["new_exact_benchmark_system_model_found"])
        self.assertEqual(5, summary["new_downloaded_checkpoint_file_count"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["generated_manifest_rows_created"])

        dp_5e = candidates["pot_monbtavw_dp_lam_jiang_2026"]
        self.assertEqual("DeepMD/DP", dp_5e["family"])
        self.assertEqual("deepmd_lammps", dp_5e["engine"])
        self.assertEqual({"Mo", "Nb", "Ta", "V", "W"}, set(dp_5e["elements"]))
        self.assertEqual("blocked", dp_5e["acceptance_status"])
        self.assertEqual(
            "56fc69e942ab80f975e04bec3e1ac88838161840aad159177531ef392e1cb321",
            dp_5e["artifact_hashes"]["model.ckpt.pt"]["sha256"],
        )

        dp_all = candidates["pot_monbtavw_dp_lam_all_jiang_2026"]
        self.assertEqual(
            "1470d1304651904a9662bd60bdda0f70c20a1ee1f66b72414a52561339d80566",
            dp_all["artifact_hashes"]["model.ckpt.pt"]["sha256"],
        )

        dpa2_5e = candidates["pot_monbtavw_dpa2_lam_jiang_2026"]
        self.assertEqual(
            "5ebe9f3a930e6925c55202954b71a1721589ffd3b6065e30d3af34e29e3ad737",
            dpa2_5e["artifact_hashes"]["finetune/model.ckpt.pt"]["sha256"],
        )
        dpa2_all = candidates["pot_monbtavw_dpa2_lam_all_jiang_2026"]
        self.assertEqual(
            "977faa90546466ed1a02b4850e801f5239eecead98d4a99ace135188903b6bec",
            dpa2_all["artifact_hashes"]["finetune/model.ckpt-800000.pt"]["sha256"],
        )
        self.assertFalse(
            audit["primary_source_evidence"]["lam_dpa3_fcc_l12_followup"]["dpa3_fcc_l12_checkpoint_found"]
        )
        self.assertIn("pot_monbtavw_dp_lam_jiang_2026", audit["registry_effect"]["added_candidate_ids"])

    def test_round17_records_gamma_prime_checkpoint_misses_without_registry_changes(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round17_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertEqual(0, summary["new_downloaded_model_files"])
        self.assertEqual(0, summary["repository_search_target_hits"])
        self.assertEqual(0, summary["huggingface_target_hits"])
        self.assertEqual(0, summary["datacite_exact_target_file_hits"])
        self.assertFalse(summary["github_code_search_usable_without_auth"])
        self.assertEqual(0, summary["aissquare_target_specific_hits"])
        self.assertFalse(summary["generated_manifest_rows_created"])

        self.assertFalse(audit["registry_effect"]["candidate_registry_status_changed"])
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertIn(
            "not used as proof of absence",
            audit["github_code_search"]["conclusion"],
        )
        self.assertIn(
            "Ni-Al-Ti gamma/gamma-prime MLIP checkpoint",
            audit["targets"],
        )
        self.assertIn(
            "usnistgov/HTE-NiTiAl",
            {lead["id"] for lead in audit["non_target_or_blocked_leads"]},
        )

    def test_round18_adds_nicrfealp_supplement_lead_without_runnable_model(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_author_supplement_followup_round18_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_downloaded_source_document_found"])
        self.assertTrue(summary["new_candidate_added"])
        self.assertEqual("pot_nicrfealp_mtp_ning_2026_lead", summary["new_candidate_id"])
        self.assertFalse(summary["new_runnable_model_file_found"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["generated_manifest_rows_created"])

        candidate = candidates["pot_nicrfealp_mtp_ning_2026_lead"]
        self.assertEqual("MTP", candidate["family"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual({"Ni", "Cr", "Fe", "Al", "P"}, set(candidate["elements"]))
        self.assertEqual(
            "3bafe2d9831a0dad21656b55d53721ffaaa08bfbf6138b3fff012f833e142bec",
            candidate["artifact_hashes"]["S0264127526010841-mmc1.pdf"]["sha256"],
        )
        self.assertTrue(audit["supplement_text_evidence"]["Ni54Cr20Fe20Al5P1"])
        self.assertTrue(audit["supplement_text_evidence"]["Ni3Al"])
        self.assertFalse(audit["supplement_text_evidence"][".mtp"])
        self.assertIn("pot_nicrfealp_mtp_ning_2026_lead", audit["registry_effect"]["added_candidate_ids"])

    def test_round19_completes_ni3al_yield_training_files_before_later_acceptance(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_checkpoint_and_ni3al_completion_round19_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_ni3al_mtp_darus_yield_anomaly_2024"
        ]
        dataverse_files = {row["filename"]: row for row in candidate["dataverse_files"]}

        self.assertFalse(audit["summary"]["new_runnable_gamma_prime_checkpoint_found"])
        self.assertEqual(
            "pot_ni3al_mtp_darus_yield_anomaly_2024",
            audit["summary"]["existing_candidate_completed_with_training_files"],
        )
        self.assertEqual(5, audit["summary"]["new_downloaded_file_count"])
        self.assertEqual("accepted", candidate["acceptance_status"])
        self.assertIn("round-51", candidate["rejection_reason"].lower())
        self.assertEqual(
            "23ba91ad36196caf937d6f7dc008c40ee7b45601806e7677b5a4ae2cf0d5e864",
            dataverse_files["train.cfg"]["sha256"],
        )
        self.assertIn("CSFE.dat", dataverse_files)
        self.assertFalse(audit["registry_effect"]["formal_generated_manifests_modified"])

    def test_round20_adds_nequip_allegro_foundation_baselines_only(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_and_nequip_allegro_followup_round20_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        self.assertEqual([], audit["api_query_summary"]["target_checkpoint_hits"])
        self.assertEqual(2, len(audit["new_downloads"]))
        self.assertEqual("general_baseline", candidates["pot_nequip_oam_l_0p1"]["target_role"])
        self.assertEqual("general_baseline", candidates["pot_allegro_oam_l_0p1"]["target_role"])
        self.assertIn("not a target-trained HEA or gamma/gamma-prime MLIP", candidates["pot_nequip_oam_l_0p1"]["rejection_reason"])
        self.assertIn("not a target-trained HEA or gamma/gamma-prime MLIP", candidates["pot_allegro_oam_l_0p1"]["rejection_reason"])

    def test_round21_records_primary_source_followup_without_new_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round21_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual([], audit["target_checkpoint_hits"])
        self.assertEqual([], audit["new_downloaded_model_files"])
        self.assertFalse(audit["formal_generated_manifests_modified"])
        source_candidates = {row["candidate_id"] for row in audit["primary_sources_checked"]}
        self.assertIn("pot_nialti_mace_fitsnap_song_2026_lead", source_candidates)
        self.assertIn("pot_nicoal_nep_qin_2025_lead", source_candidates)
        effects = {row["candidate_id"]: row["effect"] for row in audit["existing_candidate_effect"]}
        self.assertIn("pot_nialtita_mlip_candidate", effects)
        self.assertIn("pot_nialti_mace_fitsnap_song_2026_lead", effects)

    def test_round22_records_public_index_misses_without_new_downloads(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round22_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(0, summary["github_repository_total_hits"])
        self.assertEqual(0, summary["huggingface_model_hits"])
        self.assertEqual(0, summary["huggingface_dataset_hits"])
        self.assertEqual(0, summary["datacite_target_hits"])
        self.assertEqual(0, summary["aissquare_target_hits"])
        self.assertEqual(0, summary["zenodo_target_checkpoint_hits"])
        self.assertEqual([], audit["api_checks"]["github_repositories"]["target_repository_hits"])
        self.assertEqual([], audit["api_checks"]["huggingface_models"]["target_checkpoint_hits"])
        nonmatching_titles = {row["title"] for row in audit["api_checks"]["zenodo"]["notable_nonmatching_hits"]}
        self.assertIn("NequIP & Allegro Foundation Potentials", nonmatching_titles)
        effects = {row["candidate_id"]: row["effect"] for row in audit["existing_candidate_effect"]}
        self.assertIn("pot_nialti_mlip_candidate", effects)
        self.assertIn("pot_nialta_mlip_candidate", effects)
        self.assertIn("pot_nialtita_mlip_candidate", effects)

    def test_round23_adds_nist_meam_fallbacks_without_satisfying_mlip_gap(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_nist_meam_fallback_round23_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertEqual(4, summary["new_downloaded_file_count"])
        self.assertEqual(2, summary["new_candidate_count"])
        self.assertFalse(summary["new_mlip_checkpoint_found"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])

        nialti = candidates["pot_nialti_meam_sharifi_2025_classical_fallback"]
        nicoal = candidates["pot_nicoal_meam_sharifi_2025_classical_fallback"]
        self.assertEqual("classical_fallback", nialti["target_role"])
        self.assertEqual("classical_fallback", nicoal["target_role"])
        self.assertEqual("classical_potential_file", nialti["artifact_kind"])
        self.assertEqual("classical_potential_file", nicoal["artifact_kind"])
        self.assertIn("not an MLIP", nialti["rejection_reason"])
        self.assertIn("not an MLIP", nicoal["rejection_reason"])
        self.assertEqual(
            "702cf6d6a469698acb36600e888f6e40d82e140c0f86c93a764928a84df150ff",
            nialti["artifact_hashes"]["NiTiAl.meam"]["sha256"],
        )
        self.assertEqual(
            "29cc4822dfdaa0c0426333f81d1936174921f27de337fac3c0aab38e69ffce01",
            nicoal["artifact_hashes"]["NiCoAl.meam"]["sha256"],
        )

    def test_round24_adds_nialta_eam_publication_lead_without_model_file(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round24_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertEqual(["pot_nialta_eam_fan_2016_classical_lead"], summary["new_candidate_ids"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(0, summary["openkim_target_model_hits"])
        self.assertEqual(0, summary["github_repository_target_hits"])
        self.assertEqual(0, summary["huggingface_target_hits"])
        self.assertEqual(0, summary["zenodo_target_checkpoint_hits"])

        candidate = candidates["pot_nialta_eam_fan_2016_classical_lead"]
        self.assertEqual("EAM", candidate["family"])
        self.assertEqual("classical_fallback", candidate["target_role"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual("publication_only_no_public_file", candidate["artifact_kind"])
        self.assertIn("rather than MLIP", candidate["rejection_reason"])
        self.assertIn(
            "pot_nialta_mlip_candidate",
            {row["candidate_id"] for row in audit["existing_candidate_effect"]},
        )

    def test_round25_records_public_checkpoint_misses_without_registry_or_manifest_changes(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round25_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertEqual([], summary["new_candidate_ids"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(0, summary["github_repository_target_hits"])
        self.assertEqual(0, summary["huggingface_model_hits"])
        self.assertEqual(0, summary["huggingface_dataset_hits"])
        self.assertEqual(0, summary["datacite_target_hits"])
        self.assertEqual(0, summary["zenodo_target_checkpoint_hits"])

        github_queries = {row["query"]: row["total_count"] for row in audit["api_checks"]["github_repositories"]}
        self.assertEqual(0, github_queries['"Ni-Co-Al" "nep.txt"'])
        self.assertEqual(0, github_queries['"Ni-Al-Ti" "MACE" "FitSNAP"'])
        self.assertEqual(0, github_queries['"Ni-Al-Ti-Ta" "model.pt"'])

        nicoal_primary_effects = [
            row["effect"]
            for row in audit["primary_source_checks"]
            if row["candidate_id"] == "pot_nicoal_nep_qin_2025_lead"
        ]
        self.assertTrue(nicoal_primary_effects)
        self.assertTrue(any("no public nep.txt/model archive" in effect for effect in nicoal_primary_effects))
        remaining = {row["candidate_id"] for row in audit["existing_candidate_effect"]}
        self.assertIn("pot_nialti_mlip_candidate", remaining)
        self.assertIn("pot_nialta_mlip_candidate", remaining)
        self.assertIn("pot_nialtita_mlip_candidate", remaining)

    def test_round26_deep_probes_primary_sources_without_model_downloads(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_primary_source_deep_probe_round26_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertEqual([], summary["new_candidate_ids"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(0, summary["elsevier_mmc_supplement_hits"])
        self.assertEqual(0, summary["crossref_supplement_relations"])
        self.assertEqual(4, summary["aissquare_authenticated_detail_blockers"])
        self.assertEqual(0, summary["aissquare_public_object_guess_hits"])
        self.assertEqual(0, summary["programmaster_model_links_found"])

        nicoal = audit["ni_co_al_nep_primary_article"]
        self.assertEqual("pot_nicoal_nep_qin_2025_lead", nicoal["candidate_id"])
        self.assertEqual({}, nicoal["crossref_check"]["relation"])
        self.assertEqual(0, nicoal["elsevier_mmc_probe"]["hit_count"])
        self.assertIn("no public nep.txt/model archive", nicoal["effect"])

        nialti = audit["ni_al_ti_programmaster_lead"]
        self.assertEqual("pot_nialti_mace_fitsnap_song_2026_lead", nialti["candidate_id"])
        self.assertEqual("Huajing Song", nialti["author"])
        self.assertEqual("Pratt and Whitney", nialti["affiliation_evidence"]["observed_affiliation"])
        self.assertIn("does not expose", nialti["programmaster_finding"])

        aissquare = audit["aissquare_alloy53_recheck"]
        self.assertEqual("metal_domain_baseline_not_gamma_prime_specific", aissquare["classification"])
        self.assertEqual(64, aissquare["public_object_guess_probe"]["checked_count"])
        self.assertEqual(0, aissquare["public_object_guess_probe"]["hit_count"])
        self.assertEqual(
            {401},
            {row["http_status"] for row in aissquare["anonymous_detail_access"]},
        )

        blockers = {row["candidate_id"] for row in audit["existing_candidate_effect"]}
        self.assertIn("pot_nialti_mlip_candidate", blockers)
        self.assertIn("pot_nialta_mlip_candidate", blockers)
        self.assertIn("pot_nialtita_mlip_candidate", blockers)

    def test_round27_downloads_unimi_hea_nep_without_closing_target_gaps(self):
        audit = json.loads(
            Path("audits/mlip_hea_nep_unimi_and_target_gap_round27_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_downloaded_hea_related_mlip_candidate_found"])
        self.assertEqual(["pot_mncofenicu_nep_unimi_2026"], summary["new_candidate_ids"])
        self.assertEqual(3, summary["new_downloaded_model_archive_count"])
        self.assertEqual(3, summary["new_extracted_nep_txt_count"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["downloadable_exact_target_crconi_or_cocrfemnni_nep_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(0, summary["github_exact_target_repository_hits"])
        self.assertEqual(0, summary["huggingface_exact_target_model_hits"])
        self.assertEqual(0, summary["zenodo_strict_exact_target_hits"])
        self.assertEqual(0, summary["datacite_exact_target_hits"])

        candidate = candidates["pot_mncofenicu_nep_unimi_2026"]
        self.assertEqual("hea_related_mlip", candidate["target_role"])
        self.assertEqual("NEP", candidate["family"])
        self.assertEqual("gpumd", candidate["engine"])
        self.assertEqual("CC0-1.0", candidate["license"])
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertEqual({"Mn", "Co", "Fe", "Ni", "Cu"}, set(candidate["elements"]))
        self.assertIn("cannot replace", candidate["rejection_reason"])

        downloaded = {Path(row["path"]).name: row for row in audit["downloaded_files"]}
        self.assertEqual(4, len(downloaded))
        self.assertTrue(all(row["md5_matches_dataverse"] for row in downloaded.values()))
        self.assertEqual(
            "e2285129195e987ef5894e07a4115db0f2022524755616479696f882376bef99",
            downloaded["NEP_Jan08_maxF10_80percent.tar.bz2"]["sha256"],
        )

        extracted = {Path(row["path"]).parent.name: row for row in audit["extracted_key_files"]}
        self.assertEqual(3, len(extracted))
        self.assertEqual(
            "705044793225a8c920c95c33ea4f87e7b6a4dff108f42b454eb61ea0d00e4662",
            extracted["NEP_Jan08_maxF10_alltrain_alt"]["sha256"],
        )
        self.assertEqual(
            ["Mn", "Fe", "Co", "Ni", "Cu"],
            audit["new_candidate"]["element_coverage_from_nep_in"],
        )
        self.assertFalse(audit["local_engine_probe"]["gpumd"])
        self.assertFalse(audit["local_engine_probe"]["lammps"])
        self.assertIn("smoke relaxation remain blocked", audit["local_engine_probe"]["effect"])

        nialti_queries = {
            row["query"]: row["total_count"]
            for row in audit["public_index_rechecks"]["github_repositories"]
        }
        self.assertEqual(0, nialti_queries['"Ni-Al-Ti" "potential.mtp"'])
        self.assertEqual(0, nialti_queries['"Ni-Co-Al" "nep.txt"'])

    def test_round28_lightpfp_repo_probe_keeps_repository_out_of_accepted_potentials(self):
        audit = json.loads(
            Path("audits/mlip_lightpfp_hea_repository_probe_round28_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["new_candidate_added"])
        self.assertEqual("pot_alcocrfeni_lightpfp_2025_lead", summary["new_candidate_id"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["standalone_trained_checkpoint_found"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])

        candidate = candidates["pot_alcocrfeni_lightpfp_2025_lead"]
        self.assertEqual("hea_specific_candidate", candidate["target_role"])
        self.assertEqual("MTP/LightPFP", candidate["family"])
        self.assertEqual("commercial_lightpfp", candidate["engine"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual("training_data_and_configs_no_standalone_model", candidate["artifact_kind"])
        self.assertEqual({"Al", "Co", "Cr", "Fe", "Ni"}, set(candidate["elements"]))
        self.assertIn("no standalone trained HEA model checkpoint", candidate["rejection_reason"])

        tree = audit["repository_tree_probe"]
        self.assertEqual(289, tree["total_blobs"])
        self.assertEqual(164, tree["hea_blobs"])
        self.assertEqual([], tree["exact_checkpoint_like_paths"])
        self.assertIn("examples/maintext/HEA/datasets/LightPFP/HEA_crystal.h5", tree["h5_dataset_paths"])
        self.assertIn("examples/maintext/pretrained-students/AlNi3.h5", tree["h5_dataset_paths"])
        self.assertIn("examples/maintext/HEA/model_training/light-pfp.json", tree["training_config_paths"])
        self.assertTrue(audit["readme_evidence"]["root_readme_says_examples_run_on_matlantis_product"])

    def test_downloaded_potential_files_inventory_records_cache_hashes(self):
        inventory = json.loads(
            Path("audits/downloaded_potential_files_inventory_2026-07-04.json").read_text(encoding="utf-8")
        )

        summary = inventory["summary"]
        self.assertEqual(0, summary["registry_hash_mismatch_count"])
        self.assertEqual(1, summary["incomplete_file_count"])
        self.assertIn("pot_crconi_mtp_csro", summary["local_candidate_cache_ids"])
        self.assertIn("pot_mace_mp_0b3_medium", summary["accepted_baseline_ids"])
        self.assertIn("pot_nialti_meam_sharifi_2025_classical_fallback", summary["local_candidate_cache_ids"])
        self.assertIn("pot_nicoal_meam_sharifi_2025_classical_fallback", summary["local_candidate_cache_ids"])

        files = inventory["files"]
        ni3al_mtp = next(
            row
            for row in files
            if row["potential_id"] == "pot_ni3al_mtp_darus_spin_fluctuations_2023"
            and row["path"].endswith("MTP-training-set/mtp-training/24.mtp")
        )
        self.assertTrue(ni3al_mtp["complete"])
        self.assertTrue(ni3al_mtp["matches_registry_hash"])
        self.assertEqual("f3b3be363b8cb941e89c1e154048774d09d2817135ff40a198c4d0a64c3967aa", ni3al_mtp["sha256"])

        probe = next(row for row in files if row["path"].endswith("mtp_training_probe.bin"))
        self.assertEqual("partial_archive_probe", probe["file_role"])
        self.assertFalse(probe["complete"])

        chgnet = next(
            row
            for row in files
            if row["potential_id"] == "pot_chgnet_0p3p0"
            and row["path"].endswith("chgnet_0.3.0_e29f68s314m37.pth.tar")
        )
        self.assertTrue(chgnet["matches_registry_hash"])
        self.assertEqual("accepted_baseline_package_cache", chgnet["scope"])

        monbtavw_tabgap = next(
            row
            for row in files
            if row["path"].endswith("Mo-Nb-Ta-V-W_v2_2b+3b+EAM.tabgap")
        )
        self.assertEqual(842948145, monbtavw_tabgap["bytes"])
        self.assertTrue(monbtavw_tabgap["matches_registry_hash"])

        monbtavw_mtp = next(
            row
            for row in files
            if row["potential_id"] == "pot_monbtavw_mtp_wang_2024"
            and row["path"].endswith("potential/fitted.mtp")
        )
        self.assertEqual(154572, monbtavw_mtp["bytes"])
        self.assertTrue(monbtavw_mtp["matches_registry_hash"])
        self.assertEqual(
            "80075e111fa2d6676adcd57842dc014e556518a8efba319ea3cc4e573f822083",
            monbtavw_mtp["sha256"],
        )

        nialti_sharifi = next(
            row
            for row in files
            if row["potential_id"] == "pot_nialti_meam_sharifi_2025_classical_fallback"
            and row["path"].endswith("NiTiAl.meam")
        )
        self.assertEqual("download_artifact", nialti_sharifi["file_role"])
        self.assertTrue(nialti_sharifi["matches_registry_hash"])
        self.assertEqual(
            "702cf6d6a469698acb36600e888f6e40d82e140c0f86c93a764928a84df150ff",
            nialti_sharifi["sha256"],
        )

        nicoal_sharifi = next(
            row
            for row in files
            if row["potential_id"] == "pot_nicoal_meam_sharifi_2025_classical_fallback"
            and row["path"].endswith("NiCoAl.meam")
        )
        self.assertEqual("download_artifact", nicoal_sharifi["file_role"])
        self.assertTrue(nicoal_sharifi["matches_registry_hash"])
        self.assertEqual(
            "29cc4822dfdaa0c0426333f81d1936174921f27de337fac3c0aab38e69ffce01",
            nicoal_sharifi["sha256"],
        )

        monbtavw_quinary = next(
            row
            for row in files
            if row["potential_id"] == "pot_monbtavw_mtp_wang_2024"
            and row["path"].endswith("potential/quinary.json")
        )
        self.assertEqual(78117426, monbtavw_quinary["bytes"])
        self.assertTrue(monbtavw_quinary["matches_registry_hash"])

        monbtavw_dp = next(
            row
            for row in files
            if row["potential_id"] == "pot_monbtavw_dp_lam_jiang_2026"
            and row["path"].endswith("downloads/model.ckpt.pt")
        )
        self.assertEqual(17265350, monbtavw_dp["bytes"])
        self.assertTrue(monbtavw_dp["matches_registry_hash"])

        monbtavw_dpa2_finetune = next(
            row
            for row in files
            if row["potential_id"] == "pot_monbtavw_dpa2_lam_jiang_2026"
            and row["path"].endswith("downloads/finetune/model.ckpt.pt")
        )
        self.assertEqual(24660538, monbtavw_dpa2_finetune["bytes"])
        self.assertTrue(monbtavw_dpa2_finetune["matches_registry_hash"])

        nicrfealp_pdf = next(
            row
            for row in files
            if row["potential_id"] == "pot_nicrfealp_mtp_ning_2026_lead"
            and row["path"].endswith("downloads/S0264127526010841-mmc1.pdf")
        )
        self.assertEqual(2895553, nicrfealp_pdf["bytes"])
        self.assertTrue(nicrfealp_pdf["matches_registry_hash"])

        ni3al_train = next(
            row
            for row in files
            if row["potential_id"] == "pot_ni3al_mtp_darus_yield_anomaly_2024"
            and row["path"].endswith("downloads/train.cfg")
        )
        self.assertEqual(14639583, ni3al_train["bytes"])
        self.assertEqual("training_set", ni3al_train["file_role"])
        self.assertTrue(ni3al_train["matches_registry_hash"])
        self.assertEqual(
            "23ba91ad36196caf937d6f7dc008c40ee7b45601806e7677b5a4ae2cf0d5e864",
            ni3al_train["sha256"],
        )

        nequip_oam = next(
            row
            for row in files
            if row["potential_id"] == "pot_nequip_oam_l_0p1"
            and row["path"].endswith("downloads/NequIP-OAM-L-0.1.nequip.zip")
        )
        self.assertEqual(78464590, nequip_oam["bytes"])
        self.assertTrue(nequip_oam["matches_registry_hash"])
        self.assertEqual("general_baseline", nequip_oam["target_role"])

        allegro_oam = next(
            row
            for row in files
            if row["potential_id"] == "pot_allegro_oam_l_0p1"
            and row["path"].endswith("downloads/Allegro-OAM-L-0.1.nequip.zip")
        )
        self.assertEqual(80738705, allegro_oam["bytes"])
        self.assertTrue(allegro_oam["matches_registry_hash"])
        self.assertEqual("general_baseline", allegro_oam["target_role"])

    def test_nialti_mace_fitsnap_lead_is_not_admitted_without_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_nialti_mace_fitsnap_song_2026_lead_audit.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nialti_mace_fitsnap_song_2026_lead"
        ]

        self.assertTrue(audit["classification"]["is_dedicated_gamma_prime_lead"])
        self.assertFalse(audit["classification"]["admittable_now"])
        self.assertFalse(audit["public_model_search"]["public_checkpoint_found"])
        self.assertEqual("conference_abstract_no_model_file", candidate["artifact_kind"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertIn("no public checkpoint", candidate["rejection_reason"])

    def test_nibased_gamma_prime_thesis_lead_is_not_admitted_without_model_file(self):
        audit = json.loads(
            Path("audits/mlip_nibased_gamma_prime_mou_2023_lead_audit.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nibased_gamma_prime_mou_2023_lead"
        ]

        self.assertTrue(audit["classification"]["is_dedicated_gamma_prime_lead"])
        self.assertFalse(audit["classification"]["admittable_now"])
        self.assertEqual("aws_waf_challenge_html_not_pdf", audit["artifact_probe"]["pdf_download_result"])
        self.assertFalse(audit["artifact_probe"]["public_checkpoint_found"])
        self.assertFalse(audit["author_github_probe"]["public_ni_gamma_prime_model_file_found"])
        self.assertIn("HEC_GB_MCMD_MACE_scripts", audit["author_github_probe"]["public_repositories_scanned"])
        self.assertEqual("thesis_abstract_no_model_file", candidate["artifact_kind"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertIn("no public checkpoint", candidate["rejection_reason"])

    def test_tivta_w_milady_audit_tracks_downloaded_model_hashes(self):
        audit = json.loads(Path("audits/mlip_tivta_w_milady_wrobel_2025_audit.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_titavw_milady_wrobel_2025"
        ]

        artifacts = {Path(row["path"]).name: row for row in audit["downloaded_artifacts"]}

        self.assertEqual("target_specific_mlip", audit["classification"]["target_role"])
        self.assertEqual("milady_lammps", candidate["engine"])
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertIn("lammps_bso4_snap1_params.pot", artifacts)
        self.assertIn("lammps_bso4_kernl_params.pot", artifacts)
        self.assertEqual(
            "32592117665288aef4b69364552387907605e570e9dcdb16b3370d30ef755a25",
            artifacts["lammps_bso4_snap1_params.pot"]["sha256"],
        )
        self.assertEqual("pair_style milady", audit["lammps_evidence"]["potential_mod_pair_style"])

    def test_gamma_prime_public_index_audit_records_unresolved_targets(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_index_queries_2026-07-04.json").read_text(encoding="utf-8")
        )
        summary = audit["summary"]
        nonmatching_titles = {row["title"] for row in audit["notable_nonmatching_findings"]}

        self.assertFalse(summary["dedicated_ni_al_ti_model_file_found"])
        self.assertFalse(summary["dedicated_ni_al_ta_model_file_found"])
        self.assertFalse(summary["dedicated_ni_al_ti_ta_model_file_found"])
        self.assertEqual(0, summary["github_exact_repository_hits"])
        self.assertEqual(0, summary["huggingface_exact_model_hits"])
        self.assertIn("Machine Learning Potentials for Ti-V-Ta-W Alloys (Linear ML and Kernel ML)", nonmatching_titles)
        self.assertIn("Alloy53_VASP", nonmatching_titles)

    def test_gamma_prime_repository_search_round2_records_no_public_model_hits(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_repository_search_2026-07-04_round2.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(audit["summary"]["dedicated_ni_al_ti_model_file_found"])
        self.assertFalse(audit["summary"]["dedicated_ni_al_ta_model_file_found"])
        self.assertFalse(audit["summary"]["dedicated_ni_al_ti_ta_model_file_found"])
        self.assertEqual(0, audit["summary"]["huggingface_exact_model_hits"])
        self.assertEqual(0, audit["summary"]["zenodo_filtered_model_hits"])
        self.assertEqual(0, audit["summary"]["materials_cloud_filtered_model_hits"])
        self.assertIn("pot_nialti_mace_fitsnap_song_2026_lead", audit["new_leads"])

    def test_gamma_prime_public_search_round3_records_author_and_repository_misses(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_search_round3_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(audit["summary"]["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(audit["summary"]["mou_author_github_public_checkpoint_found"])
        self.assertFalse(audit["summary"]["song_nialti_public_checkpoint_found"])
        self.assertEqual(0, audit["summary"]["github_repository_exact_hits_for_nialta_or_nialtita"])
        self.assertEqual(0, audit["summary"]["aissquare_filtered_target_matches"])
        self.assertEqual(131, audit["aissquare_public_model_list_scan"]["total_scanned"])
        self.assertEqual(0, audit["aissquare_public_model_list_scan"]["filtered_match_count"])
        repo_names = {row["name"] for row in audit["mou_thesis_author_probe"]["public_repositories"]}
        self.assertIn("HEC_GB_MCMD_MACE_scripts", repo_names)

    def test_gamma_prime_public_repository_search_round4_records_index_blockers(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_repository_search_round4_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertFalse(summary["arxiv_2603_21207_public_checkpoint_found"])
        self.assertEqual([403], audit["figshare_api"]["statuses"])
        self.assertEqual(0, audit["osf_api"]["preprint_exact_title_hits"])
        self.assertEqual(0, audit["harvard_dataverse_api"]["filtered_relevant_hits"])
        self.assertEqual(0, len(audit["darus_api"]["multicomponent_gamma_prime_hits"]))
        self.assertEqual(2, len(audit["darus_api"]["known_relevant_hits"]))
        datacite_hits = {row["doi"] for row in audit["datacite_api"]["notable_hits"]}
        self.assertIn("10.48550/arxiv.2603.21207", datacite_hits)
        self.assertFalse(audit["arxiv_2603_21207_probe"]["public_checkpoint_or_model_file_found"])
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_candidates"])

    def test_gamma_prime_author_and_index_search_round5_keeps_song_lead_unadmitted(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_author_and_index_search_round5_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertTrue(summary["programmaster_confirms_nialti_mlip_lead"])
        self.assertFalse(summary["programmaster_exposes_model_artifact"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual(0, summary["github_exact_repository_hits"])
        self.assertEqual(0, summary["huggingface_exact_model_hits"])
        self.assertEqual(0, summary["zenodo_exact_title_hits_for_song_nialti"])
        self.assertTrue(summary["crossref_reinforces_classical_nialti_meam_only"])
        self.assertEqual([], audit["programmaster_nialti_lead"]["artifact_links_found_on_page"])
        self.assertEqual(10, audit["github_repository_api"]["query_count"])
        self.assertEqual(10, audit["huggingface_models_api"]["query_count"])
        self.assertIn("pot_nialti_mace_fitsnap_song_2026_lead", audit["reinforced_candidates"])
        self.assertEqual(
            "pot_nialti_meam_kim_2017_classical_fallback",
            audit["crossref_and_classical_fallback_check"]["notable_hit"]["already_tracked_as"],
        )

    def test_gamma_prime_code_host_and_profile_round6_records_no_public_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_code_host_and_profile_search_round6_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertEqual(0, summary["gitlab_public_project_hits"])
        self.assertEqual(0, summary["codeberg_repository_hits"])
        self.assertEqual(410, summary["bitbucket_api_status"])
        self.assertEqual(0, summary["orcid_target_like_works"])
        self.assertFalse(summary["lanl_profile_exposes_model_artifact"])
        self.assertEqual(0, summary["nomad_target_checkpoint_hits"])
        self.assertEqual(404, summary["materials_cloud_query_endpoint_status"])
        self.assertEqual(0, audit["codeberg_repository_api"]["total_hits"])
        self.assertEqual([], audit["author_profile_checks"]["orcid"]["target_like_works"])
        self.assertEqual(
            "FT8UX98FS5KtDbNBw-dU3A",
            audit["nomad_api"]["notable_noise"]["upload_id"],
        )
        self.assertIn("pot_nialta_mlip_candidate", audit["reinforced_candidates"])

    def test_gamma_prime_preprint_and_supplement_round7_records_no_target_checkpoint(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_preprint_and_supplement_search_round7_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertEqual(0, summary["osti_target_artifact_hits"])
        self.assertFalse(summary["nsf_par_target_terms_seen"])
        self.assertEqual(404, summary["researchsquare_endpoint_status"])
        self.assertEqual(0, summary["chemrxiv_target_checkpoint_hits"])
        self.assertTrue(summary["classical_fallback_only_reinforced"])
        self.assertEqual(0, audit["chemrxiv_via_crossref"]["target_checkpoint_hits"])
        self.assertFalse(audit["nsf_par"]["target_terms_seen_any_query"])
        osti_queries = {row["query"]: row for row in audit["osti_api"]["query_results"]}
        self.assertEqual(0, osti_queries["Huajing Song Ni-Al-Ti MACE FitSNAP"]["count"])
        self.assertIn("pot_nialtita_mlip_candidate", audit["reinforced_candidates"])

    def test_gamma_prime_code_and_data_api_round8_records_no_file_pattern_hits(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_code_and_data_api_search_round8_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )

        summary = audit["summary"]
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_found"])
        self.assertEqual(429, summary["grep_app_status"])
        self.assertTrue(summary["github_code_api_authenticated_or_rate_limited"])
        self.assertEqual(0, summary["huggingface_model_hits"])
        self.assertEqual(0, summary["huggingface_dataset_hits"])
        self.assertFalse(summary["mendeley_target_terms_seen"])
        self.assertEqual(0, summary["dryad_hits"])
        self.assertEqual(0, summary["core_target_checkpoint_hits"])
        self.assertEqual(0, summary["openaire_new_target_checkpoint_hits"])
        self.assertEqual(12, audit["huggingface_api"]["query_count"])
        self.assertEqual(0, audit["dryad_api"]["total_hits"])
        self.assertEqual(0, audit["openaire_api"]["new_target_checkpoint_hits"])
        self.assertIn("pot_nialti_mace_fitsnap_song_2026_lead", audit["reinforced_candidates"])

    def test_gamma_prime_programmaster_followup_round9_keeps_lead_unadmitted(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_programmaster_followup_round9_2026-07-04.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nialti_mace_fitsnap_song_2026_lead"
        ]

        summary = audit["summary"]
        self.assertTrue(summary["programmaster_confirms_nialti_mlip_lead"])
        self.assertFalse(summary["programmaster_exposes_model_artifact"])
        self.assertFalse(summary["downloadable_dedicated_gamma_prime_checkpoint_found"])
        self.assertEqual(0, summary["programmaster_artifact_link_count"])
        self.assertEqual("planned", summary["proceedings_inclusion_status"])
        self.assertEqual([], audit["programmaster_source_page"]["artifact_links_found_on_page"])
        self.assertFalse(audit["registry_effect"]["potentials_catalog_changed"])
        self.assertFalse(audit["registry_effect"]["accepted_potential_created"])
        self.assertEqual("research_needed", candidate["acceptance_status"])
        self.assertEqual("conference_abstract_no_model_file", candidate["artifact_kind"])
        self.assertIn("pot_nialti_mlip_candidate", audit["reinforced_candidates"])

    def test_nialcoti_meam_audit_keeps_classical_fallback_out_of_mlip_bucket(self):
        audit = json.loads(
            Path("audits/mlip_nialcoti_meam_huang_2024_classical_fallback_audit.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nialcoti_meam_huang_2024_classical_fallback"
        ]
        artifacts = {Path(row["path"]).name: row for row in audit["downloaded_artifacts"]}

        self.assertTrue(audit["classification"]["not_mlip"])
        self.assertEqual("classical_fallback", candidate["target_role"])
        self.assertEqual("classical_potential_file", candidate["artifact_kind"])
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertIn("not an MLIP", candidate["rejection_reason"])
        self.assertIn("Ni-Al-Ti", audit["lammps_evidence"]["contains_terms"])
        self.assertEqual(
            "974e59ebc93c14a7207591c365877b2fdbc0404253255eeccb32b63a61e270d0",
            artifacts["library.meam"]["sha256"],
        )
        self.assertEqual(
            "a883ff0b4e4f12f33246912ef1cd854568a8b54224d354a14d9dfd2b4ca440ff",
            artifacts["NiAlCoTi.meam"]["sha256"],
        )

    def test_nialti_meam_audit_keeps_classical_fallback_out_of_mlip_bucket(self):
        audit = json.loads(
            Path("audits/mlip_nialti_meam_kim_2017_classical_fallback_audit.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidate = {row["candidate_id"]: row for row in registry["candidates"]}[
            "pot_nialti_meam_kim_2017_classical_fallback"
        ]
        artifacts = {Path(row["path"]).name: row for row in audit["downloaded_artifacts"]}

        self.assertTrue(audit["classification"]["not_mlip"])
        self.assertEqual("classical_fallback", candidate["target_role"])
        self.assertEqual("classical_potential_file", candidate["artifact_kind"])
        self.assertEqual("blocked", candidate["acceptance_status"])
        self.assertIn("not an MLIP", candidate["rejection_reason"])
        self.assertIn("Ni-Al-Ti", audit["lammps_evidence"]["contains_terms"])
        self.assertEqual(
            "abc3c9be75d99ada38f66a17398da927d497ed61a275ec67580a9f2556b7b51b",
            artifacts["library.meam"]["sha256"],
        )
        self.assertEqual(
            "06ed0ac0eb798dbc41ef0690a325583a67ca104c9ffd54c38e29bc4f14cfe1c8",
            artifacts["NiAlTi.meam"]["sha256"],
        )

    def test_candidates_audit_validates_registry_without_creating_potential_dirs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_mace_mp_0b3_medium",
                                "target_role": "general_baseline",
                                "family": "MACE",
                                "engine": "mace",
                                "model_name": "MACE-MP-0b3-medium",
                                "source_url": "https://github.com/ACEsuit/mace-foundations",
                                "doi": "10.48550/arXiv.2401.00096",
                                "license": "MIT",
                                "redistribution_policy": "metadata_only",
                                "elements": ["Cr", "Co", "Ni"],
                                "phases": ["fcc", "bcc"],
                                "target_systems": ["CrCoNi"],
                                "install_method": "pip install mace-torch",
                                "artifact_kind": "package_pretrained",
                                "acceptance_status": "candidate",
                                "rejection_reason": "",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "candidates",
                    "audit",
                    "--root",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual(1, payload["candidate_count"])
        self.assertEqual(0, payload["issue_count"])
        self.assertFalse((root / "data" / "potentials").exists())

    def test_candidates_audit_reports_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_bad",
                                "family": "MACE",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "candidates",
                    "audit",
                    "--root",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)

        self.assertEqual(1, result.returncode)
        self.assertIn("missing_required_field", {issue["code"] for issue in payload["issues"]})

    def test_download_candidate_stages_external_artifact_in_local_cache_with_hash_audit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "a" / "model.mtp"
            source.parent.mkdir()
            model_bytes = b"level = 16\nspecies = Cr Co Ni\n"
            source.write_bytes(model_bytes)
            expected_sha = hashlib.sha256(model_bytes).hexdigest()
            second_source = root / "b" / "model.mtp"
            second_source.parent.mkdir()
            second_source.write_bytes(b"second model")
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_crconi_mtp_csro",
                                "target_role": "target_specific_mlip",
                                "family": "MTP",
                                "engine": "mtp_lammps",
                                "model_name": "CrCoNi cSRO MTP test",
                                "source_url": source.as_uri(),
                                "artifact_urls": [source.as_uri(), second_source.as_uri()],
                                "doi": "unassigned",
                                "license": "license-audit-needed",
                                "redistribution_policy": "metadata_only",
                                "elements": ["Cr", "Co", "Ni"],
                                "phases": ["fcc"],
                                "target_systems": ["CrCoNi"],
                                "install_method": "download external model file",
                                "artifact_kind": "external_model_file",
                                "acceptance_status": "candidate",
                                "rejection_reason": "",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "download",
                    "--root",
                    str(root),
                    "--candidate-id",
                    "pot_crconi_mtp_csro",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            cached_file = root / "local_potential_cache" / "pot_crconi_mtp_csro" / "downloads" / "model.mtp"
            audit = json.loads(
                (root / "local_potential_cache" / "pot_crconi_mtp_csro" / "download_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            cached_exists = cached_file.exists()
            cached_paths = [row["path"] for row in payload["files"]]

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("downloaded", payload["status"])
        self.assertFalse((root / "data" / "potentials" / "pot_crconi_mtp_csro").exists())
        self.assertTrue(cached_exists)
        self.assertEqual(len(cached_paths), len(set(cached_paths)))
        self.assertEqual(expected_sha, payload["files"][0]["sha256"])
        self.assertEqual(expected_sha, audit["files"][0]["sha256"])

    def test_download_candidate_supports_fairdata_authorized_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            model_bytes = b"tabgap model"
            expected_sha = hashlib.sha256(model_bytes).hexdigest()
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_monbtavw_tabgap_byggmastar",
                                "target_role": "target_specific_mlip",
                                "family": "tabGAP",
                                "engine": "tabgap_lammps",
                                "model_name": "MoNbTaVW tabGAP test",
                                "source_url": "https://etsin.fairdata.fi/dataset/test",
                                "doi": "10.23729/test",
                                "license": "CC-BY-4.0",
                                "redistribution_policy": "redistributable_with_attribution",
                                "elements": ["Mo", "Nb", "Ta", "V", "W"],
                                "phases": ["bcc"],
                                "target_systems": ["MoNbTaVW"],
                                "install_method": "download Fairdata model file",
                                "artifact_kind": "external_model_file",
                                "fairdata_files": [
                                    {
                                        "dataset_id": "test-dataset",
                                        "path": "/potentials/Mo-Nb-Ta-V-W.tabgap",
                                        "filename": "Mo-Nb-Ta-V-W.tabgap",
                                        "sha256": expected_sha,
                                        "size": len(model_bytes),
                                    }
                                ],
                                "acceptance_status": "blocked",
                                "rejection_reason": "missing engine",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            def fake_download(spec, destination):
                destination.write_bytes(model_bytes)
                return "https://download.fairdata.fi/download?token=test"

            with patch("alloy_xrd_bench.mlip._download_fairdata_file", side_effect=fake_download):
                payload = download_candidate(root, "pot_monbtavw_tabgap_byggmastar")
            cached_file = (
                root
                / "local_potential_cache"
                / "pot_monbtavw_tabgap_byggmastar"
                / "downloads"
                / "Mo-Nb-Ta-V-W.tabgap"
            )
            audit = json.loads(
                (
                    root
                    / "local_potential_cache"
                    / "pot_monbtavw_tabgap_byggmastar"
                    / "download_audit.json"
                ).read_text(encoding="utf-8")
            )
            cached_exists = cached_file.exists()
            potential_dir_exists = (root / "data" / "potentials" / "pot_monbtavw_tabgap_byggmastar").exists()

        self.assertEqual("downloaded", payload["status"])
        self.assertTrue(cached_exists)
        self.assertFalse(potential_dir_exists)
        self.assertEqual(expected_sha, payload["files"][0]["sha256"])
        self.assertTrue(payload["files"][0]["hash_matches_expected"])
        self.assertEqual(expected_sha, audit["files"][0]["expected_sha256"])

    def test_download_candidate_supports_dataverse_files_with_md5_audit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            model_bytes = b"Ni3Al MTP"
            expected_md5 = hashlib.md5(model_bytes).hexdigest()
            expected_sha = hashlib.sha256(model_bytes).hexdigest()
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_ni3al_mtp_darus_yield_anomaly_2024",
                                "target_role": "gamma_prime_specific_mlip",
                                "family": "MTP",
                                "engine": "mtp_lammps",
                                "model_name": "Ni3Al MTP test",
                                "source_url": "https://doi.org/10.18419/DARUS-4480",
                                "doi": "10.18419/DARUS-4480",
                                "license": "DaRUS-DUA",
                                "redistribution_policy": "metadata_only",
                                "elements": ["Ni", "Al"],
                                "phases": ["l12"],
                                "target_systems": ["Ni-Al"],
                                "install_method": "download Dataverse model file",
                                "artifact_kind": "external_model_file",
                                "dataverse_files": [
                                    {
                                        "file_id": 317664,
                                        "filename": "Ni3Al.mtp",
                                        "md5": expected_md5,
                                        "size": len(model_bytes),
                                    }
                                ],
                                "acceptance_status": "blocked",
                                "rejection_reason": "missing engine",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            def fake_download(spec, destination):
                destination.write_bytes(model_bytes)
                return f"https://darus.uni-stuttgart.de/api/access/datafile/{spec['file_id']}"

            with patch("alloy_xrd_bench.mlip._download_dataverse_file", side_effect=fake_download):
                payload = download_candidate(root, "pot_ni3al_mtp_darus_yield_anomaly_2024")
            cached_file = (
                root
                / "local_potential_cache"
                / "pot_ni3al_mtp_darus_yield_anomaly_2024"
                / "downloads"
                / "Ni3Al.mtp"
            )
            cached_exists = cached_file.exists()
            potential_dir_exists = (root / "data" / "potentials" / "pot_ni3al_mtp_darus_yield_anomaly_2024").exists()

        self.assertEqual("downloaded", payload["status"])
        self.assertTrue(cached_exists)
        self.assertFalse(potential_dir_exists)
        self.assertEqual(expected_sha, payload["files"][0]["sha256"])
        self.assertEqual(expected_md5, payload["files"][0]["md5"])
        self.assertTrue(payload["files"][0]["md5_matches_expected"])

    def test_download_candidate_extracts_declared_zip_members_without_staging_archive(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            archive = root / "RHEA-data.zip"
            nep_bytes = b"nep4 test model\n"
            readme_bytes = b"RHEA potential files\n"
            expected_sha = hashlib.sha256(nep_bytes).hexdigest()
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("potential_files/nep-RHEA-large.txt", nep_bytes)
                handle.writestr("potential_files/README", readme_bytes)

            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_candidates.yml").write_text(
                yaml.safe_dump(
                    {
                        "candidates": [
                            {
                                "candidate_id": "pot_rhea9_nep_byggmastar_2026",
                                "target_role": "refractory_alloy_specific_mlip",
                                "family": "NEP",
                                "engine": "gpumd",
                                "model_name": "RHEA NEP test",
                                "source_url": "https://zenodo.org/records/18863415",
                                "doi": "10.5281/zenodo.18863415",
                                "license": "CC-BY-4.0",
                                "redistribution_policy": "redistributable_with_attribution",
                                "elements": ["Cr", "Hf", "Mo", "Nb", "Ta", "Ti", "V", "W", "Zr"],
                                "phases": ["bcc"],
                                "target_systems": ["TiTaVW"],
                                "install_method": "extract NEP member from RHEA-data.zip",
                                "artifact_kind": "external_model_file",
                                "zip_members": [
                                    {
                                        "archive_url": archive.as_uri(),
                                        "member_path": "potential_files/nep-RHEA-large.txt",
                                        "filename": "nep-RHEA-large.txt",
                                        "sha256": expected_sha,
                                        "size": len(nep_bytes),
                                    },
                                    {
                                        "archive_url": archive.as_uri(),
                                        "member_path": "potential_files/README",
                                        "filename": "README",
                                        "size": len(readme_bytes),
                                    },
                                ],
                                "acceptance_status": "blocked",
                                "rejection_reason": "missing gpumd",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            payload = download_candidate(root, "pot_rhea9_nep_byggmastar_2026")
            cached_file = (
                root
                / "local_potential_cache"
                / "pot_rhea9_nep_byggmastar_2026"
                / "downloads"
                / "nep-RHEA-large.txt"
            )
            staged_archive = (
                root
                / "local_potential_cache"
                / "pot_rhea9_nep_byggmastar_2026"
                / "downloads"
                / "RHEA-data.zip"
            )
            cached_exists = cached_file.exists()
            staged_archive_exists = staged_archive.exists()
            potential_dir_exists = (root / "data" / "potentials" / "pot_rhea9_nep_byggmastar_2026").exists()

        self.assertEqual("downloaded", payload["status"])
        self.assertTrue(cached_exists)
        self.assertFalse(staged_archive_exists)
        self.assertFalse(potential_dir_exists)
        self.assertEqual(expected_sha, payload["files"][0]["sha256"])
        self.assertTrue(payload["files"][0]["hash_matches_expected"])
        self.assertEqual("zip_member", payload["files"][0]["source"])

    def test_verify_lammps_backed_adapters_report_readiness_blocker(self):
        payloads = []
        for engine in ("mtp_lammps", "tabgap_lammps", "snap_lammps", "deepmd_lammps"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tempdir:
                root = Path(tempdir)
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "scripts/make_minimal_workspace.py",
                        "--root",
                        str(root),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
                catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
                catalog["potentials"][0]["engine"] = engine
                catalog["potentials"][0]["elements"] = ["Cr", "Co", "Ni"]
                catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "alloy_xrd_bench.potentials",
                        "verify",
                        "--root",
                        str(root),
                        "--potential-id",
                        "pot_mace_mp_0",
                        "--reference-initial-id",
                        "init_CrCoNi_fcc_000001",
                        "--json",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(result.stdout)
                payloads.append(payload)

            self.assertEqual("", result.stderr)
            self.assertEqual(1, result.returncode)
        for payload in payloads:
            self.assertEqual("fail", payload["status"])
            self.assertIn(payload["load"]["code"], {"missing_engine", "missing_model"})
            if payload["load"]["code"] == "missing_model":
                self.assertTrue(payload["load"]["executable"].lower().endswith("lmp.exe"))

    def test_nep_adapter_uses_mdapy_nep4ase_for_single_point_when_available(self):
        from ase.calculators.calculator import Calculator, all_changes
        from pymatgen.core import Lattice, Structure

        class FakeNepCalculator(Calculator):
            implemented_properties = ["energy", "forces", "stress"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                atom_count = len(atoms) if atoms is not None else 0
                self.results = {
                    "energy": -4.2,
                    "forces": np.zeros((atom_count, 3)),
                    "stress": np.zeros(6),
                }

        fake_nep4ase = types.ModuleType("mdapy.nep4ase")
        fake_nep4ase.NEP4ASE = lambda model_filename: FakeNepCalculator()
        fake_mdapy = types.ModuleType("mdapy")

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            files_dir = root / "data" / "potentials" / "pot_nep_test" / "potential" / "files"
            files_dir.mkdir(parents=True)
            (files_dir / "nep.in").write_text("type 1 Ni\n", encoding="utf-8")
            (files_dir / "nep.txt").write_text("nep4 1 Ni\n", encoding="utf-8")
            potential = {
                "potential_id": "pot_nep_test",
                "engine": "gpumd",
                "elements": ["Ni"],
                "file_sha256": "",
            }
            structure = Structure(Lattice.cubic(3.6), ["Ni"], [[0.0, 0.0, 0.0]])
            loaded = LoadedInitial(
                initial_id="init_Ni_test",
                row={"composition": {"Ni": 1}},
                document=structure.as_dict(),
                structure=structure,
            )

            with patch.dict(sys.modules, {"mdapy": fake_mdapy, "mdapy.nep4ase": fake_nep4ase}):
                adapter = NepAdapter()
                load = adapter.load(root, potential)
                single_point = adapter.single_point(root, potential, loaded)

        self.assertTrue(load["ok"])
        self.assertEqual("mdapy_nep", load["adapter"])
        self.assertTrue(load["model_file"].endswith("nep.txt"))
        self.assertEqual("pass", single_point["status"])
        self.assertEqual(-4.2, single_point["energy_eV"])

    def test_package_adapters_report_cached_model_file_hashes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            home = root / "home"
            mace_file = home / ".cache" / "mace" / "model"
            mace_file.parent.mkdir(parents=True)
            mace_file.write_bytes(b"mace model")
            mace_hash = hashlib.sha256(b"mace model").hexdigest()
            matgl_state = (
                home
                / ".cache"
                / "matgl"
                / "models--materialyze--M3GNet-PES-MatPES-PBE-2025.2"
                / "snapshots"
                / "abc"
                / "state.pt"
            )
            matgl_state.parent.mkdir(parents=True)
            matgl_state.write_bytes(b"matgl state")
            matgl_hash = hashlib.sha256(b"matgl state").hexdigest()

            with patch("pathlib.Path.home", return_value=home):
                mace_files = MaceAdapter().model_files(root, {"potential_id": "pot_mace", "file_sha256": mace_hash})
                matgl_files = MatglAdapter().model_files(
                    root,
                    {"potential_id": "pot_matgl", "file_sha256": matgl_hash},
                )

        self.assertEqual(mace_hash, mace_files[0]["sha256"])
        self.assertTrue(mace_files[0]["matches_catalog_hash"])
        self.assertEqual(matgl_hash, matgl_files[0]["sha256"])
        self.assertTrue(matgl_files[0]["matches_catalog_hash"])

    def test_verify_fake_potential_writes_audit_and_summary_only(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            make_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["engine"] = "fake"
            catalog["potentials"][0]["elements"] = ["Cr", "Co", "Ni"]
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            before_relaxations = pd.read_parquet(
                root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "generated"
                / "manifests"
                / "relaxations.parquet"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "verify",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_mace_mp_0",
                    "--reference-initial-id",
                    "init_CrCoNi_fcc_000001",
                    "--smoke-initial-id",
                    "init_CrCoNi_fcc_000001",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            after_relaxations = pd.read_parquet(
                root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "generated"
                / "manifests"
                / "relaxations.parquet"
            )
            payload = json.loads(result.stdout)
            audit_path = (
                root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "audits"
                / "potential_verification.json"
            )
            summary_path = root / "audits" / "potential_verification_summary.parquet"
            audit_exists = audit_path.exists()
            summary_exists = summary_path.exists()

        self.assertEqual(0, make_result.returncode)
        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("pass", payload["status"])
        self.assertTrue(audit_exists)
        self.assertTrue(summary_exists)
        self.assertEqual(len(before_relaxations), len(after_relaxations))

    def test_verify_unsupported_elements_are_skipped_not_failed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["engine"] = "fake"
            catalog["potentials"][0]["elements"] = ["Al"]
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "verify",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_mace_mp_0",
                    "--smoke-initial-id",
                    "init_CrCoNi_fcc_000001",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("pass", payload["status"])
        self.assertEqual("skipped_unsupported_elements", payload["smoke_relaxations"][0]["status"])

    def test_round37_records_gamma_prime_public_checkpoint_misses(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_public_checkpoint_search_round37_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertEqual([], summary["new_downloaded_model_files"])
        self.assertFalse(summary["accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_modified"])
        self.assertEqual(
            audit["candidate_registry_effect"]["formal_manifest_counts_before"],
            audit["candidate_registry_effect"]["formal_manifest_counts_after"],
        )

        github_counts = {
            row["query"]: row["total_count"]
            for row in audit["query_matrix"]["github_repository_api"]["total_hits_by_query"]
        }
        hf_counts = {
            row["query"]: row["count"]
            for row in audit["query_matrix"]["hugging_face_api"]["model_hits_by_query"]
        }
        self.assertTrue(github_counts)
        self.assertTrue(hf_counts)
        self.assertTrue(all(count == 0 for count in github_counts.values()))
        self.assertTrue(all(count == 0 for count in hf_counts.values()))
        self.assertEqual(0, audit["query_matrix"]["zenodo_api"]["target_model_hits"])
        self.assertEqual(0, audit["query_matrix"]["crossref_api"]["target_model_artifact_hits"])

        leads = {row["candidate_id"]: row for row in audit["primary_lead_rechecks"]}
        self.assertEqual("on_request", leads["pot_nicoal_nep_qin_2025_lead"]["data_availability"])
        self.assertFalse(audit["external_source_rechecks"]["materialsmodeling_nep_table"]["contains_nicoal_entry"])
        smcx = audit["external_source_rechecks"]["smc_x_supplementary_repo"]
        self.assertEqual(1, smcx["file_count"])
        self.assertFalse(smcx["has_direct_model_file"])
        self.assertIsNone(smcx["license"])
        self.assertEqual(0, smcx["releases_count"])

        for candidate_id in audit["candidate_registry_effect"]["updated_rejection_reasons"]:
            self.assertEqual("research_needed", candidates[candidate_id]["acceptance_status"])
            self.assertIn("Round-37", candidates[candidate_id]["rejection_reason"])

    def test_round46_records_lammps_and_deepmd_runtime_resolution(self):
        audit = json.loads(
            Path("audits/mlip_lammps_deepmd_runtime_resolution_round46_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["lammps_bundled_executable_found"])
        self.assertTrue(summary["deepmd_dp_executable_found"])
        self.assertFalse(summary["deepmd_model_show_ok"])
        self.assertFalse(summary["accepted_potentials_changed"])
        self.assertFalse(summary["formal_generated_manifests_changed"])

        lammps_probe = audit["lammps_probe"]
        self.assertIn("ML-SNAP", lammps_probe["installed_ml_packages_detected"])
        self.assertIn("snap", lammps_probe["available_pair_styles"])
        self.assertIn("mliap", lammps_probe["available_pair_styles"])
        self.assertIn("meam", lammps_probe["available_pair_styles"])
        self.assertIn("uf3", lammps_probe["available_pair_styles"])
        for missing_style in ("deepmd", "pace", "hdnnp", "mlip", "quip", "tabgap"):
            self.assertIn(missing_style, lammps_probe["missing_pair_styles_or_interfaces"])

        deepmd_probe = audit["deepmd_probe"]
        self.assertEqual("fail", deepmd_probe["status"])
        self.assertEqual("pytorch_cxx11_abi_mismatch", deepmd_probe["error_code"])
        self.assertIn("CXX11_ABI_FLAG", deepmd_probe["evidence"])

        for candidate_id in (
            "pot_monbtavw_dpa2_lam_jiang_2026",
            "pot_crconi_mtp_csro",
            "pot_monbtavw_tabgap_byggmastar",
            "pot_wtacrv_tabgap_byggmastar_2025",
        ):
            self.assertIn("Round-46", candidates[candidate_id]["rejection_reason"])

    def test_round47_records_gamma_prime_checkpoint_misses_and_cantor_mtp_confirmation(self):
        audit = json.loads(
            Path("audits/mlip_gamma_prime_and_cantor_mtp_followup_round47_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertFalse(summary["new_admittable_gamma_prime_checkpoint_found"])
        self.assertFalse(summary["new_accepted_potential_created"])
        self.assertFalse(summary["formal_generated_manifests_changed"])
        self.assertTrue(summary["cocrfemnni_cantor_mtp_public_model_confirmed"])
        self.assertEqual("GPL-3.0", summary["cocrfemnni_cantor_mtp_license_confirmed"])

        github_counts = {
            row["query"]: row["total_count"]
            for row in audit["public_index_queries"]["github_repository_api"]
        }
        self.assertTrue(github_counts)
        self.assertTrue(all(count == 0 for count in github_counts.values()))
        hf_counts = {
            row["query"]: row["count"]
            for row in audit["public_index_queries"]["hugging_face_model_api"]
        }
        self.assertTrue(hf_counts)
        self.assertTrue(all(count == 0 for count in hf_counts.values()))

        primary = audit["primary_source_rechecks"]
        self.assertFalse(primary["gpumd_nep_data_table"]["contains_nicoal_entry"])
        self.assertFalse(primary["materialsmodeling_nep_models_table"]["contains_nicoal_entry"])
        cantor = primary["cantor_alloy_mtp_github"]
        self.assertIn("Trained_MTP_models/mtp8-rb10", cantor["trained_model_entries_seen"])
        self.assertIn("GNU GENERAL PUBLIC LICENSE", cantor["license_evidence"])

        for candidate_id in (
            "pot_cocrfemnni_mtp_cantor_alloy_mlip",
            "pot_nialti_mace_fitsnap_song_2026_lead",
            "pot_nicoal_nep_qin_2025_lead",
            "pot_nialti_mlip_candidate",
            "pot_nialta_mlip_candidate",
            "pot_nialtita_mlip_candidate",
        ):
            self.assertIn("Round-47", candidates[candidate_id]["rejection_reason"])

    def test_round48_records_mlip3_mtp_reference_inference_without_acceptance(self):
        audit = json.loads(
            Path("audits/mlip3_mtp_runtime_reference_round48_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["mlip3_mlp_built"])
        self.assertEqual(2, summary["reference_inference_pass_count"])
        self.assertFalse(summary["accepted_potentials_changed"])
        self.assertFalse(summary["formal_generated_manifests_changed"])
        self.assertIn("type", summary["reason_not_accepted_yet"].lower())

        for candidate_id in ("pot_crconi_mtp_csro", "pot_cocrfemnni_mtp_cantor_alloy_mlip"):
            case = audit["cases"][candidate_id]
            inference = case["reference_inference"]
            self.assertEqual("pass", inference["status"])
            self.assertIsInstance(inference["energy"], float)
            self.assertGreater(inference["atom_count"], 0)
            self.assertGreater(inference["max_force"], 0)
            self.assertIn("assumed", case["type_map_status"])
            self.assertIn("Round-48", candidates[candidate_id]["rejection_reason"])
        self.assertEqual("accepted", candidates["pot_crconi_mtp_csro"]["acceptance_status"])
        self.assertEqual("blocked", candidates["pot_cocrfemnni_mtp_cantor_alloy_mlip"]["acceptance_status"])

    def test_round49_records_crconi_mtp_mapping_and_smoke_split(self):
        audit = json.loads(
            Path("audits/mlip_mtp_mapping_and_smoke_round49_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}

        summary = audit["summary"]
        self.assertTrue(summary["crconi_mtp_type_mapping_confirmed"])
        self.assertTrue(summary["crconi_mtp_relaxation_smoke_passed"])
        self.assertFalse(summary["cocrfemnni_mtp_type_mapping_confirmed"])
        self.assertEqual("ambiguous_primary_source_evidence", summary["cocrfemnni_mtp_mapping_status"])
        self.assertFalse(summary["accepted_potentials_changed"])
        self.assertFalse(summary["formal_generated_manifests_changed"])

        crconi = audit["crconi_mtp_csro"]
        self.assertEqual("confirmed", crconi["mapping_evidence"]["status"])
        self.assertEqual({"0": "Cr", "1": "Co", "2": "Ni"}, crconi["mapping_evidence"]["type_map"])
        smoke = crconi["relaxation_smoke"]
        self.assertEqual("pass", smoke["status"])
        self.assertLess(smoke["relaxed_max_force_ev_per_angstrom"], smoke["force_gate_ev_per_angstrom"])
        self.assertFalse(smoke["formal_manifest_written"])
        self.assertTrue(Path(smoke["relaxed_cfg"]).exists())

        cantor = audit["cocrfemnni_mtp_cantor_alloy_mlip"]
        self.assertEqual("ambiguous", cantor["mapping_evidence"]["status"])
        self.assertGreaterEqual(len(cantor["mapping_evidence"]["conflicting_primary_source_observations"]), 3)
        self.assertFalse(cantor["reference_inference"]["formal_acceptance_usable"])

        self.assertIn("Round-49", candidates["pot_crconi_mtp_csro"]["rejection_reason"])
        self.assertIn("Round-49", candidates["pot_cocrfemnni_mtp_cantor_alloy_mlip"]["rejection_reason"])
        self.assertEqual("accepted", candidates["pot_crconi_mtp_csro"]["acceptance_status"])
        self.assertEqual("blocked", candidates["pot_cocrfemnni_mtp_cantor_alloy_mlip"]["acceptance_status"])

    def test_round50_accepts_crconi_mtp_and_generates_formal_relaxed_structures(self):
        audit = json.loads(
            Path("audits/mlip_crconi_mtp_csro_formal_admission_round50_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}
        catalog = yaml.safe_load(Path("data/registry/potentials_catalog.yml").read_text(encoding="utf-8"))
        potentials = {row["potential_id"]: row for row in catalog["potentials"]}

        summary = audit["summary"]
        self.assertTrue(summary["accepted_potential_created"])
        self.assertTrue(summary["formal_generated_manifests_changed"])
        self.assertFalse(summary["formal_xrd_generated"])
        self.assertEqual("pass", summary["verification_status"])
        self.assertEqual(5, summary["formal_relaxed_structure_count"])
        self.assertEqual(5, summary["formal_relaxation_count"])
        self.assertEqual(20, summary["formal_provenance_edge_count"])
        self.assertLess(summary["max_fmax_ev_per_angstrom"], 0.05)

        candidate = candidates["pot_crconi_mtp_csro"]
        self.assertEqual("accepted", candidate["acceptance_status"])
        self.assertEqual("mtp_mlp", candidate["engine"])
        self.assertIn("Round-50", candidate["rejection_reason"])
        self.assertEqual("redistributable_with_attribution", candidate["redistribution_policy"])

        potential = potentials["pot_crconi_mtp_csro"]
        self.assertEqual("mtp_mlp", potential["engine"])
        self.assertEqual({"0": "Cr", "1": "Co", "2": "Ni"}, potential["type_map"])

        manifest_dir = Path("data/potentials/pot_crconi_mtp_csro/generated/manifests")
        relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
        relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")
        edges = pd.read_parquet(manifest_dir / "provenance_edges.parquet")
        self.assertEqual(5, len(relaxations))
        self.assertEqual(5, len(relaxed))
        self.assertEqual(20, len(edges))
        self.assertTrue(relaxations["converged"].all())
        self.assertEqual({"pass"}, set(relaxed["quality_tier"]))
        self.assertTrue(Path(audit["verification"]["audit_path"]).exists())

    def test_round51_accepts_ni3al_mtp_and_limits_formal_generation_to_l12(self):
        audit = json.loads(
            Path("audits/mlip_ni3al_mtp_darus_yield_anomaly_formal_admission_round51_2026-07-05.json").read_text(
                encoding="utf-8"
            )
        )
        registry = yaml.safe_load(Path("data/registry/mlip_candidates.yml").read_text(encoding="utf-8"))
        candidates = {row["candidate_id"]: row for row in registry["candidates"]}
        catalog = yaml.safe_load(Path("data/registry/potentials_catalog.yml").read_text(encoding="utf-8"))
        potentials = {row["potential_id"]: row for row in catalog["potentials"]}

        summary = audit["summary"]
        self.assertTrue(summary["accepted_potential_created"])
        self.assertTrue(summary["formal_generated_manifests_changed"])
        self.assertFalse(summary["formal_xrd_generated"])
        self.assertTrue(summary["phase_scope_filter_added"])
        self.assertEqual("pass", summary["verification_status"])
        self.assertEqual(2, summary["formal_relaxed_structure_count"])
        self.assertEqual(2, summary["formal_relaxation_count"])
        self.assertEqual(8, summary["formal_provenance_edge_count"])
        self.assertEqual("skipped_unsupported_phase", summary["unsupported_fcc_smoke_status"])
        self.assertEqual("skipped_unsupported_elements", summary["unsupported_ti_smoke_status"])

        candidate = candidates["pot_ni3al_mtp_darus_yield_anomaly_2024"]
        self.assertEqual("accepted", candidate["acceptance_status"])
        self.assertEqual("mtp_mlp", candidate["engine"])
        self.assertIn("round-51", candidate["rejection_reason"].lower())
        self.assertEqual({"0": "Al", "1": "Ni"}, candidate["type_map"])

        potential = potentials["pot_ni3al_mtp_darus_yield_anomaly_2024"]
        self.assertEqual("mtp_mlp", potential["engine"])
        self.assertEqual(["l12"], potential["phases"])
        self.assertEqual({"0": "Al", "1": "Ni"}, potential["type_map"])

        manifest_dir = Path("data/potentials/pot_ni3al_mtp_darus_yield_anomaly_2024/generated/manifests")
        relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
        relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")
        edges = pd.read_parquet(manifest_dir / "provenance_edges.parquet")
        self.assertEqual(2, len(relaxations))
        self.assertEqual(2, len(relaxed))
        self.assertEqual(8, len(edges))
        self.assertTrue(relaxations["converged"].all())
        self.assertTrue(relaxations["initial_structure_id"].str.contains("_l12_").all())
        self.assertFalse(relaxations["initial_structure_id"].str.contains("_fcc_").any())
        self.assertEqual({"pass"}, set(relaxed["quality_tier"]))
        self.assertFalse((manifest_dir / "xrd_samples.parquet").exists())
        self.assertTrue(Path(audit["verification"]["audit_path"]).exists())

    def test_smoke_mlip_cli_writes_audit_without_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["engine"] = "fake"
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            manifest_path = (
                root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "generated"
                / "manifests"
                / "relaxed_structures.parquet"
            )
            before = pd.read_parquet(manifest_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.relax",
                    "smoke-mlip",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_mace_mp_0",
                    "--initial-id",
                    "init_CrCoNi_fcc_000001",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            after = pd.read_parquet(manifest_path)
            payload = json.loads(result.stdout)
            smoke_audit = (
                root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "audits"
                / "smoke_relaxations"
                / "init_CrCoNi_fcc_000001.json"
            )
            smoke_audit_exists = smoke_audit.exists()

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("pass", payload["status"])
        self.assertTrue(smoke_audit_exists)
        self.assertEqual(len(before), len(after))


if __name__ == "__main__":
    unittest.main()
