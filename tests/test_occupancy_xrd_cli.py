import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import h5py
import numpy as np
from pymatgen.core import Lattice, Structure


class OccupancyXrdCliTests(unittest.TestCase):
    def test_legacy_windows_source_path_has_portable_filename(self):
        from alloy_xrd_bench.occupancy_production import _source_csv_name

        self.assertEqual(
            "hea.2023-04-06.csv",
            _source_csv_name(
                {"source_csv": r"E:\project\xrd-benchmark\raw\hea.2023-04-06.csv"}
            ),
        )

    def test_final_parent_topology_overrides_conflicting_source_lattice(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "transformed_parent.csv"
            output_dir = root / "ds_hea"
            _write_transformed_parent_csv(csv_path)
            _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
            _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "1")

            structure = pd.read_parquet(
                output_dir / "manifests" / "structures.parquet"
            ).iloc[0]
            self.assertEqual("bcc", structure["source_lattice"])
            self.assertEqual("fcc_like", structure["structure_family"])
            self.assertEqual("fcc", structure["parent_lattice"])
            self.assertAlmostEqual(0.0, structure["template_mapping_rms"], places=6)

    def test_l12_sublattice_mapping_is_invariant_to_origin_choice(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "shifted_l12.csv"
            output_dir = root / "ds_hea"
            _write_shifted_l12_csv(csv_path)
            _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
            _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "1")

            structure = pd.read_parquet(
                output_dir / "manifests" / "structures.parquet"
            ).iloc[0]
            self.assertEqual("l12_like", structure["structure_family"])
            self.assertEqual({"Al": 1.0}, json.loads(structure["corner_element_distribution_json"]))
            self.assertEqual({"Ni": 1.0}, json.loads(structure["face_element_distribution_json"]))
            self.assertEqual(1.0, structure["l12_order_parameter"])

            sites = pd.read_parquet(output_dir / "manifests" / "occupancy_sites.parquet")
            self.assertEqual(
                ["Al"],
                sorted(sites[sites["parent_lattice_site"] == "corner"]["element"].unique()),
            )
            self.assertEqual(
                ["Ni"],
                sorted(sites[sites["parent_lattice_site"] == "face"]["element"].unique()),
            )

    def test_review_export_prioritizes_ordered_structures_and_writes_review_artifacts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            review_dir = root / "review"
            _write_benchmark_csv(csv_path)
            _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
            _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "4")
            _run_cli("split", "--dataset", str(output_dir))
            _run_cli("simulate", "--dataset", str(output_dir), "--subset", "pilot")

            report = _run_cli(
                "review-export",
                "--dataset",
                str(output_dir),
                "--output",
                str(review_dir),
                "--limit",
                "3",
                "--batch-size",
                "2",
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(3, report["selected_count"])
            self.assertEqual(2, report["batch_count"])
            review = pd.concat(
                [pd.read_csv(path) for path in sorted(review_dir.glob("batch_*/review.csv"))],
                ignore_index=True,
            )
            self.assertEqual({"l12_like", "b2_like"}, set(review.iloc[:2]["structure_family"]))
            self.assertTrue(review["review_status"].eq("pending").all())
            for relative_path in review["cif_path"]:
                self.assertTrue((review_dir / relative_path).exists())
            for relative_path in review["xrd_preview_path"]:
                self.assertTrue((review_dir / relative_path).exists())
            self.assertEqual(2, len(list(review_dir.glob("batch_*/index.html"))))

    def test_review_apply_updates_queue_and_rejects_unknown_structure_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            review_csv = root / "review.csv"
            _write_benchmark_csv(csv_path)
            _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
            _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "4")

            queue = pd.read_parquet(output_dir / "manifests" / "pilot_review_queue.parquet")
            updates = queue[["dataset_structure_id"]].copy()
            updates["review_status"] = ["pass", "pending", "pending", "pending"]
            updates["reviewer_label"] = queue["structure_family"]
            updates["review_notes"] = "reviewed"
            updates.to_csv(review_csv, index=False)

            report = _run_cli(
                "review-apply", "--dataset", str(output_dir), "--input", str(review_csv)
            )
            self.assertEqual("pass", report["status"])
            self.assertEqual(4, report["updated_count"])
            partial_audit = _run_cli("audit", "--dataset", str(output_dir))
            self.assertFalse(partial_audit["manual_review_complete"])
            self.assertEqual(1, partial_audit["manual_reviewed_count"])
            self.assertEqual(3, partial_audit["manual_review_pending_count"])
            self.assertEqual(1.0, partial_audit["manual_review_pass_rate"])

            updates["review_status"] = ["pass", "pass", "pass", "fail"]
            updates.to_csv(review_csv, index=False)
            report = _run_cli(
                "review-apply", "--dataset", str(output_dir), "--input", str(review_csv)
            )
            self.assertEqual("pass", report["status"])
            audit = _run_cli("audit", "--dataset", str(output_dir))
            self.assertTrue(audit["manual_review_complete"])
            self.assertEqual(0.75, audit["manual_review_pass_rate"])

            updates.loc[0, "dataset_structure_id"] = "unknown_structure"
            updates.to_csv(review_csv, index=False)
            invalid = _run_cli(
                "review-apply", "--dataset", str(output_dir), "--input", str(review_csv)
            )
            self.assertEqual("fail", invalid["status"])
            self.assertIn("unknown_structure", invalid["unknown_structure_ids"])

    def test_staged_pipeline_writes_inventory_labels_pilot_splits_and_audit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            _write_benchmark_csv(csv_path)

            inventory = _run_cli(
                "inventory", "--csv", str(csv_path), "--output", str(output_dir)
            )
            self.assertEqual("pass", inventory["status"])
            self.assertEqual(4, inventory["source_row_count"])
            self.assertEqual(4, inventory["accounted_row_count"])
            inventory_summary = json.loads(
                (output_dir / "reports" / "inventory_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual("hea_sample.csv", inventory_summary["source_csv"])
            self.assertFalse(Path(inventory_summary["source_csv"]).is_absolute())

            labels = _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "4")
            self.assertEqual("pass", labels["status"])
            structures = pd.read_parquet(output_dir / "manifests" / "structures.parquet")
            by_fid = structures.set_index("fid")
            self.assertEqual("l12_like", by_fid.loc["fid_l12", "structure_family"])
            self.assertEqual("b2_like", by_fid.loc["fid_b2", "structure_family"])
            self.assertEqual("fcc_like", by_fid.loc["fid_fcc_sqs", "structure_family"])
            self.assertNotEqual("l12_like", by_fid.loc["fid_complex", "structure_family"])
            self.assertTrue(
                (output_dir / "manifests" / "occupancy_site_groups.parquet").exists()
            )
            sites = pd.read_parquet(output_dir / "manifests" / "occupancy_sites.parquet")
            self.assertEqual(
                {8}, set(sites[sites["fid"] == "fid_b2"]["coordination_number"])
            )
            self.assertEqual(
                {12}, set(sites[sites["fid"] == "fid_fcc_sqs"]["coordination_number"])
            )
            self.assertEqual(4, len(pd.read_parquet(output_dir / "manifests" / "pilot.parquet")))

            split_report = _run_cli("split", "--dataset", str(output_dir))
            self.assertEqual("pass", split_report["status"])
            splits = pd.read_parquet(output_dir / "manifests" / "splits.parquet")
            self.assertEqual(4, len(splits))
            self.assertIn("chemistry_ood_split", splits.columns)
            self.assertIn("test", set(splits["phase_ood_split"]))
            self.assertIn("validation", set(splits["phase_ood_split"]))

            audit = _run_cli("audit", "--dataset", str(output_dir))
            self.assertEqual("pass", audit["status"])
            self.assertEqual(4, audit["inventory_rows"])

    def test_sharded_simulation_writes_4001_point_hdf5_and_phase_specific_peaks(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            _write_benchmark_csv(csv_path)
            _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
            _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "4")
            _run_cli("split", "--dataset", str(output_dir))

            report = _run_cli(
                "simulate",
                "--dataset",
                str(output_dir),
                "--subset",
                "pilot",
                "--shard-size",
                "2",
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(4, report["xrd_sample_count"])
            samples = pd.read_parquet(output_dir / "manifests" / "xrd_samples.parquet")
            self.assertEqual(2, samples["grid_shard_path"].nunique())
            first = samples.iloc[0]
            with h5py.File(output_dir / first["grid_shard_path"], "r") as handle:
                self.assertEqual(4001, handle["two_theta"].shape[0])
                grid = handle["intensity"][int(first["grid_row_index"])]
                self.assertTrue(np.isfinite(grid).all())
                self.assertGreaterEqual(float(grid.min()), 0.0)
                self.assertAlmostEqual(100.0, float(grid.max()), places=5)
            peaks = pd.concat(
                [pd.read_parquet(output_dir / path) for path in samples["peak_shard_path"].unique()]
            )
            l12_peaks = peaks[peaks["dataset_structure_id"] == "occstr_fid_l12"]
            fcc_peaks = peaks[peaks["dataset_structure_id"] == "occstr_fid_fcc_sqs"]
            self.assertTrue(l12_peaks["reflection_class"].isin(["fundamental", "superlattice"]).all())
            self.assertTrue(fcc_peaks["reflection_class"].eq("not_applicable").all())

    def test_production_run_is_scoped_and_resumable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"

            plan = _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "run_a",
                "--subset", "all", "--config", str(config_path), "--shard-size", "2",
            )
            self.assertEqual("pass", plan["status"])
            self.assertEqual(4, plan["structure_count"])
            self.assertEqual(2, plan["shard_count"])

            incompatible = _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "run_a",
                "--subset", "all", "--config", str(config_path), "--shard-size", "4",
            )
            self.assertEqual("fail", incompatible["status"])
            self.assertEqual("run_plan_incompatible", incompatible["reason"])

            generated = _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "run_a",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("generated", generated["action"])
            run_root = output_dir / "xrd" / "runs" / "run_a"
            self.assertTrue((run_root / "grids" / "grids_0000.h5").exists())
            self.assertTrue((run_root / "completed" / "shard_0000.json").exists())

            resumed = _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "run_a",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("skipped", resumed["action"])
            self.assertFalse(any(run_root.rglob("*.tmp")))

            _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "run_b",
                "--subset", "pilot", "--config", str(config_path), "--shard-size", "4",
            )
            _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "run_b",
                "--shard-index", "0", "--resume",
            )
            self.assertTrue((output_dir / "xrd" / "runs" / "run_b" / "grids" / "grids_0000.h5").exists())
            self.assertTrue((run_root / "grids" / "grids_0000.h5").exists())

    def test_xrd_artifact_run_defers_unreviewed_semantic_labels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"

            plan_report = _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "artifact",
                "--purpose", "xrd-artifact", "--subset", "all",
                "--config", str(config_path), "--shard-size", "4",
            )

            self.assertEqual("pass", plan_report["status"])
            self.assertEqual("xrd-artifact", plan_report["purpose"])
            run_root = output_dir / "xrd" / "runs" / "artifact"
            plan = json.loads((run_root / "run_plan.json").read_text(encoding="utf-8"))
            self.assertEqual("xrd-artifact", plan["purpose"])
            self.assertEqual("deferred", plan["label_binding"])

            generated = _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "artifact",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("pass", generated["status"])
            samples = pd.read_parquet(run_root / "manifests" / "samples_0000.parquet")
            self.assertTrue(samples["structure_family"].eq("unassigned").all())
            self.assertTrue(samples["parent_lattice"].eq("unassigned").all())
            self.assertTrue(samples["benchmark_subset"].eq("unassigned").all())
            self.assertTrue(samples["split"].eq("unassigned").all())
            peaks = pd.read_parquet(run_root / "peaks" / "peaks_0000.parquet")
            self.assertTrue(peaks["reflection_class"].eq("not_assigned").all())
            self.assertFalse(peaks["is_superlattice_candidate"].any())

    def test_benchmark_release_plan_requires_completed_scientific_gate(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"

            blocked = _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "blocked_release",
                "--purpose", "benchmark-release", "--subset", "all",
                "--config", str(config_path), "--shard-size", "4",
            )

            self.assertEqual("fail", blocked["status"])
            self.assertEqual("benchmark_release_gate_failed", blocked["reason"])
            self.assertFalse(blocked["benchmark_ready"])
            self.assertFalse(
                (output_dir / "xrd" / "runs" / "blocked_release" / "run_plan.json").exists()
            )

    def test_production_shard_refuses_unmarked_final_artifacts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"
            _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "orphan",
                "--subset", "all", "--config", str(config_path), "--shard-size", "4",
            )
            orphan = output_dir / "xrd" / "runs" / "orphan" / "grids" / "grids_0000.h5"
            orphan.write_bytes(b"incomplete")

            report = _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "orphan",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("fail", report["status"])
            self.assertEqual("unmarked_shard_artifacts_exist", report["reason"])
            self.assertEqual(b"incomplete", orphan.read_bytes())

    def test_production_finalize_rejects_missing_and_corrupt_shards(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"
            _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "full",
                "--subset", "all", "--config", str(config_path), "--shard-size", "2",
            )
            _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "full",
                "--shard-index", "0", "--resume",
            )

            missing = _run_cli(
                "simulate-finalize", "--dataset", str(output_dir), "--run-id", "full"
            )
            self.assertEqual("fail", missing["status"])
            self.assertEqual([1], missing["missing_shards"])

            _run_cli(
                "simulate", "--dataset", str(output_dir), "--run-id", "full",
                "--shard-index", "1", "--resume",
            )
            grid_path = output_dir / "xrd" / "runs" / "full" / "grids" / "grids_0001.h5"
            with h5py.File(grid_path, "r+") as handle:
                handle["intensity"][0, 0] = np.nan
            corrupt = _run_cli(
                "simulate-finalize", "--dataset", str(output_dir), "--run-id", "full"
            )
            self.assertEqual("fail", corrupt["status"])
            self.assertIn(1, corrupt["invalid_shards"])

    def test_production_finalize_merges_deterministically_and_run_audit_passes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output_dir = _prepare_benchmark_dataset(root)
            config_path = Path(__file__).parents[1] / "configs" / "xrd_config_default.json"
            _run_cli(
                "simulate-plan", "--dataset", str(output_dir), "--run-id", "release",
                "--subset", "all", "--config", str(config_path), "--shard-size", "2",
            )
            for shard_index in (1, 0):
                _run_cli(
                    "simulate", "--dataset", str(output_dir), "--run-id", "release",
                    "--shard-index", str(shard_index), "--resume",
                )

            finalized = _run_cli(
                "simulate-finalize", "--dataset", str(output_dir), "--run-id", "release"
            )
            self.assertEqual("pass", finalized["status"])
            self.assertEqual(4, finalized["xrd_sample_count"])
            run_root = output_dir / "xrd" / "runs" / "release"
            samples = pd.read_parquet(run_root / "xrd_samples.parquet")
            self.assertEqual(sorted(samples["dataset_structure_id"]), list(samples["dataset_structure_id"]))
            self.assertEqual(4, samples["dataset_structure_id"].nunique())
            self.assertTrue((run_root / "checksums.sha256").exists())

            audited = _run_cli(
                "audit", "--dataset", str(output_dir), "--run-id", "release"
            )
            self.assertEqual("pass", audited["status"])
            self.assertTrue(audited["run_integrity"])
            self.assertEqual(4001, audited["points_per_pattern"])
            self.assertEqual("pass", audited["code_reproducibility"]["status"])

            from alloy_xrd_bench.occupancy_production import audit_production_run

            with patch(
                "alloy_xrd_bench.occupancy_production._code_sha256",
                return_value="f" * 64,
            ):
                historical_audit = audit_production_run(
                    dataset_dir=output_dir, run_id="release"
                )
            self.assertEqual("pass", historical_audit["status"])
            self.assertTrue(historical_audit["run_integrity"])
            self.assertEqual(
                "version_mismatch",
                historical_audit["code_reproducibility"]["status"],
            )

            (run_root / "checksums.sha256").write_text(
                "0" * 64 + "  xrd_samples.parquet\n", encoding="utf-8"
            )
            tampered_audit = _run_cli(
                "audit", "--dataset", str(output_dir), "--run-id", "release"
            )
            self.assertEqual("fail", tampered_audit["status"])
            self.assertIn("checksum_mismatch:xrd_samples.parquet", tampered_audit["issues"])

    def test_ingest_extracts_l12_occupancy_labels_from_relaxed_structure_csv(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            _write_l12_csv(csv_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.occupancy_xrd",
                    "ingest",
                    "--csv",
                    str(csv_path),
                    "--output",
                    str(output_dir),
                    "--limit",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual("", result.stderr)
            self.assertEqual(0, result.returncode)
            report = json.loads(result.stdout)
            self.assertEqual("pass", report["status"])
            self.assertEqual(1, report["structure_count"])

            structures = pd.read_parquet(output_dir / "manifests" / "structures.parquet")
            row = structures.iloc[0].to_dict()
            self.assertEqual("fid_l12", row["fid"])
            self.assertEqual("AlNi3", row["formula"])
            self.assertEqual("l12_like", row["phase_hint"])
            self.assertEqual("pass", row["quality_tier"])
            self.assertAlmostEqual(1.0, row["corner_Al_fraction"])
            self.assertAlmostEqual(1.0, row["face_Ni_fraction"])
            self.assertAlmostEqual(1.0, row["l12_order_parameter"])
            self.assertTrue((output_dir / row["artifact_path_json"]).exists())

            sites = pd.read_parquet(output_dir / "manifests" / "occupancy_sites.parquet")
            self.assertEqual({"corner", "face"}, set(sites["sublattice_label"]))
            self.assertEqual(["Al"], sorted(sites[sites["sublattice_label"] == "corner"]["element"].unique()))
            self.assertEqual(["Ni"], sorted(sites[sites["sublattice_label"] == "face"]["element"].unique()))

    def test_simulate_generates_xrd_occupancy_outputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            csv_path = root / "hea_sample.csv"
            output_dir = root / "ds_hea"
            _write_l12_csv(csv_path)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.occupancy_xrd",
                    "ingest",
                    "--csv",
                    str(csv_path),
                    "--output",
                    str(output_dir),
                    "--limit",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.occupancy_xrd",
                    "simulate",
                    "--dataset",
                    str(output_dir),
                    "--split",
                    "smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual("", result.stderr)
            self.assertEqual(0, result.returncode)
            report = json.loads(result.stdout)
            self.assertEqual("pass", report["status"])
            self.assertEqual(1, report["xrd_sample_count"])

            samples = pd.read_parquet(output_dir / "manifests" / "xrd_occupancy_samples.parquet")
            row = samples.iloc[0].to_dict()
            grid = pd.read_parquet(output_dir / row["xrd_grid_path"])
            peaks = pd.read_parquet(output_dir / row["xrd_peak_path"])
            self.assertEqual("smoke", row["split"])
            self.assertGreater(len(grid), 0)
            self.assertGreater(len(peaks), 0)
            self.assertAlmostEqual(100.0, float(grid["intensity"].max()), places=6)
            self.assertIn("is_superlattice_candidate", peaks.columns)


def _write_l12_csv(path: Path) -> None:
    structure = Structure(
        Lattice.cubic(3.57),
        ["Al", "Ni", "Ni", "Ni"],
        [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    )
    row = {
        "fid": "fid_l12",
        "reduced_formula": "AlNi3",
        "chemical_system": "Al-Ni",
        "lattice": "",
        "nelements": 2,
        "NIONS": 4,
        "space_group_number": 221,
        "volume_per_atom": structure.volume / len(structure),
        "pressure": 0.0,
        "stress": "",
        "e_per_atom": -5.0,
        "Ef_per_atom": -0.1,
        "e_above_hull": 0.0,
        "magmom": "",
        "charge": "",
        "structure_ini_as_dict": "",
        "structure_as_dict": repr(structure.as_dict()),
        "kpt": "10 10 10",
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_shifted_l12_csv(path: Path) -> None:
    structure = Structure(
        Lattice.cubic(3.57),
        ["Al", "Ni", "Ni", "Ni"],
        [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0, 0, 0)],
    )
    row = {
        "fid": "fid_l12_shifted_origin",
        "reduced_formula": "AlNi3",
        "chemical_system": "Al-Ni",
        "lattice": "",
        "nelements": 2,
        "NIONS": 4,
        "space_group_number": 221,
        "volume_per_atom": structure.volume / len(structure),
        "pressure": 0.0,
        "stress": "",
        "e_per_atom": -5.0,
        "Ef_per_atom": -0.1,
        "e_above_hull": 0.0,
        "magmom": "",
        "charge": "",
        "structure_ini_as_dict": "",
        "structure_as_dict": repr(structure.as_dict()),
        "kpt": "10 10 10",
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_transformed_parent_csv(path: Path) -> None:
    structure = Structure(
        Lattice.cubic(3.6),
        ["Al", "Ni", "Co", "Cr"],
        [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    )
    row = {
        "fid": "fid_bcc_to_fcc",
        "reduced_formula": "AlCoCrNi",
        "chemical_system": "Al-Co-Cr-Ni",
        "lattice": "bcc",
        "nelements": 4,
        "NIONS": 4,
        "space_group_number": 1,
        "volume_per_atom": structure.volume / len(structure),
        "pressure": 0.0,
        "stress": "",
        "e_per_atom": -5.0,
        "Ef_per_atom": -0.1,
        "e_above_hull": 0.0,
        "magmom": "",
        "charge": "",
        "structure_ini_as_dict": "",
        "structure_as_dict": repr(structure.as_dict()),
        "kpt": "10 10 10",
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_benchmark_csv(path: Path) -> None:
    l12 = Structure(
        Lattice.cubic(3.57),
        ["Al", "Ni", "Ni", "Ni"],
        [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    )
    b2 = Structure(Lattice.cubic(2.88), ["Al", "Ni"], [(0, 0, 0), (0.5, 0.5, 0.5)])
    fcc = Structure(
        Lattice.cubic(3.6),
        ["Al", "Ni", "Co", "Cr"],
        [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    )
    fcc.make_supercell([2, 2, 2])
    fcc.translate_sites([0], [0.003, -0.002, 0.001], frac_coords=True)
    complex_structure = Structure(
        Lattice.orthorhombic(5.5, 6.0, 6.5),
        ["Al", "Al", "Al", "Ni", "Ni", "Ni", "Ni", "Ni"],
        [
            (0.125, 0.125, 0.125), (0.125, 0.125, 0.625),
            (0.125, 0.625, 0.125), (0.125, 0.625, 0.625),
            (0.625, 0.125, 0.125), (0.625, 0.125, 0.625),
            (0.625, 0.625, 0.125), (0.625, 0.625, 0.625),
        ],
    )
    specs = [
        ("fid_l12", "AlNi3", "Al-Ni", "", 221, l12),
        ("fid_b2", "AlNi", "Al-Ni", "", 221, b2),
        ("fid_fcc_sqs", "AlCoCrNi", "Al-Co-Cr-Ni", "fcc", 1, fcc),
        ("fid_complex", "Al3Ni5", "Al-Ni", "", 1, complex_structure),
    ]
    rows = []
    for fid, formula, system, lattice, space_group, structure in specs:
        rows.append(
            {
                "fid": fid,
                "reduced_formula": formula,
                "chemical_system": system,
                "lattice": lattice,
                "nelements": len(structure.composition.elements),
                "NIONS": len(structure),
                "space_group_number": space_group,
                "volume_per_atom": structure.volume / len(structure),
                "pressure": 0.0,
                "stress": "",
                "e_per_atom": -5.0,
                "Ef_per_atom": -0.1,
                "e_above_hull": 0.0,
                "magmom": "",
                "charge": "",
                "structure_ini_as_dict": "",
                "structure_as_dict": repr(structure.as_dict()),
                "kpt": "10 10 10",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_benchmark_dataset(root: Path) -> Path:
    csv_path = root / "hea_sample.csv"
    output_dir = root / "ds_hea"
    _write_benchmark_csv(csv_path)
    _run_cli("inventory", "--csv", str(csv_path), "--output", str(output_dir))
    _run_cli("label", "--dataset", str(output_dir), "--pilot-size", "4")
    _run_cli("split", "--dataset", str(output_dir))
    return output_dir


def _run_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "alloy_xrd_bench.occupancy_xrd", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"CLI failed unexpectedly\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout)
