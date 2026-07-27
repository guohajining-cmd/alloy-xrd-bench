from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from alloy_xrd_bench.dataset_paper_visuals import (
    _format_gamma_component_annotation,
    generate_dataset_paper_visuals,
    reconstruct_gamma_components,
    project_gamma_components_from_full_grid,
    select_gamma_examples,
    select_hea_pair,
    select_hea_pair_lazy,
)


class DatasetPaperVisualSelectionTests(unittest.TestCase):
    def test_gamma_annotation_identifies_phase_and_hkl_group(self):
        annotation = _format_gamma_component_annotation({
            "phase": "gamma_prime",
            "h": 2,
            "k": 2,
            "l": 0,
            "hkl_group_id": "gamma_prime:220",
            "center_deg": 51.25,
            "fwhm_deg": 0.18,
            "integrated_intensity": 3.4,
        })

        self.assertIn("gamma-prime (220)", annotation)
        self.assertIn("group=gamma_prime:220", annotation)

    def test_hea_pair_is_train_only_composition_matched_and_deterministic(self):
        candidates = pd.DataFrame([
            {
                "sample_id": "sample-a",
                "structure_state_id": "state-a",
                "split": "train",
                "composition_key": "Co1Cr1Ni1",
                "parent_topology": "fcc",
                "sro_vector": [0.0, 0.0],
            },
            {
                "sample_id": "sample-b",
                "structure_state_id": "state-b",
                "split": "train",
                "composition_key": "Co1Cr1Ni1",
                "parent_topology": "fcc",
                "sro_vector": [0.8, -0.6],
            },
            {
                "sample_id": "sample-c",
                "structure_state_id": "state-c",
                "split": "test",
                "composition_key": "Co1Cr1Ni1",
                "parent_topology": "fcc",
                "sro_vector": [1.0, 1.0],
            },
            {
                "sample_id": "sample-d",
                "structure_state_id": "state-d",
                "split": "train",
                "composition_key": "Al1Ni3",
                "parent_topology": "fcc",
                "sro_vector": [1.0, 1.0],
            },
        ])
        spectra = {
            "sample-a": np.array([1.0, 0.1, 0.0]),
            "sample-b": np.array([1.0, 0.099, 0.001]),
            "sample-c": np.array([1.0, 0.1, 0.0]),
            "sample-d": np.array([1.0, 0.1, 0.0]),
        }

        selected = select_hea_pair(candidates, spectra, min_cosine_similarity=0.995)

        self.assertEqual("sample-a", selected["left_sample_id"])
        self.assertEqual("sample-b", selected["right_sample_id"])
        self.assertGreaterEqual(selected["cosine_similarity"], 0.995)
        self.assertAlmostEqual(1.4, selected["sro_l1_distance"])
        self.assertEqual("train", selected["split"])

    def test_hea_pair_rejects_when_no_pair_meets_similarity_threshold(self):
        candidates = pd.DataFrame([
            {
                "sample_id": "sample-a", "structure_state_id": "state-a",
                "split": "train", "composition_key": "Co1Cr1Ni1",
                "parent_topology": "fcc", "sro_vector": [0.0],
            },
            {
                "sample_id": "sample-b", "structure_state_id": "state-b",
                "split": "train", "composition_key": "Co1Cr1Ni1",
                "parent_topology": "fcc", "sro_vector": [1.0],
            },
        ])
        spectra = {
            "sample-a": np.array([1.0, 0.0]),
            "sample-b": np.array([0.0, 1.0]),
        }

        with self.assertRaisesRegex(ValueError, "cosine similarity"):
            select_hea_pair(candidates, spectra, min_cosine_similarity=0.995)

    def test_lazy_hea_selection_checks_candidates_in_sro_distance_order(self):
        candidates = pd.DataFrame([
            {"sample_id": "a", "structure_state_id": "sa", "split": "train", "composition_key": "x", "parent_topology": "fcc", "sro_vector": [0.0]},
            {"sample_id": "b", "structure_state_id": "sb", "split": "train", "composition_key": "x", "parent_topology": "fcc", "sro_vector": [1.0]},
            {"sample_id": "c", "structure_state_id": "sc", "split": "train", "composition_key": "x", "parent_topology": "fcc", "sro_vector": [0.8]},
        ])
        spectra = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 0.001]),
        }
        loaded: list[str] = []

        def load(sample_id: str) -> np.ndarray:
            loaded.append(sample_id)
            return spectra[sample_id]

        selected = select_hea_pair_lazy(candidates, load)

        self.assertEqual(("a", "c"), (selected["left_sample_id"], selected["right_sample_id"]))
        self.assertEqual(0.8, selected["sro_l1_distance"])
        self.assertEqual(["a", "b", "a", "c"], loaded)

    def test_gamma_examples_are_training_median_representatives(self):
        candidates = pd.DataFrame([
            {"sample_id": "resolved-low", "split": "train", "difficulty": "resolved", "component_count": 1, "snr": 10.0, "phase_weight": 0.1},
            {"sample_id": "resolved-mid", "split": "train", "difficulty": "resolved", "component_count": 2, "snr": 20.0, "phase_weight": 0.5},
            {"sample_id": "resolved-high", "split": "train", "difficulty": "resolved", "component_count": 3, "snr": 30.0, "phase_weight": 0.9},
            {"sample_id": "severe-low", "split": "train", "difficulty": "severe", "component_count": 2, "scored_component_count": 2, "snr": 8.0, "phase_weight": 0.2},
            {"sample_id": "severe-mid", "split": "train", "difficulty": "severe", "component_count": 3, "scored_component_count": 2, "snr": 16.0, "phase_weight": 0.5},
            {"sample_id": "severe-high", "split": "train", "difficulty": "severe", "component_count": 4, "scored_component_count": 2, "snr": 24.0, "phase_weight": 0.8},
            {"sample_id": "severe-context-only", "split": "train", "difficulty": "severe", "component_count": 3, "scored_component_count": 0, "snr": 16.0, "phase_weight": 0.5},
            {"sample_id": "test-perfect", "split": "test", "difficulty": "severe", "component_count": 3, "snr": 16.0, "phase_weight": 0.5},
        ])

        selected = select_gamma_examples(candidates)

        self.assertEqual("resolved-mid", selected["resolved"]["sample_id"])
        self.assertEqual("severe-mid", selected["severe"]["sample_id"])
        self.assertEqual("train", selected["resolved"]["split"])
        self.assertEqual("train", selected["severe"]["split"])

    def test_gamma_component_profiles_reconstruct_clean_pattern(self):
        grid = np.linspace(40.0, 50.0, 1001)
        components = pd.DataFrame([
            {"center_deg": 44.0, "fwhm_deg": 0.2, "integrated_intensity": 4.0, "eta": 0.4},
            {"center_deg": 44.3, "fwhm_deg": 0.3, "integrated_intensity": 2.0, "eta": 0.6},
        ])

        total, curves = reconstruct_gamma_components(grid, components)

        self.assertEqual((2, 1001), curves.shape)
        np.testing.assert_allclose(total, curves.sum(axis=0), rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(6.0, float(np.trapezoid(total, grid)), places=10)

    def test_gamma_window_projection_preserves_full_grid_normalization(self):
        full_grid = np.linspace(20.0, 100.0, 4001)
        window_grid = full_grid[1200:1351]
        components = pd.DataFrame([
            {"center_deg": 44.0, "fwhm_deg": 0.2, "integrated_intensity": 4.0, "eta": 0.4},
            {"center_deg": 44.3, "fwhm_deg": 0.3, "integrated_intensity": 2.0, "eta": 0.6},
        ])
        full_total, full_curves = reconstruct_gamma_components(full_grid, components)

        projected_total, projected_curves = project_gamma_components_from_full_grid(
            window_grid, full_grid, components
        )

        np.testing.assert_allclose(projected_total, full_total[1200:1351], atol=1e-12)
        np.testing.assert_allclose(projected_curves, full_curves[:, 1200:1351], atol=1e-12)


class DatasetPaperVisualIntegrationTests(unittest.TestCase):
    def test_generator_writes_traceable_figures_examples_manifest_and_table(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            hea = root / "hea"
            gamma = root / "gamma"
            self._write_hea_release(hea)
            self._write_gamma_release(gamma)
            hea_results, gamma_results = self._write_results(root)

            result = generate_dataset_paper_visuals(
                hea,
                gamma,
                root / "paper",
                seed=2027,
                hea_results_path=hea_results,
                gamma_results_path=gamma_results,
            )

            expected = {
                "dataset_overview", "hea_sro_example", "gamma_decomp_example",
                "dataset_coverage", "examples", "manifest", "compact_results",
            }
            self.assertEqual(expected, set(result))
            for key in expected:
                self.assertGreater(Path(result[key]).stat().st_size, 20, key)
            examples = json.loads(Path(result["examples"]).read_text(encoding="utf-8"))
            self.assertEqual("sample-a", examples["hea_sro"]["left_sample_id"])
            self.assertEqual("sample-b", examples["hea_sro"]["right_sample_id"])
            self.assertEqual("source_relaxed_dft", examples["hea_sro"]["candidate_fidelity"])
            self.assertEqual("resolved-mid", examples["gamma_decomp"]["resolved"]["sample_id"])
            self.assertEqual("severe-mid", examples["gamma_decomp"]["severe"]["sample_id"])
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual("sitexrd-dataset-paper-visuals-v1.1", manifest["schema_version"])
            self.assertEqual(2027, manifest["seed"])
            self.assertEqual(0.995, manifest["configuration"]["hea_min_cosine_similarity"])
            self.assertEqual("[001]", manifest["configuration"]["structure_projection"])
            self.assertEqual(
                hashlib.sha256(
                    Path("alloy_xrd_bench/dataset_paper_visuals.py").read_bytes()
                ).hexdigest(),
                manifest["script_sha256"],
            )
            source_names = {Path(record["path"]).name for record in manifest["sources"]}
            self.assertIn("grid.h5", source_names)
            self.assertTrue({"resolved-mid.npz", "severe-mid.npz"}.issubset(source_names))
            for record in manifest["sources"]:
                source = Path(record["path"])
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record["sha256"])
            for record in manifest["artifacts"]:
                artifact = Path(result[record["key"]])
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), record["sha256"])
            table = Path(result["compact_results"]).read_text(encoding="utf-8")
            self.assertIn("Spectrum Transformer", table)
            self.assertIn("Peak Feature", table)
            self.assertNotIn("nan", table.lower())

    def test_generator_rejects_missing_formal_hea_manifests(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            gamma = root / "gamma"
            self._write_gamma_release(gamma)
            hea_results, gamma_results = self._write_results(root)

            with self.assertRaisesRegex(FileNotFoundError, "HEA manifest"):
                generate_dataset_paper_visuals(
                    root / "missing-hea",
                    gamma,
                    root / "paper",
                    hea_results_path=hea_results,
                    gamma_results_path=gamma_results,
                )

    def test_generator_reads_formal_hea_structure_payload_table_by_source_row(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            hea = root / "benchmark_v1_1"
            gamma = root / "gamma"
            self._write_hea_release(hea)
            self._write_gamma_release(gamma)
            hea_results, gamma_results = self._write_results(root)

            structures = pd.read_parquet(hea / "manifests/structures.parquet")
            structures = structures.drop(columns=["artifact_path_json"])
            structures.to_parquet(hea / "manifests/structures.parquet", index=False)
            payload_path = (
                root
                / "benchmark_v1/xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet"
            )
            payload_path.parent.mkdir(parents=True)
            payload_rows = []
            for source_row, state_id in enumerate(("state-a", "state-b")):
                structure_doc = json.loads(
                    (hea / f"structures/{state_id}.json").read_text(encoding="utf-8")
                )
                payload_rows.append(
                    {"source_row": source_row, "structure_as_dict": repr(structure_doc)}
                )
            pd.DataFrame(payload_rows).to_parquet(payload_path, index=False)
            source_grid = root / "benchmark_v1/grids/grid.h5"
            source_grid.parent.mkdir(parents=True)
            (hea / "grids/grid.h5").replace(source_grid)
            release = hea / "release"
            release.mkdir()
            (release / "source_provenance.json").write_text(
                json.dumps({"source_dataset_root": "/home/GJM/data/hea/benchmark_v1"}),
                encoding="utf-8",
            )

            result = generate_dataset_paper_visuals(
                hea,
                gamma,
                root / "paper",
                hea_results_path=hea_results,
                gamma_results_path=gamma_results,
            )

            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            source_names = {Path(record["path"]).name for record in manifest["sources"]}
            self.assertIn("structure_payloads.parquet", source_names)

    def test_generator_is_byte_stable_for_same_inputs_and_seed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            hea = root / "hea"
            gamma = root / "gamma"
            output = root / "paper"
            self._write_hea_release(hea)
            self._write_gamma_release(gamma)
            hea_results, gamma_results = self._write_results(root)

            first = generate_dataset_paper_visuals(
                hea,
                gamma,
                output,
                seed=2027,
                hea_results_path=hea_results,
                gamma_results_path=gamma_results,
            )
            first_hashes = {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in first.items()
            }
            second = generate_dataset_paper_visuals(
                hea,
                gamma,
                output,
                seed=2027,
                hea_results_path=hea_results,
                gamma_results_path=gamma_results,
            )
            second_hashes = {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in second.items()
            }

            self.assertEqual(first_hashes, second_hashes)

    def _write_hea_release(self, root: Path) -> None:
        manifests = root / "manifests"
        structures_dir = root / "structures"
        grids = root / "grids"
        manifests.mkdir(parents=True)
        structures_dir.mkdir()
        grids.mkdir()
        structure_a = Structure(
            Lattice.cubic(3.6), ["Co", "Cr", "Ni", "Ni"],
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        structure_b = Structure(
            Lattice.cubic(3.6), ["Ni", "Cr", "Co", "Ni"],
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        for name, structure in (("state-a", structure_a), ("state-b", structure_b)):
            (structures_dir / f"{name}.json").write_text(
                json.dumps(structure.as_dict()), encoding="utf-8"
            )
        grid = np.linspace(20.0, 100.0, 4001, dtype=np.float32)
        spectrum_a = np.exp(-0.5 * ((grid - 44.0) / 0.2) ** 2)
        spectrum_b = spectrum_a + 0.001 * np.exp(-0.5 * ((grid - 46.0) / 0.2) ** 2)
        with h5py.File(grids / "grid.h5", "w") as handle:
            handle.create_dataset("two_theta", data=grid)
            handle.create_dataset("intensity", data=np.stack([spectrum_a, spectrum_b]))
            handle.create_dataset("xrd_sample_id", data=np.asarray([b"sample-a", b"sample-b"]))
        rows = []
        for index, name in enumerate(("a", "b")):
            rows.append({
                "structure_state_id": f"state-{name}",
                "parent_group_id": f"parent-{name}",
                "structure_family": "fcc_like",
                "fidelity": "source_relaxed_dft",
                "xrd_sample_id": f"sample-{name}",
                "grid_shard_path": "grids/grid.h5",
                "grid_row_index": index,
                "sro_shell1_json": json.dumps({"Co->Cr": 0.0 if name == "a" else 0.8}),
                "sro_shell2_json": json.dumps({"Co->Cr": 0.0 if name == "a" else -0.6}),
                "id_split": "train",
                "composition_ood_split": "train",
                "chemistry_ood_split": "train",
                "sro_ood_split": "train",
            })
        pd.DataFrame(rows).to_parquet(manifests / "task_samples.parquet", index=False)
        composition_rows = []
        for name in ("a", "b"):
            row = {"structure_state_id": f"state-{name}", "parent_group_id": f"parent-{name}"}
            row.update({f"fraction_{element}": 0.0 for element in ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"]})
            row.update({"fraction_Co": 0.25, "fraction_Cr": 0.25, "fraction_Ni": 0.5})
            composition_rows.append(row)
        pd.DataFrame(composition_rows).to_parquet(
            manifests / "composition_only_features.parquet", index=False
        )
        pd.DataFrame([
            {"dataset_structure_id": "state-a", "source_row": 0, "artifact_path_json": "structures/state-a.json"},
            {"dataset_structure_id": "state-b", "source_row": 1, "artifact_path_json": "structures/state-b.json"},
        ]).to_parquet(manifests / "structures.parquet", index=False)
        pd.DataFrame({"identifiability_ratio": [0.1, 0.2, 0.4, 0.8]}).to_parquet(
            manifests / "occupancy_pairs_calibrated.parquet", index=False
        )

    def _write_gamma_release(self, root: Path) -> None:
        core = root / "gamma-decomp-core-v1.1"
        provenance = root / "gamma-decomp-provenance-v1.1"
        for directory in (
            core / "full/patterns", core / "window/patterns", core / "slices",
            provenance / "structures/gamma", provenance / "structures/gamma_prime",
        ):
            directory.mkdir(parents=True)
        gamma_structure = Structure(
            Lattice.cubic(3.58), ["Ni", "Ni", "Ni", "Ni"],
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        gamma_prime = Structure(
            Lattice.cubic(3.57), ["Al", "Ni", "Ni", "Ni"],
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        for phase, structure in (("gamma", gamma_structure), ("gamma_prime", gamma_prime)):
            path = provenance / f"structures/{phase}/{phase}.json"
            path.write_text(json.dumps(structure.as_dict()), encoding="utf-8")
        pd.DataFrame([
            {"parent_structure_id": "gamma", "phase_role": "gamma", "artifact_path": "structures/gamma/gamma.json", "eligibility": "accepted_core", "composition": {"Ni": 4}},
            {"parent_structure_id": "gamma_prime", "phase_role": "gamma_prime", "artifact_path": "structures/gamma_prime/gamma_prime.json", "eligibility": "accepted_core", "composition": {"Al": 1, "Ni": 3}},
        ]).to_parquet(provenance / "parent_structures.parquet", index=False)
        pd.DataFrame([
            {"parent_pair_id": "pair-0", "gamma_parent_id": "gamma", "gamma_prime_parent_id": "gamma_prime", "split": "train"}
        ]).to_parquet(provenance / "parent_pairs.parquet", index=False)
        grid = np.linspace(42.0, 46.0, 151)
        sample_rows = []
        component_rows = []
        slice_rows = []
        group_rows = []
        specs = [
            ("resolved-low", "resolved", 1, 10.0, 0.1),
            ("resolved-mid", "resolved", 2, 20.0, 0.5),
            ("resolved-high", "resolved", 3, 30.0, 0.9),
            ("severe-low", "severe", 2, 8.0, 0.2),
            ("severe-mid", "severe", 3, 16.0, 0.5),
            ("severe-high", "severe", 4, 24.0, 0.8),
        ]
        for sample_id, difficulty, count, snr, phase_weight in specs:
            rows = []
            for index in range(count):
                phase = "gamma_prime" if index < round(count * phase_weight) else "gamma"
                rows.append({
                    "sample_id": sample_id, "component_id": f"{sample_id}-{index}",
                    "phase": phase, "h": 1 + index, "k": 1, "l": 1,
                    "hkl_group_id": f"{phase}:{1 + index}11",
                    "center_deg": 43.5 + index * (0.5 if difficulty == "resolved" else 0.08),
                    "fwhm_deg": 0.2, "integrated_intensity": 1.0,
                    "eta": 0.5, "score_eligible": True, "split": "train",
                })
            components = pd.DataFrame(rows)
            clean, _ = reconstruct_gamma_components(grid, components)
            noise_scale = float(clean.max()) / snr
            observed = clean + noise_scale * np.sin(np.linspace(0, 3 * np.pi, len(grid)))
            np.savez_compressed(core / f"window/patterns/{sample_id}.npz", two_theta=grid, clean=clean, observed=observed)
            sample_rows.append({"sample_id": sample_id, "task_id": "gamma_window_decomp_v1", "split": "train", "point_count": 151, "pattern_path": f"window/patterns/{sample_id}.npz", "artifact_sha256": "fixture"})
            component_rows.extend(rows)
            slice_rows.append({"sample_id": sample_id, "split": "train", "difficulty": difficulty, "minimum_separation_ratio": 4.0 if difficulty == "resolved" else 0.4, "scored_component_count": count, "context_component_ids_json": "[]"})
            group_rows.append({"sample_id": sample_id, "parent_pair_id": "pair-0", "split": "train"})
        pd.DataFrame(sample_rows).to_parquet(core / "window/samples.parquet", index=False)
        pd.DataFrame(component_rows).to_parquet(core / "window/components.parquet", index=False)
        pd.DataFrame(slice_rows).to_parquet(core / "slices/overlap_slices.parquet", index=False)
        pd.DataFrame(group_rows).to_parquet(core / "window/evaluation_groups.parquet", index=False)
        full_grid = np.linspace(20.0, 100.0, 4001)
        full_components = pd.DataFrame([
            {"sample_id": "full-0", "component_id": "full-0-0", "phase": "gamma", "h": 1, "k": 1, "l": 1, "hkl_group_id": "gamma:111", "center_deg": 44.0, "fwhm_deg": 0.2, "integrated_intensity": 4.0, "eta": 0.5, "score_eligible": True, "split": "train"},
            {"sample_id": "full-0", "component_id": "full-0-1", "phase": "gamma_prime", "h": 1, "k": 0, "l": 0, "hkl_group_id": "gamma_prime:100", "center_deg": 45.0, "fwhm_deg": 0.2, "integrated_intensity": 1.0, "eta": 0.5, "score_eligible": True, "split": "train"},
        ])
        full_clean, _ = reconstruct_gamma_components(full_grid, full_components)
        np.savez_compressed(core / "full/patterns/full-0.npz", two_theta=full_grid, clean=full_clean, observed=full_clean)
        pd.DataFrame([{"sample_id": "full-0", "task_id": "gamma_full_decomp_v1", "split": "train", "point_count": 4001, "pattern_path": "full/patterns/full-0.npz", "artifact_sha256": "fixture"}]).to_parquet(core / "full/samples.parquet", index=False)
        full_components.to_parquet(core / "full/components.parquet", index=False)
        pd.DataFrame([{"sample_id": "full-0", "parent_pair_id": "pair-0", "split": "train"}]).to_parquet(core / "full/evaluation_groups.parquet", index=False)

    def _write_results(self, root: Path) -> tuple[Path, Path]:
        hea = root / "hea_results.csv"
        gamma = root / "gamma_results.csv"
        pd.DataFrame([
            {"split_column": "id_split", "model": "composition-ridge", "parent_macro_mae_mean": 0.45, "random_seed_std": 0.0},
            {"split_column": "id_split", "model": "transformer", "parent_macro_mae_mean": 0.34, "random_seed_std": 0.01},
        ]).to_csv(hea, index=False)
        pd.DataFrame([
            {"task": "full", "model": "peak-feature", "model_label": "Peak Feature", "component_f1_mean": 0.68, "component_f1_std": 0.01, "component_f1_count": 3},
        ]).to_csv(gamma, index=False)
        return hea, gamma


if __name__ == "__main__":
    unittest.main()
