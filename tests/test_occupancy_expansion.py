from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from alloy_xrd_bench.occupancy_expansion import (
    adjudicate_parent_family,
    assign_group_splits,
    generate_counterfactual_variants,
    render_observation,
)
from test_occupancy_xrd_cli import _prepare_benchmark_dataset


class OccupancyExpansionPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        structure = Structure(
            Lattice.cubic(3.6),
            ["Al", "Co", "Cr", "Ni"],
            [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
        )
        structure.make_supercell([2, 2, 2])
        self.structure = structure

    def test_counterfactuals_preserve_geometry_composition_and_are_deterministic(self):
        config = {
            "strategies": [
                "local_swap", "randomized", "ordering_enhanced", "clustering_enhanced"
            ],
            "candidates_per_strategy": 4,
            "local_swap_fraction": 0.05,
            "monte_carlo_steps": 32,
        }

        first = generate_counterfactual_variants(
            self.structure,
            parent_structure_id="parent_fcc",
            parent_lattice="fcc",
            parent_hash="a" * 64,
            config=config,
        )
        second = generate_counterfactual_variants(
            self.structure,
            parent_structure_id="parent_fcc",
            parent_lattice="fcc",
            parent_hash="a" * 64,
            config=config,
        )

        self.assertEqual(4, len(first))
        self.assertEqual(
            [row["structure_sha256"] for row in first],
            [row["structure_sha256"] for row in second],
        )
        self.assertEqual(4, len({row["structure_sha256"] for row in first}))
        parent_coords = np.mod(self.structure.frac_coords, 1.0)
        parent_counts = self.structure.composition.get_el_amt_dict()
        for row in first:
            derived = Structure.from_dict(json.loads(row["structure_json"]))
            self.assertTrue(np.allclose(self.structure.lattice.matrix, derived.lattice.matrix))
            self.assertTrue(np.allclose(parent_coords, np.mod(derived.frac_coords, 1.0)))
            self.assertEqual(parent_counts, derived.composition.get_el_amt_dict())
            self.assertEqual("fixed_lattice_counterfactual", row["fidelity"])
            self.assertGreater(row["swapped_site_count"], 0)

    def test_observation_renderer_is_deterministic_and_zero_perturbation_is_clean(self):
        axis = np.linspace(20.0, 100.0, 4001)
        peaks = pd.DataFrame(
            [
                {"two_theta": 44.5, "intensity": 100.0, "hkls_json": '[{"hkl": [1, 1, 1], "multiplicity": 8}]'},
                {"two_theta": 51.8, "intensity": 45.0, "hkls_json": '[{"hkl": [2, 0, 0], "multiplicity": 6}]'},
            ]
        )
        clean_params = {
            "zero_shift_deg": 0.0,
            "instrument_fwhm_deg": 0.1,
            "microstrain_fraction": 0.0,
            "crystallite_size_nm": None,
            "background_fraction": 0.0,
            "background_coefficients": [0.0, 0.0, 0.0],
            "gaussian_noise_fraction": 0.0,
            "poisson_scale": 0.0,
            "march_dollase_r": 1.0,
            "texture_axis": [1, 0, 0],
            "seed": 7,
        }

        clean = render_observation(axis, peaks, clean_params)
        repeated = render_observation(axis, peaks, clean_params)

        self.assertTrue(np.array_equal(clean, repeated))
        self.assertEqual((4001,), clean.shape)
        self.assertGreaterEqual(float(clean.min()), 0.0)
        self.assertAlmostEqual(100.0, float(clean.max()), places=5)

    def test_observation_renderer_accepts_hexagonal_four_index_reflections(self):
        axis = np.linspace(20.0, 100.0, 4001)
        peaks = pd.DataFrame(
            [{"two_theta": 47.5, "intensity": 100.0, "hkls_json": '[{"hkl": [1, 0, -1, 1], "multiplicity": 12}]'}]
        )
        params = {
            "instrument_fwhm_deg": 0.1,
            "march_dollase_r": 0.8,
            "texture_axis": [1, 0, 0],
            "seed": 3,
        }

        values = render_observation(axis, peaks, params)

        self.assertEqual((4001,), values.shape)
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(100.0, float(values.max()), places=5)

    def test_group_split_assignment_keeps_parent_and_variants_together(self):
        states = pd.DataFrame(
            [
                {"structure_state_id": "parent_a", "parent_group_id": "parent_a"},
                {"structure_state_id": "cf_a_1", "parent_group_id": "parent_a"},
                {"structure_state_id": "cf_a_2", "parent_group_id": "parent_a"},
                {"structure_state_id": "parent_b", "parent_group_id": "parent_b"},
            ]
        )

        assigned = assign_group_splits(states)

        self.assertEqual(2, assigned["parent_group_id"].nunique())
        self.assertEqual(1, assigned[assigned["parent_group_id"] == "parent_a"]["id_split"].nunique())
        self.assertEqual(
            assigned.set_index("structure_state_id").loc["parent_a", "id_split"],
            assigned.set_index("structure_state_id").loc["cf_a_1", "id_split"],
        )

    def test_ptm_adjudication_rule_uses_frozen_fraction_and_rmsd_thresholds(self):
        self.assertEqual("fcc_like", adjudicate_parent_family("fcc_like", 0.50, 0.20))
        self.assertEqual("bcc_like", adjudicate_parent_family("bcc_like", 0.75, 0.10))
        self.assertEqual("complex", adjudicate_parent_family("hcp_like", 0.49, 0.10))
        self.assertEqual("complex", adjudicate_parent_family("fcc_like", 0.80, 0.201))
        self.assertEqual("complex", adjudicate_parent_family("unknown", 0.90, 0.01))


class OccupancyExpansionCliTests(unittest.TestCase):
    def test_counterfactual_observation_and_benchmark_workflow(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset = _prepare_benchmark_dataset(root)
            project = Path(__file__).parents[1]
            xrd_config = project / "configs" / "xrd_config_default.json"
            cf_config = root / "counterfactual.json"
            cf_config.write_text(
                json.dumps(
                    {
                        "config_id": "counterfactual_test_v1",
                        "strategies": [
                            "local_swap", "randomized", "ordering_enhanced", "clustering_enhanced"
                        ],
                        "candidates_per_strategy": 4,
                        "local_swap_fraction": 0.05,
                        "monte_carlo_steps": 32,
                        "min_elements": 3,
                        "min_natoms_for_sro": 27,
                        "max_template_mapping_rms": 0.25,
                        "xrd_config": json.loads(xrd_config.read_text(encoding="utf-8")),
                    }
                ),
                encoding="utf-8",
            )

            blocked_full = _run_expansion_cli(
                "counterfactual-plan", "--dataset", str(dataset), "--run-id", "cf_full",
                "--parents", "1001", "--variants-per-parent", "4", "--config", str(cf_config),
            )
            self.assertEqual("fail", blocked_full["status"])
            self.assertEqual("benchmark_release_gate_failed", blocked_full["reason"])

            planned = _run_expansion_cli(
                "counterfactual-plan", "--dataset", str(dataset), "--run-id", "cf_test",
                "--parents", "1", "--variants-per-parent", "4", "--config", str(cf_config),
                "--shard-size", "1",
            )
            self.assertEqual("pass", planned["status"])
            self.assertEqual(1, planned["parent_count"])
            self.assertEqual(4, planned["expected_variant_count"])
            repeated_plan = _run_expansion_cli(
                "counterfactual-plan", "--dataset", str(dataset), "--run-id", "cf_test",
                "--parents", "1", "--variants-per-parent", "4", "--config", str(cf_config),
                "--shard-size", "1",
            )
            self.assertEqual("existing", repeated_plan["action"])

            generated = _run_expansion_cli(
                "counterfactual-generate", "--dataset", str(dataset), "--run-id", "cf_test",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("generated", generated["action"])
            resumed = _run_expansion_cli(
                "counterfactual-generate", "--dataset", str(dataset), "--run-id", "cf_test",
                "--shard-index", "0", "--resume",
            )
            self.assertEqual("skipped", resumed["action"])

            finalized = _run_expansion_cli(
                "counterfactual-finalize", "--dataset", str(dataset), "--run-id", "cf_test"
            )
            self.assertEqual("pass", finalized["status"])
            self.assertEqual(4, finalized["counterfactual_count"])
            cf_root = dataset / "derived" / "occupancy" / "cf_test"
            derived = pd.read_parquet(cf_root / "structures.parquet")
            self.assertEqual(4, derived["structure_sha256"].nunique())
            xrd_root = dataset / "xrd" / "runs" / "cf_test_xrd"
            samples = pd.read_parquet(xrd_root / "xrd_samples.parquet")
            with __import__("h5py").File(dataset / samples.iloc[0]["grid_shard_path"], "r") as handle:
                self.assertEqual((4, 4001), handle["intensity"].shape)

            canonical_plan = _run_expansion_cli(
                "simulate-plan", "--dataset", str(dataset), "--run-id", "canonical",
                "--subset", "all", "--config", str(xrd_config), "--shard-size", "4",
            )
            self.assertEqual("pass", canonical_plan["status"])
            _run_expansion_cli(
                "simulate", "--dataset", str(dataset), "--run-id", "canonical",
                "--shard-index", "0", "--resume",
            )
            _run_expansion_cli(
                "simulate-finalize", "--dataset", str(dataset), "--run-id", "canonical"
            )

            benchmark = _run_expansion_cli(
                "benchmark-build", "--dataset", str(dataset),
                "--counterfactual-run", "cf_test", "--canonical-run", "canonical",
            )
            self.assertEqual("pass", benchmark["status"])
            pairs = pd.read_parquet(dataset / "manifests" / "occupancy_pairs.parquet")
            splits = pd.read_parquet(dataset / "manifests" / "split_assignments.parquet")
            composition_features = pd.read_parquet(
                dataset / "manifests" / "composition_only_features.parquet"
            )
            self.assertEqual(4, len(pairs))
            self.assertIn("fraction_Al", composition_features.columns)
            self.assertEqual(benchmark["structure_state_count"], len(composition_features))
            self.assertEqual(1, splits[splits["parent_group_id"] == derived.iloc[0]["parent_group_id"]]["id_split"].nunique())

            perturbation = root / "perturbation.json"
            perturbation.write_text(
                json.dumps(
                    {
                        "config_id": "perturbation_test_v1",
                        "zero_shift_deg": [-0.1, 0.1],
                        "instrument_fwhm_deg": [0.05, 0.3],
                        "microstrain_fraction": [0.0, 0.003],
                        "crystallite_size_nm": [20.0, 200.0],
                        "background_fraction": [0.0, 0.1],
                        "gaussian_noise_fraction": [0.001, 0.02],
                        "poisson_scale": [10.0, 100.0],
                        "march_dollase_r": [0.7, 1.3],
                        "texture_axes": [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                    }
                ),
                encoding="utf-8",
            )
            frozen = _run_expansion_cli(
                "observations-freeze", "--dataset", str(dataset), "--run-id", "obs_test",
                "--config", str(perturbation), "--variants", "2",
                "--splits", "train,validation,test",
            )
            self.assertEqual("pass", frozen["status"])
            task_samples = pd.read_parquet(dataset / "manifests" / "task_samples.parquet")
            unique_states = task_samples["structure_state_id"].nunique()
            self.assertEqual(unique_states * 2, frozen["observation_count"])
            observations = pd.read_parquet(
                dataset / "xrd" / "runs" / "obs_test" / "xrd_samples.parquet"
            )
            self.assertTrue(observations["perturbation_json"].str.len().gt(2).all())

            observation_root = dataset / "xrd" / "runs" / "obs_test"
            last_shard = frozen["shard_count"] - 1
            for path in (
                observation_root / "run_summary.json",
                observation_root / "checksums.sha256",
                observation_root / "xrd_samples.parquet",
                observation_root / "completed" / f"shard_{last_shard:04d}.json",
                observation_root / "grids" / f"grids_{last_shard:04d}.h5",
                observation_root / "manifests" / f"samples_{last_shard:04d}.parquet",
            ):
                path.unlink()
            resumed_observations = _run_expansion_cli(
                "observations-freeze", "--dataset", str(dataset), "--run-id", "obs_test",
                "--config", str(perturbation), "--variants", "2",
                "--splits", "train,validation,test",
            )
            self.assertEqual("pass", resumed_observations["status"])
            self.assertEqual(frozen["observation_count"], resumed_observations["observation_count"])
            observation_audit = _run_expansion_cli(
                "observations-audit", "--dataset", str(dataset), "--run-id", "obs_test"
            )
            self.assertEqual("pass", observation_audit["status"])
            self.assertEqual(frozen["observation_count"], observation_audit["observation_count"])

            selected_parent = pairs[["parent_group_id"]].drop_duplicates().head(1)
            parent_filter = root / "occupancy_parents.parquet"
            selected_parent.to_parquet(parent_filter, index=False)
            v1_1 = root / "benchmark_v1_1"
            filtered = _run_expansion_cli(
                "observations-freeze", "--dataset", str(dataset),
                "--output-dataset", str(v1_1), "--run-id", "obs_occupancy_v1_1",
                "--parent-groups", str(parent_filter), "--config", str(perturbation),
                "--variants", "2", "--splits", "train,validation,test",
            )
            expected_states = task_samples.loc[
                task_samples["parent_group_id"].isin(selected_parent["parent_group_id"]),
                "structure_state_id",
            ].nunique()
            self.assertEqual(expected_states * 2, filtered["observation_count"])
            self.assertFalse((dataset / "xrd" / "runs" / "obs_occupancy_v1_1").exists())
            self.assertTrue(
                (v1_1 / "xrd" / "runs" / "obs_occupancy_v1_1" / "xrd_samples.parquet").exists()
            )


def _run_expansion_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "alloy_xrd_bench.occupancy_xrd", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"CLI failed unexpectedly\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout)


if __name__ == "__main__":
    unittest.main()
