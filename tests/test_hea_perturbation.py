import json
from pathlib import Path
import tempfile
import unittest

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

from alloy_xrd_bench.hea_perturbation import (
    aggregate_perturbation_reports,
    evaluate_perturbed_checkpoint,
)
from alloy_xrd_bench.hea_validation_models import (
    PerTargetRidge,
    PhysicsConstrainedTransformer,
    ResNet1D,
    peak_features,
)


class HeaPerturbationTests(unittest.TestCase):
    def test_physics_checkpoint_loader_restores_forward_model(self):
        from alloy_xrd_bench.hea_perturbation import _model_from_checkpoint

        source = PhysicsConstrainedTransformer(4, peak_count=3)
        restored = _model_from_checkpoint(
            {
                "model": "physics-transformer",
                "peak_count": 3,
                "state_dict": source.state_dict(),
            },
            4,
        )

        with torch.no_grad():
            result = restored(torch.zeros(1, 1, 4001))
        self.assertEqual((1, 4), tuple(result["sro"].shape))

    def test_observations_may_live_in_v1_1_while_truth_remains_in_v1(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "benchmark_v1"
            observations_root = root / "benchmark_v1_1"
            run = observations_root / "xrd" / "runs" / "observations"
            (source / "manifests").mkdir(parents=True)
            (run / "grids").mkdir(parents=True)
            targets = json.loads(Path("configs/hea_validation_v1.json").read_text())["sro_targets"]
            pd.DataFrame([{"structure_state_id": "state-0", "parent_group_id": "parent-0", "id_split": "test", "sro_shell1_json": json.dumps({"Al->Al": 0.0}), "sro_shell2_json": "{}"}]).to_parquet(source / "manifests" / "task_samples.parquet")
            pd.DataFrame([{"xrd_sample_id": "obs-0", "structure_state_id": "state-0", "parent_group_id": "parent-0", "grid_shard_path": "xrd/runs/observations/grids/grid.h5", "grid_row_index": 0}]).to_parquet(run / "xrd_samples.parquet")
            with h5py.File(run / "grids" / "grid.h5", "w") as handle:
                handle.create_dataset("intensity", data=np.ones((1, 4001), dtype=np.float32))
                handle.create_dataset("xrd_sample_id", data=np.asarray([b"obs-0"]))
            model = ResNet1D(len(targets))
            checkpoint = root / "model.pt"
            torch.save({"model": "resnet1d", "seed": 2027, "split_column": "id_split", "state_dict": model.state_dict()}, checkpoint)

            report = evaluate_perturbed_checkpoint(
                source,
                "observations",
                checkpoint,
                Path("configs/hea_validation_v1.json"),
                root / "result",
                observation_dataset=observations_root,
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(str(observations_root.resolve()), report["observation_dataset"])

    def test_aggregate_requires_three_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for seed, mae in [(2027, 0.3), (2028, 0.4), (2029, 0.5)]:
                path = root / str(seed)
                path.mkdir()
                (path / "perturbation_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "evaluation_scope": "held_out_test",
                            "clean_reference_scope": "identical_structure_states",
                            "seed": seed,
                            "perturbed_parent_macro_mae": mae,
                            "clean_to_perturbed_retention": 0.8,
                        }
                    )
                )
            report = aggregate_perturbation_reports(root)
            self.assertEqual("pass", report["status"])
            self.assertEqual(3, report["seed_count"])
            self.assertAlmostEqual(0.4, report["perturbed_parent_macro_mae_mean"])

    def test_aggregate_requires_five_seeds_for_every_required_xrd_model(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for model in ["peak-xgboost", "resnet1d", "transformer"]:
                for seed in range(2027, 2032):
                    path = root / model / str(seed)
                    path.mkdir(parents=True)
                    (path / "perturbation_summary.json").write_text(
                        json.dumps({"status": "pass", "evaluation_scope": "held_out_test", "clean_reference_scope": "identical_structure_states", "model": model, "seed": seed, "perturbed_parent_macro_mae": 0.4, "clean_to_perturbed_retention": 0.8})
                    )
            ridge_metrics = root / "ridge.metrics.json"
            ridge_metrics.write_text(json.dumps({"sro": {"masked_parent_macro_mae": 0.45}}))

            unmatched = aggregate_perturbation_reports(
                root,
                minimum_seeds=5,
                required_models=["peak-xgboost", "resnet1d", "transformer"],
                composition_ridge_metrics=ridge_metrics,
            )
            self.assertEqual("unmatched_composition_reference", unmatched["reason"])
            report = aggregate_perturbation_reports(root, minimum_seeds=5,
                required_models=["peak-xgboost", "resnet1d", "transformer"])
            self.assertEqual("pass", report["status"])
            self.assertEqual(5, report["seed_count"])
            self.assertEqual(3, len(report["metrics_by_model"]))

            (root / "transformer" / "2031" / "perturbation_summary.json").unlink()
            failed = aggregate_perturbation_reports(
                root,
                minimum_seeds=5,
                required_models=["peak-xgboost", "resnet1d", "transformer"],
            )
            self.assertEqual("fail", failed["status"])
            self.assertEqual([2031], failed["missing_seeds_by_model"]["transformer"])

    def test_checkpoint_evaluation_aggregates_views_by_state_then_parent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            run = root / "xrd" / "runs" / "observations"
            (root / "manifests").mkdir(parents=True)
            (run / "grids").mkdir(parents=True)
            targets = json.loads(Path("configs/hea_validation_v1.json").read_text())["sro_targets"]
            task_rows = []
            for index in range(2):
                task_rows.append(
                    {
                        "structure_state_id": f"state-{index}",
                        "parent_group_id": f"parent-{index}",
                        "id_split": "test",
                        "sro_shell1_json": json.dumps({"Al->Co": 0.1 * index}),
                        "sro_shell2_json": "{}",
                    }
                )
            pd.DataFrame(task_rows).to_parquet(root / "manifests" / "task_samples.parquet")
            observations = pd.DataFrame(
                [
                    {"xrd_sample_id": "obs-0-a", "structure_state_id": "state-0", "parent_group_id": "parent-0", "grid_shard_path": "xrd/runs/observations/grids/grid.h5", "grid_row_index": 0, "perturbation_json": json.dumps({"zero_shift_deg": 0.0, "instrument_fwhm_deg": 0.05, "microstrain_fraction": 0.0, "crystallite_size_nm": 200.0, "background_fraction": 0.0, "gaussian_noise_fraction": 0.001, "poisson_scale": 100.0, "march_dollase_r": 1.0})},
                    {"xrd_sample_id": "obs-0-b", "structure_state_id": "state-0", "parent_group_id": "parent-0", "grid_shard_path": "xrd/runs/observations/grids/grid.h5", "grid_row_index": 1, "perturbation_json": json.dumps({"zero_shift_deg": 0.1, "instrument_fwhm_deg": 0.3, "microstrain_fraction": 0.003, "crystallite_size_nm": 20.0, "background_fraction": 0.1, "gaussian_noise_fraction": 0.02, "poisson_scale": 10.0, "march_dollase_r": 0.7})},
                    {"xrd_sample_id": "obs-1-a", "structure_state_id": "state-1", "parent_group_id": "parent-1", "grid_shard_path": "xrd/runs/observations/grids/grid.h5", "grid_row_index": 2, "perturbation_json": json.dumps({"zero_shift_deg": 0.05, "instrument_fwhm_deg": 0.15, "microstrain_fraction": 0.001, "crystallite_size_nm": 100.0, "background_fraction": 0.05, "gaussian_noise_fraction": 0.01, "poisson_scale": 50.0, "march_dollase_r": 1.1})},
                ]
            )
            observations.to_parquet(run / "xrd_samples.parquet")
            with h5py.File(run / "grids" / "grid.h5", "w") as handle:
                handle.create_dataset("intensity", data=np.ones((3, 4001), dtype=np.float32))
                handle.create_dataset("xrd_sample_id", data=np.asarray([b"obs-0-a", b"obs-0-b", b"obs-1-a"]))
            checkpoint = root / "model.pt"
            model = ResNet1D(len(targets))
            torch.save({"model": "resnet1d", "seed": 2027, "state_dict": model.state_dict()}, checkpoint)
            clean_score = root / "clean" / "metrics.json"
            clean_score.parent.mkdir()
            clean_score.write_text(json.dumps({"sro": {"masked_parent_macro_mae": 0.25}}))

            report = evaluate_perturbed_checkpoint(
                root,
                "observations",
                checkpoint,
                Path("configs/hea_validation_v1.json"),
                root / "result",
                batch_size=2,
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(3, report["observation_count"])
            self.assertEqual(2, report["structure_state_count"])
            self.assertEqual(2, report["parent_group_count"])
            self.assertNotIn("clean_to_perturbed_retention", report)
            self.assertEqual("held_out_test", report["evaluation_scope"])
            self.assertGreaterEqual(len(report["severity_curve"]), 2)
            predictions = pd.read_parquet(root / "result" / "perturbation_predictions.parquet")
            self.assertEqual(3, len(predictions))
            self.assertIn("perturbation_severity", predictions)

    def test_peak_model_bundle_is_reloadable_for_perturbed_views(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            run = root / "xrd" / "runs" / "observations"
            (root / "manifests").mkdir(parents=True)
            (run / "grids").mkdir(parents=True)
            config = json.loads(Path("configs/hea_validation_v1.json").read_text())
            targets = config["sro_targets"]
            truth = np.zeros((2, len(targets)), dtype=float)
            truth[:, 0] = [0.0, 0.1]
            task_rows = [
                {"structure_state_id": f"state-{index}", "parent_group_id": f"parent-{index}", "id_split": "test", "sro_shell1_json": json.dumps({"Al->Al": float(truth[index, 0])}), "sro_shell2_json": "{}"}
                for index in range(2)
            ]
            pd.DataFrame(task_rows).to_parquet(root / "manifests" / "task_samples.parquet")
            axis = np.linspace(20.0, 100.0, 4001)
            spectra = np.zeros((2, 4001), dtype=np.float32)
            spectra[0, 1000] = 100.0
            spectra[1, 2000] = 100.0
            observations = pd.DataFrame(
                [
                    {"xrd_sample_id": f"obs-{index}", "structure_state_id": f"state-{index}", "parent_group_id": f"parent-{index}", "grid_shard_path": "xrd/runs/observations/grids/grid.h5", "grid_row_index": index}
                    for index in range(2)
                ]
            )
            observations.to_parquet(run / "xrd_samples.parquet")
            with h5py.File(run / "grids" / "grid.h5", "w") as handle:
                handle.create_dataset("two_theta", data=axis)
                handle.create_dataset("intensity", data=spectra)
                handle.create_dataset("xrd_sample_id", data=np.asarray([b"obs-0", b"obs-1"]))
            features = np.stack([peak_features(row, axis, max_peaks=1) for row in spectra])
            mask = np.ones_like(truth, dtype=bool)
            estimator = PerTargetRidge().fit(features, truth, mask)
            checkpoint = root / "peak.joblib"
            joblib.dump(
                {"model": "peak-xgboost", "seed": 2027, "split_column": "id_split", "max_peaks": 1, "estimator": estimator},
                checkpoint,
            )
            clean_score = root / "clean.json"
            clean_score.write_text(json.dumps({"sro": {"masked_parent_macro_mae": 0.25}}))
            report = evaluate_perturbed_checkpoint(
                root,
                "observations",
                checkpoint,
                Path("configs/hea_validation_v1.json"),
                root / "result",
            )
            self.assertEqual("peak-xgboost", report["model"])
            self.assertEqual(2, report["observation_count"])


if __name__ == "__main__":
    unittest.main()
