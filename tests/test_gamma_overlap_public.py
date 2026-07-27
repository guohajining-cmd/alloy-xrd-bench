from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class GammaOverlapPublicTests(unittest.TestCase):
    def test_instrument_ood_requires_two_train_quantile_violations(self):
        from alloy_xrd_bench.gamma_overlap_public import _write_instrument_slices

        with tempfile.TemporaryDirectory() as tempdir:
            core = Path(tempdir)
            train = [
                {
                    "sample_id": f"train-{index}",
                    "split": "train",
                    "instrument_fwhm_deg": float(index),
                    "microstrain_fraction": float(index),
                }
                for index in range(10)
            ]
            samples = pd.DataFrame(train + [
                {
                    "sample_id": "one-outside",
                    "split": "validation",
                    "instrument_fwhm_deg": 20.0,
                    "microstrain_fraction": 5.0,
                },
                {
                    "sample_id": "two-outside",
                    "split": "test",
                    "instrument_fwhm_deg": 20.0,
                    "microstrain_fraction": 20.0,
                },
            ])

            _write_instrument_slices(core, samples)

            thresholds = json.loads((core / "slices/instrument_thresholds.json").read_text())
            slices = pd.read_parquet(core / "slices/instrument_slices.parquet").set_index("sample_id")
            self.assertEqual(2, thresholds["minimum_outside_fields"])
            self.assertEqual("instrument_id", slices.loc["one-outside", "instrument_slice"])
            self.assertEqual("instrument_ood", slices.loc["two-outside", "instrument_slice"])

    def test_release_validation_rejects_missing_overlap_slice_manifest(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release, validate_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            (core / "slices/overlap_slices.parquet").unlink()

            report = validate_public_release(core)

            self.assertIn("window:missing_overlap_slice_manifest", report["issues"])

    def test_release_validation_does_not_mutate_frozen_audit_artifacts(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release, validate_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            summary = core / "audits/summary.json"
            frozen_content = b"frozen release audit\n"
            summary.write_bytes(frozen_content)

            validate_public_release(core)

            self.assertEqual(frozen_content, summary.read_bytes())

    def test_release_validation_enforces_split_specific_pattern_arrays(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release, validate_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            samples = pd.read_parquet(core / "full/samples.parquet")
            validation = samples.loc[samples["split"].eq("validation")].iloc[0]
            artifact_path = core / validation["pattern_path"]
            with np.load(artifact_path) as artifact:
                grid = artifact["two_theta"]
                observed = artifact["observed"]
            np.savez_compressed(
                artifact_path,
                two_theta=grid,
                observed=observed,
                clean=observed,
            )
            samples.loc[
                samples["sample_id"].eq(validation["sample_id"]), "artifact_sha256"
            ] = _sha256(artifact_path)
            samples.to_parquet(core / "full/samples.parquet", index=False)

            report = validate_public_release(core)

            self.assertIn("full:validation:invalid_pattern_arrays", report["issues"])

    def test_public_cli_export_validate_template_and_evaluate(self):
        from alloy_xrd_bench.gamma_overlap import main

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            release = root / "release"
            _write_internal_release(source)

            self.assertEqual(0, main(["export-public", "--source", str(source), "--output", str(release)]))
            core = release / "gamma-decomp-core-v1.1"
            self.assertEqual(0, main(["validate-release", "--release", str(core)]))
            submission = root / "submission"
            self.assertEqual(
                0,
                main([
                    "make-submission-template", "--release", str(core), "--task", "window",
                    "--split", "validation", "--output", str(submission),
                ]),
            )
            self.assertEqual(
                0,
                main([
                    "make-oracle", "--release", str(core), "--task", "window",
                    "--split", "validation", "--output", str(root / "oracle"),
                ]),
            )
            self.assertTrue((root / "oracle/submission_components.parquet").exists())
            self.assertEqual(
                0,
                main([
                    "train-baseline", "--release", str(core), "--task", "window",
                    "--model", "resnet-set", "--output", str(root / "neural-cli"),
                    "--epochs", "1", "--patience", "1", "--batch-size", "1",
                    "--embedding-dim", "32", "--device", "cpu",
                ]),
            )
            self.assertEqual(
                0,
                main([
                    "train-baseline", "--release", str(core), "--task", "window",
                    "--split", "validation", "--model", "bic", "--output", str(root / "baseline"),
                ]),
            )
            self.assertEqual(
                0,
                main([
                    "evaluate", "--release", str(core), "--task", "window", "--split", "validation",
                    "--submission", str(submission), "--output", str(root / "score"),
                ]),
            )
            self.assertNotIn("NaN", (root / "score/scorecard.json").read_text())

    def test_export_loader_template_and_perfect_evaluation(self):
        from alloy_xrd_bench.gamma_overlap_public import (
            GammaOverlapDataset,
            evaluate_public_submission,
            export_public_release,
            make_submission_template,
            validate_public_release,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            report = export_public_release(source_dir=source, output_dir=root / "release", include_provenance=True)
            core = Path(report["core_dir"])

            self.assertEqual("gamma-decomp-core-v1.1", core.name)
            self.assertTrue((root / "release/gamma-decomp-provenance-v1.1").exists())
            for relative in (
                "task_registry.json", "dataset_card.md", "splits/train.parquet",
                "splits/validation.parquet", "splits/test.parquet",
                "schemas/submission_samples.schema.json",
                "schemas/submission_components.schema.json", "checksums.sha256",
                "baselines/training_candidate_catalog.parquet",
                "slices/instrument_thresholds.json",
                "slices/instrument_slices.parquet",
                "slices/overlap_slices.parquet",
            ):
                self.assertTrue((core / relative).exists(), relative)
            registry = json.loads((core / "task_registry.json").read_text())
            self.assertEqual(
                {"train": 1, "validation": 1, "test": 1},
                registry["tasks"]["gamma_full_decomp_v1"]["split_counts"],
            )
            self.assertFalse((core / "private").exists())
            self.assertEqual({"train", "validation", "test"}, set(pd.read_parquet(core / "full/samples.parquet")["split"]))
            full_sample_columns = set(pd.read_parquet(core / "full/samples.parquet").columns)
            self.assertFalse(
                {"parent_pair_id", "sample_seed", "gamma_prime_weight", "instrument_fwhm_deg"}
                & full_sample_columns
            )
            self.assertTrue((core / "full/evaluation_groups.parquet").exists())
            self.assertTrue((core / "window/evaluation_groups.parquet").exists())
            self.assertTrue(
                pd.read_parquet(core / "full/components.parquet")["split"].eq("test").any()
            )
            self.assertTrue(
                pd.read_parquet(core / "window/components.parquet")["split"].eq("test").any()
            )
            self.assertTrue(
                {"hkl_group_id", "allowed_hkls_json", "degenerate_group", "merged_integrated_intensity"}
                <= set(pd.read_parquet(core / "full/components.parquet").columns)
            )
            window_samples = pd.read_parquet(core / "window/samples.parquet")
            self.assertFalse(
                {
                    "difficulty",
                    "minimum_separation_ratio",
                    "scored_component_count",
                    "context_component_ids_json",
                }
                & set(window_samples.columns)
            )
            overlap_slices = pd.read_parquet(core / "slices/overlap_slices.parquet")
            self.assertEqual(1, int(overlap_slices.iloc[0]["scored_component_count"]))
            with np.load(core / pd.read_parquet(core / "full/samples.parquet").iloc[0]["pattern_path"]) as artifact:
                self.assertEqual({"two_theta", "observed", "clean"}, set(artifact.files))
            test_sample = pd.read_parquet(core / "full/samples.parquet")
            test_sample = test_sample.loc[test_sample["split"].eq("test")].iloc[0]
            with np.load(core / test_sample["pattern_path"]) as artifact:
                self.assertEqual({"two_theta", "observed"}, set(artifact.files))
            self.assertEqual("pass", validate_public_release(core)["status"])
            full = GammaOverlapDataset(core, task="full", split="validation")
            window = GammaOverlapDataset(core, task="window", split="validation")
            self.assertEqual((4001, 2), full[0]["x"].shape)
            self.assertEqual((151, 2), window[0]["x"].shape)

            submission = root / "submission"
            make_submission_template(core, task="window", split="validation", output_dir=submission)
            self.assertEqual(
                {"sample_id", "predicted_component_count"},
                set(pd.read_parquet(submission / "submission_samples.parquet").columns),
            )
            truth = pd.read_parquet(core / "window/components.parquet")
            truth = truth[(truth["split"] == "validation") & truth["score_eligible"]].copy()
            truth = truth[[
                "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id", "center_deg",
                "fwhm_deg", "integrated_intensity", "eta",
            ]]
            truth["component_id"] = "prediction_" + truth["component_id"].astype(str)
            truth["confidence"] = 1.0
            truth.to_parquet(submission / "submission_components.parquet", index=False)
            samples = pd.read_parquet(submission / "submission_samples.parquet")
            counts = truth.groupby("sample_id").size()
            samples["predicted_component_count"] = samples["sample_id"].map(counts).fillna(0).astype(int)
            samples.to_parquet(submission / "submission_samples.parquet", index=False)
            score = evaluate_public_submission(
                release_dir=core, task="window", split="validation", submission_dir=submission, output_dir=root / "score"
            )
            self.assertEqual(1.0, score["metrics"]["component_f1"])
            self.assertTrue((root / "score/scorecard.json").exists())
            self.assertTrue((root / "score/per_parent_pair.parquet").exists())
            self.assertTrue((root / "score/per_difficulty.parquet").exists())
            self.assertTrue((root / "score/per_instrument_slice.parquet").exists())
            self.assertTrue((root / "score/submission_validation.json").exists())

    def test_submission_requires_eta_and_exact_sample_coverage(self):
        from alloy_xrd_bench.gamma_overlap_public import evaluate_public_submission, export_public_release, make_submission_template

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source_dir=source, output_dir=root / "release")["core_dir"])
            submission = root / "submission"
            make_submission_template(core, task="full", split="validation", output_dir=submission)
            truth = pd.read_parquet(core / "full/components.parquet").query("split == 'validation'").copy()
            truth["confidence"] = 1.0
            truth = truth.drop(columns="eta")
            truth.to_parquet(submission / "submission_components.parquet", index=False)
            samples = pd.read_parquet(submission / "submission_samples.parquet")
            samples["predicted_component_count"] = 1
            samples.to_parquet(submission / "submission_samples.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "eta"):
                evaluate_public_submission(core, "full", "validation", submission, root / "score")

            invalid = pd.read_parquet(core / "full/components.parquet").query("split == 'validation'").copy()
            invalid = invalid[[
                "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id", "center_deg",
                "fwhm_deg", "integrated_intensity", "eta",
            ]]
            invalid["confidence"] = 1.0
            invalid["phase"] = "not_a_phase"
            invalid.to_parquet(submission / "submission_components.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "phase/HKL"):
                evaluate_public_submission(core, "full", "validation", submission, root / "score")

    def test_public_window_baseline_produces_evaluable_submission(self):
        from alloy_xrd_bench.gamma_overlap_baseline import run_public_window_baseline
        from alloy_xrd_bench.gamma_overlap_public import export_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            report = run_public_window_baseline(
                release_dir=core,
                output_dir=root / "baseline",
                split="validation",
                baseline_kind="bic",
            )
            self.assertEqual("pass", report["status"])
            self.assertEqual(1, report["sample_count"])
            self.assertTrue((root / "baseline/submission/submission_samples.parquet").exists())
            self.assertTrue((root / "baseline/score/scorecard.json").exists())

    def test_neural_set_baseline_runs_public_training_smoke(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release
        from alloy_xrd_bench.gamma_overlap_training import train_set_baseline

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            report = train_set_baseline(
                release_dir=core,
                output_dir=root / "neural",
                task="window",
                encoder_name="resnet1d",
                seed=2027,
                maximum_epochs=1,
                patience=1,
                batch_size=1,
                embedding_dim=32,
                device_name="cpu",
                max_train_samples=1,
                max_validation_samples=1,
            )
            self.assertEqual("pass", report["status"])
            self.assertTrue((root / "neural/best_model.pt").exists())
            self.assertTrue((root / "neural/validation_score/scorecard.json").exists())

    def test_weak_peak_ablation_masks_only_model_input(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release
        from alloy_xrd_bench.gamma_overlap_training import GammaSetDataset, build_public_class_catalog

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            catalog = build_public_class_catalog(core, "full")
            clean = GammaSetDataset(core, "full", "train", catalog)[0]
            ablated = GammaSetDataset(
                core, "full", "train", catalog, minimum_relative_intensity=0.1
            )[0]

            self.assertGreater(
                int(torch.count_nonzero(clean["x"][1])),
                int(torch.count_nonzero(ablated["x"][1])),
            )
            torch.testing.assert_close(
                clean["target"]["intensity"], ablated["target"]["intensity"]
            )

    def test_clean_training_view_is_train_only(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release
        from alloy_xrd_bench.gamma_overlap_training import GammaSetDataset, build_public_class_catalog

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])
            catalog = build_public_class_catalog(core, "full")

            clean_train = GammaSetDataset(
                core, "full", "train", catalog, spectrum_view="clean"
            )
            self.assertEqual((2, 4001), tuple(clean_train[0]["x"].shape))
            with self.assertRaisesRegex(ValueError, "train split"):
                GammaSetDataset(core, "full", "validation", catalog, spectrum_view="clean")

    def test_neural_set_baseline_can_hold_out_test_for_final_evaluation(self):
        from alloy_xrd_bench.gamma_overlap_public import export_public_release
        from alloy_xrd_bench.gamma_overlap_training import train_set_baseline

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])

            report = train_set_baseline(
                release_dir=core,
                output_dir=root / "neural-test",
                task="window",
                encoder_name="resnet1d",
                evaluation_split="test",
                seed=2027,
                maximum_epochs=1,
                patience=1,
                batch_size=1,
                embedding_dim=32,
                device_name="cpu",
            )

            self.assertEqual("test", report["evaluation_split"])
            self.assertTrue((root / "neural-test/validation_score/scorecard.json").exists())
            self.assertTrue((root / "neural-test/test_submission/submission_samples.parquet").exists())
            self.assertTrue((root / "neural-test/test_score/scorecard.json").exists())

    def test_no_spectrum_prior_uses_train_labels_and_scores_test(self):
        from alloy_xrd_bench.gamma_overlap_baseline import run_public_no_spectrum_prior
        from alloy_xrd_bench.gamma_overlap_public import export_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])

            report = run_public_no_spectrum_prior(
                release_dir=core,
                output_dir=root / "prior",
                task="window",
                split="test",
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual("no-spectrum", report["baseline_kind"])
            self.assertTrue((root / "prior/score/scorecard.json").exists())
            self.assertTrue((root / "prior/submission/submission_components.parquet").exists())

    def test_peak_feature_baseline_trains_and_scores_without_raw_grid_features(self):
        from alloy_xrd_bench.gamma_overlap_baseline import run_public_peak_feature_baseline
        from alloy_xrd_bench.gamma_overlap_public import export_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])

            report = run_public_peak_feature_baseline(
                release_dir=core,
                output_dir=root / "peak-feature",
                task="window",
                split="validation",
                seed=2027,
                n_estimators=4,
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual("peak-feature", report["baseline_kind"])
            self.assertTrue((root / "peak-feature/model.joblib").exists())
            self.assertTrue((root / "peak-feature/score/scorecard.json").exists())


def _write_internal_release(root: Path) -> None:
    (root / "patterns").mkdir(parents=True)
    (root / "windows").mkdir()
    (root / "private").mkdir()
    samples = []
    components = []
    windows = []
    splits = ("train", "validation", "private_test")
    for index, split in enumerate(splits):
        sample_id = f"sample_{index}"
        grid = np.linspace(20.0, 100.0, 4001)
        observed = np.exp(-0.5 * ((grid - 44.0) / 0.1) ** 2)
        np.savez_compressed(root / f"patterns/{sample_id}.npz", two_theta=grid, observed=observed, clean=observed, background=np.zeros_like(grid))
        samples.append({"sample_id": sample_id, "parent_pair_id": f"pair_{index}", "split": split, "point_count": 4001,
                        "pattern_path": f"patterns/{sample_id}.npz", "pattern_sha256": "unused"})
        component_id = f"component_{index}"
        component = {"sample_id": sample_id, "component_id": component_id, "parent_pair_id": f"pair_{index}",
                     "phase": "gamma", "h": 1, "k": 1, "l": 1, "center_deg": 44.0,
                     "fwhm_deg": 0.2, "integrated_intensity": 1.0, "eta": 0.5,
                     "profile_kind": "pseudo_voigt", "score_eligible": True, "split": split, "window_ids_json": "[]"}
        components.append(component)
        window_id = f"window_{index}"
        start = 1125
        np.savez_compressed(root / f"windows/{window_id}.npz", two_theta=grid[start:start+151], observed=observed[start:start+151], clean=observed[start:start+151])
        windows.append({"window_id": window_id, "sample_id": sample_id, "parent_pair_id": f"pair_{index}",
                        "two_theta_min": float(grid[start]), "two_theta_max": float(grid[start+150]), "component_count": 1,
                        "difficulty": "single", "minimum_separation_ratio": None,
                        "target_component_ids_json": json.dumps([component_id]), "window_path": f"windows/{window_id}.npz",
                        "window_sha256": "unused", "split": split})
    pd.DataFrame(samples).to_parquet(root / "mixture_samples.parquet", index=False)
    frame = pd.DataFrame(components)
    frame[frame.split != "private_test"].to_parquet(root / "components.parquet", index=False)
    frame[frame.split == "private_test"].to_parquet(root / "private/components.parquet", index=False)
    window_frame = pd.DataFrame(windows)
    public_windows = window_frame.copy()
    public_windows.loc[public_windows.split == "private_test", ["component_count", "difficulty", "minimum_separation_ratio", "target_component_ids_json"]] = None
    public_windows.to_parquet(root / "overlap_windows.parquet", index=False)
    window_frame[window_frame.split == "private_test"].to_parquet(root / "private/overlap_windows_truth.parquet", index=False)
    pd.DataFrame([{"parent_pair_id": f"pair_{i}", "gamma_parent_id": f"g{i}", "gamma_prime_parent_id": f"gp{i}",
                  "source_run_id": f"run{i}", "split": split, "split_component_id": f"sc{i}"} for i, split in enumerate(splits)]).to_parquet(root / "parent_pairs.parquet", index=False)
    for name in ("parent_structures", "phase_patterns", "phase_reflections", "satellite_reflections", "provenance_edges"):
        pd.DataFrame({"id": []}).to_parquet(root / f"{name}.parquet", index=False)
    (root / "summary.json").write_text(json.dumps({"release_id": "test", "mixture_sample_count": 3}))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
