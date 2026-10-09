import json
import pytest

from scripts.rescore_sro_matrix import verify_downloads, expected_runs, verify_score
from scripts.run_sro_corrected_batch import build_jobs


def test_full_matrix_has_84_unique_runs():
    config = dict(split_columns=["id_split", "composition_ood_split", "chemistry_ood_split", "sro_ood_split"], seeds=list(range(2027, 2032)))
    runs = expected_runs(config)
    assert len(runs) == len(set(runs)) == 84
    assert len([r for r in runs if r[0] == "composition-ridge"]) == 4


def test_download_hash_mismatch_stops_scoring(tmp_path):
    (tmp_path / "x.parquet").write_bytes(b"wrong")
    (tmp_path / "remote_files.sha256").write_text("0"*64 + "  /remote/x.parquet\n")
    with pytest.raises(ValueError, match="hash"):
        verify_downloads(tmp_path)


def test_download_manifest_cannot_alias_two_sources(tmp_path):
    (tmp_path / "remote_files.sha256").write_text("0"*64+"  /a/x.parquet\n"+"0"*64+"  /b/x.parquet\n")
    with pytest.raises(ValueError):
        verify_downloads(tmp_path)


def test_external_download_manifest_with_nested_predictions(tmp_path):
    import hashlib
    directory = tmp_path / 'run'
    (directory/'predictions').mkdir(parents=True)
    (directory/'predictions/x.parquet').write_bytes(b'checked')
    digest = hashlib.sha256(b'checked').hexdigest()
    manifest = tmp_path/'checksums.sha256'
    manifest.write_text(digest+'  /server/predictions/x.parquet\n')
    assert verify_downloads(directory, manifest) == {'x.parquet': digest}


def test_corrected_ood_batch_has_60_unique_cases():
    splits = ['composition_ood_split', 'chemistry_ood_split', 'sro_ood_split']
    config = dict(seeds=list(range(2027, 2032)), split_columns=splits)
    jobs = build_jobs(config, ['peak-xgboost', 'resnet1d', 'transformer', 'physics-transformer'], config['seeds'], splits)
    assert len(jobs) == len(set(jobs)) == 60


def test_corrected_id_batch_retains_five_seeds_and_rejects_duplicates():
    config = dict(seeds=list(range(2027, 2032)), split_columns=["id_split"])
    models = ["peak-xgboost", "resnet1d", "transformer", "physics-transformer"]
    assert len(build_jobs(config, models, config["seeds"], ["id_split"])) == 20
    with pytest.raises(ValueError):
        build_jobs(config, models, [2027, 2027], ["id_split"])
    with pytest.raises(ValueError):
        build_jobs(config, models, [2027], ["unknown_split"])


def test_recomputed_scores_check_counts_and_shell_correlations():
    score = dict(masked_parent_macro_mae=.4, rmse=.5, parent_count=12,
                 valid_target_count=50, shell_spearman=dict(shell1=.3, shell2=None))
    verify_score(score, score, "fixture")
    for changed in [dict(parent_count=13), dict(rmse=float("nan")),
                    dict(shell_spearman=dict(shell1=.31, shell2=None)),
                    dict(shell_spearman=dict(shell1=.3, shell2=0.))]:
        with pytest.raises(ValueError, match="fixture"):
            verify_score({**score, **changed}, score, "fixture")
