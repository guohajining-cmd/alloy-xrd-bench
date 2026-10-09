"""Locally re-score downloaded matched perturbation vectors and inspect tails."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.sro_revision import apply_label_revision, file_sha256
from scripts.rescore_sro_matrix import verify_downloads, stem
from scripts.evaluate_sro_corrected_perturbations import matched_test_observations


def aggregate_errors(truth, prediction, mask, frame):
    if truth.shape != prediction.shape or not np.isfinite(prediction).all() or not mask.any(axis=1).all():
        raise ValueError('Invalid score arrays')
    observation_errors = pd.Series(np.where(mask, np.abs(truth-prediction), 0).sum(axis=1)/mask.sum(axis=1))
    state_errors = observation_errors.groupby([frame.parent_group_id, frame.structure_state_id]).mean()
    return observation_errors, state_errors.groupby(level='parent_group_id').mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'observations', 'revision', 'trained-run', 'results', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT/'data'):
        raise ValueError('Use a fresh audit directory outside source data')
    downloads = verify_downloads(args.results)
    verify_downloads(args.trained_run)
    config = json.loads((args.trained_run/'config.json').read_text())
    protocol = json.loads((args.results/'protocol.json').read_text())
    if protocol['status'] != 'verified':
        raise ValueError('Incomplete perturbation evaluation')
    source = args.dataset/'manifests/task_samples.parquet'
    tasks = apply_label_revision(pd.read_parquet(source), source, args.revision, config['sro_targets'])
    observations = pd.read_parquet(args.observations/'xrd/runs/hea_observation_v1_1/xrd_samples.parquet')
    scores = pd.read_csv(args.results/'scores.csv')
    if set(map(tuple, scores[['model', 'seed', 'split_column']].itertuples(index=False, name=None))) != set(map(tuple, protocol['runs'])) or len(scores) != len(protocol['runs']):
        raise ValueError('Incomplete or duplicate run matrix')
    cache, checks = {}, []
    for record in scores.itertuples(index=False):
        split = record.split_column
        if split not in cache:
            frame = matched_test_observations(observations, tasks, split)
            pairs = [vectorize_sro(r.sro_shell1_json, r.sro_shell2_json, config['sro_targets']) for r in frame.itertuples()]
            truth, mask = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
            eligible = mask.any(axis=1)
            frame, truth, mask = frame[eligible].reset_index(drop=True), truth[eligible], mask[eligible]
            selection = pd.read_csv(args.results/f'{split}.selection.csv')
            pd.testing.assert_frame_equal(frame[selection.columns], selection)
            cache[split] = frame, truth, mask
        frame, truth, mask = cache[split]
        name = stem(record.model, record.seed, split)
        result = pd.read_parquet(args.results/f'{name}.parquet')
        if not result.xrd_sample_id.is_unique or set(result.xrd_sample_id) != set(frame.xrd_sample_id):
            raise ValueError('Wrong perturbation observation membership')
        result = result.set_index('xrd_sample_id').loc[frame.xrd_sample_id]
        if result.parent_group_id.tolist() != frame.parent_group_id.tolist() or result.structure_state_id.tolist() != frame.structure_state_id.tolist():
            raise ValueError('Perturbation prediction identity mismatch')
        prediction = np.stack(result.sro_predictions).astype(np.float32)
        saved = pd.read_parquet(args.trained_run/'predictions'/f'{name}.parquet').set_index('xrd_sample_id')
        if not saved.index.is_unique or saved.loc[frame.clean_xrd_sample_id, 'parent_group_id'].tolist() != frame.parent_group_id.tolist():
            raise ValueError('Clean prediction identity mismatch')
        clean = np.stack(saved.loc[frame.clean_xrd_sample_id, 'sro_predictions']).astype(np.float32)
        errors, parents = aggregate_errors(truth, prediction, mask, frame)
        _, clean_parents = aggregate_errors(truth, clean, mask, frame)
        expected = pd.read_csv(args.results/f'{name}.parents.csv').set_index('parent_group_id')
        np.testing.assert_allclose(parents.sort_index(), expected.perturbed.sort_index(), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(clean_parents.sort_index(), expected.clean.sort_index(), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose([parents.mean(), clean_parents.mean()], [record.perturbed_mae, record.clean_mae], atol=1e-7, rtol=1e-7)
        valid = prediction[mask]
        checks.append(dict(model=record.model, seed=int(record.seed), split_column=split,
            clean_mae=float(clean_parents.mean()), perturbed_mae=float(parents.mean()),
            valid_prediction_min=float(valid.min()), valid_prediction_max=float(valid.max()),
            fraction_valid_predictions_above_one=float((valid > 1).mean()),
            observation_mae_median=float(errors.median()), observation_mae_p95=float(errors.quantile(.95)),
            parent_mae_median=float(parents.median()), parent_mae_p95=float(parents.quantile(.95)),
            max_parent_mae=float(parents.max()), worst_parent_id=str(parents.idxmax()),
            valid_entries=int(mask.sum()), parent_count=len(parents),
            prediction_sha256=downloads[f'{name}.parquet']))
    args.output.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(args.output/'verified_scores_and_tails.csv', index=False)
    report = dict(status='verified', run_count=len(checks),
        source_manifest_sha256=file_sha256(source), labels_sha256=file_sha256(args.revision/'labels.parquet'),
        results_protocol_sha256=file_sha256(args.results/'protocol.json'), download_file_count=len(downloads),
        qualification='Predictions and all seeds retained; no clipping. Target-changing mask correction is not experimental SRO validation.')
    (args.output/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
