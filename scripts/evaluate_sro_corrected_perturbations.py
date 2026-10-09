"""Matched held-out perturbation evaluation of verified corrected SRO models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.sro_revision import apply_label_revision, file_sha256
from alloy_xrd_bench.sro_reassessment import paired_interval
from scripts.rescore_sro_matrix import stem, verify_downloads


def matched_test_observations(observations, tasks, split):
    if tasks.structure_state_id.duplicated().any() or observations.xrd_sample_id.duplicated().any():
        raise ValueError('Duplicate structure or observation ID')
    if (tasks.groupby('parent_group_id')[split].nunique() != 1).any():
        raise ValueError('Parent split leakage')
    renamed = tasks.rename(columns={name: 'clean_'+name for name in
                                    ['xrd_sample_id', 'grid_shard_path', 'grid_row_index']})
    merged = observations.merge(renamed, on='structure_state_id', how='left',
        validate='many_to_one', suffixes=('_observation', ''), indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Observation state absent from source manifest')
    for column in ['parent_group_id', 'id_split', split]:
        other = column+'_observation'
        if other in merged and not merged[column].eq(merged[other]).all():
            raise ValueError(f'Observation/source mismatch: {column}')
    if not merged.source_xrd_sample_id.eq(merged.clean_xrd_sample_id).all():
        raise ValueError('Observation has incorrect clean source ID')
    selected = merged[merged[split].eq('test')].reset_index(drop=True)
    if selected.empty:
        raise ValueError('No held-out observations for this split')
    return selected


def paired_errors(truth, mask, clean, perturbed, frame):
    if not (truth.shape == mask.shape == clean.shape == perturbed.shape) or len(frame) != len(truth):
        raise ValueError('Inconsistent matched array shapes')
    if not mask.any(axis=1).all() or not all(np.isfinite(x).all() for x in [truth, clean, perturbed]):
        raise ValueError('Invalid predictions or empty target masks')
    errors = frame[['parent_group_id', 'structure_state_id']].copy()
    for name, prediction in [('clean', clean), ('perturbed', perturbed)]:
        errors[name] = np.where(mask, np.abs(truth-prediction), 0).sum(axis=1)/mask.sum(axis=1)
    states = errors.groupby(['parent_group_id', 'structure_state_id'])[['clean', 'perturbed']].mean()
    return states.groupby(level='parent_group_id').mean()


def load_raw(root, frame, prefix, hashes):
    root = root.resolve()
    output = np.empty((len(frame), 4001), dtype=np.float32)
    common_axis = None
    for relative, indexes in frame.groupby(prefix+'grid_shard_path').groups.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Spectrum path escapes source root')
        hashes[str(path)] = file_sha256(path)
        subset = frame.loc[indexes]
        rows = subset[prefix+'grid_row_index'].to_numpy(dtype=int)
        with h5py.File(path, 'r') as handle:
            if rows.min() < 0 or rows.max() >= len(handle['xrd_sample_id']):
                raise ValueError('Spectrum row outside shard')
            values = np.asarray(handle['intensity'])[rows].astype(np.float32)
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(handle['xrd_sample_id'])[rows]]
            axis = np.asarray(handle['two_theta'], dtype=np.float32)
        if ids != subset[prefix+'xrd_sample_id'].tolist() or values.shape != (len(subset), 4001):
            raise ValueError('Spectrum identity or shape mismatch')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid spectrum intensities')
        if common_axis is not None and not np.allclose(axis, common_axis, rtol=0, atol=1e-6):
            raise ValueError('Inconsistent diffraction axes')
        common_axis = axis
        output[np.asarray(indexes)] = values
    return output, common_axis


def predict(checkpoint, raw, axis, threads, device='cpu', batch_size=128):
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    from alloy_xrd_bench.hea_perturbation import _model_from_checkpoint
    from alloy_xrd_bench.hea_validation_models import peak_features, seed_everything
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        if checkpoint.suffix == '.joblib':
            payload = joblib.load(checkpoint)
            features = np.stack([peak_features(row, axis, max_peaks=payload['max_peaks']) for row in raw])
            result = payload['estimator'].predict(features)
        else:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            seed_everything(payload['seed'])
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            model = _model_from_checkpoint(payload, 128).to(device)
            # Match the original spectrum_matrix operation order and scalar division.
            transformed = np.empty_like(raw)
            for index, row in enumerate(raw):
                value = np.log1p(row)
                scale = float(value.max())
                transformed[index] = value / scale if scale > 0 else value
            parts = []
            with torch.no_grad():
                for start in range(0, len(raw), batch_size):
                    value = model(torch.from_numpy(transformed[start:start+batch_size]).unsqueeze(1).to(device))
                    parts.append((value['sro'] if isinstance(value, dict) else value).cpu().numpy())
            result = np.concatenate(parts)
    return payload, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--observation-run', default='hea_observation_v1_1')
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--trained-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--checksums', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT/'data') or out.exists() or args.threads < 1:
        raise ValueError('Use a fresh, non-data output directory and positive thread count')
    hashes = verify_downloads(args.trained_run, args.checksums)
    protocol_path = args.trained_run/'protocol.json'
    training = json.loads(protocol_path.read_text())
    if training['status'] != 'complete' or not training['input_hashes_unchanged']:
        raise ValueError('Training not complete/verified')
    manifest_path = args.dataset/'manifests/task_samples.parquet'
    if training['labels_sha256'] != file_sha256(args.revision/'labels.parquet') or training['source_manifest_sha256'] != file_sha256(manifest_path):
        raise ValueError('Training label or source mismatch')
    config = json.loads((args.trained_run/'config.json').read_text())
    if len(config['sro_targets']) != 128:
        raise ValueError('Unsupported output vocabulary')
    tasks = apply_label_revision(pd.read_parquet(manifest_path), manifest_path, args.revision, config['sro_targets'])
    observation_path = args.observations/'xrd/runs'/args.observation_run/'xrd_samples.parquet'
    observations = pd.read_parquet(observation_path)
    out.mkdir(parents=True)
    source_hashes = {str(path.resolve()): file_sha256(path) for path in [manifest_path, observation_path,
        args.revision/'labels.parquet', args.revision/'revision.json',
        args.trained_run/'config.json', protocol_path, Path(__file__)]}
    for relative in ['alloy_xrd_bench/hea_validation_models.py', 'alloy_xrd_bench/hea_perturbation.py',
                     'alloy_xrd_bench/hea_validation_data.py']:
        digest = file_sha256(ROOT/relative)
        if digest != training['source_hashes'][relative]:
            raise ValueError(f'Inference implementation differs from trained snapshot: {relative}')
        source_hashes[str((ROOT/relative).resolve())] = digest
    records, parent_results = [], {}
    for split in dict.fromkeys(row[2] for row in training['runs']):
        frame = matched_test_observations(observations, tasks, split)
        pairs = [vectorize_sro(r.sro_shell1_json, r.sro_shell2_json, config['sro_targets']) for r in frame.itertuples()]
        truth, mask = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        eligible = mask.any(axis=1)
        excluded = int((~eligible).sum())
        frame, truth, mask = frame[eligible].reset_index(drop=True), truth[eligible], mask[eligible]
        if frame.empty:
            raise ValueError('No eligible held-out SRO targets')
        frame[['xrd_sample_id', 'structure_state_id', 'parent_group_id', 'clean_xrd_sample_id']].to_csv(out/f'{split}.selection.csv', index=False)
        # Full original test order preserves neural batch shapes for exact CUDA replay.
        clean_frame = tasks[tasks[split].eq('test')].rename(columns={name: 'clean_'+name for name in
            ['xrd_sample_id', 'grid_shard_path', 'grid_row_index']}).reset_index(drop=True)
        clean_raw, axis = load_raw(args.dataset, clean_frame, 'clean_', source_hashes)
        pert_raw, pert_axis = load_raw(args.observations, frame, '', source_hashes)
        if not np.allclose(axis, pert_axis, rtol=0, atol=1e-6):
            raise ValueError('Clean and perturbed spectra axes differ')
        indexes = pd.Index(clean_frame.structure_state_id).get_indexer(frame.structure_state_id)
        for model, seed, case_split in training['runs']:
            if case_split != split:
                continue
            name = stem(model, seed, split)
            checkpoint = args.trained_run/'predictions'/(name+('.joblib' if model=='peak-xgboost' else '.pt'))
            if hashes[checkpoint.name] != file_sha256(checkpoint):
                raise ValueError('Unbound checkpoint')
            batch_size = int(config['model_budgets'].get(model, {}).get('batch_size', 128))
            payload, clean_prediction = predict(checkpoint, clean_raw, axis, args.threads, args.device, batch_size)
            if (payload['model'], payload['seed'], payload['split_column']) != (model, seed, split):
                raise ValueError('Checkpoint identity mismatch')
            saved = pd.read_parquet(args.trained_run/'predictions'/f'{name}.parquet').set_index('xrd_sample_id')
            if not saved.index.is_unique or not saved.loc[clean_frame.clean_xrd_sample_id, 'parent_group_id'].tolist() == clean_frame.parent_group_id.tolist():
                raise ValueError('Saved prediction parent mismatch')
            reference = np.stack(saved.loc[clean_frame.clean_xrd_sample_id, 'sro_predictions'])
            max_diff = float(np.max(np.abs(reference-clean_prediction)))
            if not np.allclose(reference, clean_prediction, atol=2e-5, rtol=2e-5):
                raise ValueError(f'Clean checkpoint inference does not reproduce saved predictions: {name} maxdiff={max_diff}')
            _, prediction = predict(checkpoint, pert_raw, axis, args.threads, args.device, batch_size)
            parent = paired_errors(truth, mask, clean_prediction[indexes], prediction, frame)
            parent.to_csv(out/f'{name}.parents.csv')
            parent_results[(split, model, seed)] = parent
            record = dict(model=model, seed=seed, split_column=split, clean_mae=float(parent.clean.mean()),
                perturbed_mae=float(parent.perturbed.mean()), delta_mae=float((parent.perturbed-parent.clean).mean()),
                parent_count=len(parent), structure_count=frame.structure_state_id.nunique(), observation_count=len(frame),
                excluded_observations_no_valid_target=excluded, clean_prediction_max_difference=max_diff,
                checkpoint_sha256=hashes[checkpoint.name])
            records.append(record)
            prediction_frame = frame[['xrd_sample_id', 'structure_state_id', 'parent_group_id']].copy()
            prediction_frame['sro_predictions'] = list(prediction)
            prediction_frame.to_parquet(out/f'{name}.parquet', index=False)
            print(json.dumps(record), flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out/'scores.csv', index=False)
    comparisons = []
    for (split, model), part in table.groupby(['split_column', 'model']):
        errors = pd.concat([parent_results[(split, model, seed)] for seed in part.seed]).groupby(level=0).mean()
        comparisons.append(dict(split_column=split, model=model, seed_count=len(part),
            clean_mae_mean=float(part.clean_mae.mean()), perturbed_mae_mean=float(part.perturbed_mae.mean()),
            perturbed_seed_sd=float(part.perturbed_mae.std(ddof=1)) if len(part)>1 else 0.,
            **paired_interval(errors.perturbed-errors.clean)))
    if any(file_sha256(path) != digest for path, digest in source_hashes.items()):
        raise ValueError('Evaluation source changed during run')
    (out/'summary.json').write_text(json.dumps(comparisons, indent=2, allow_nan=False))
    protocol = dict(status='verified', trained_run=str(args.trained_run.resolve()), source_hashes=source_hashes,
        runs=training['runs'], completed_run_count=len(records), device=args.device,
        packages={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'torch', 'xgboost']},
        aggregation='observation -> structure state -> parent; same clean/perturbed structures and corrected masks',
        inference_check='Clean checkpoint predictions reproduced within atol=rtol=2e-5',
        qualification='Frozen simulated perturbation panel, test parents only; not the complete held-out dataset or experimental noise calibration')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'VERIFIED: {len(records)} matched held-out perturbation cases', flush=True)


if __name__ == '__main__':
    main()
