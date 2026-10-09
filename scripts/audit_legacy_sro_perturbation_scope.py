"""Read-only audit of historical perturbation membership and clean references."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.dataset/'manifests/task_samples.parquet'
    tasks = pd.read_parquet(manifest, columns=['structure_state_id', 'parent_group_id', 'id_split'])
    records = []
    for path in sorted((args.run/'perturbation').glob('*/*/perturbation_summary.json')):
        summary = json.loads(path.read_text())
        predictions = path.parent/'perturbation_predictions.parquet'
        frame = pd.read_parquet(predictions, columns=['xrd_sample_id', 'structure_state_id', 'parent_group_id'])
        merged = frame.merge(tasks, on='structure_state_id', validate='many_to_one', how='left', suffixes=('', '_truth'))
        if len(merged) != len(frame) or merged.id_split.isna().any() or not merged.parent_group_id.eq(merged.parent_group_id_truth).all():
            raise ValueError('Historical prediction identity mismatch')
        model = summary['model']
        prefix = model.replace('-', '_') if model=='peak-xgboost' else model
        clean_path = args.run/'predictions'/f"{prefix}_seed{summary['seed']}_id_split_test.metrics.json"
        clean = json.loads(clean_path.read_text())
        records.append(dict(model=model, seed=summary['seed'], observation_count=len(frame),
            parent_count=frame.parent_group_id.nunique(),
            observations_by_original_split=merged.id_split.value_counts().to_dict(),
            parents_by_original_split=merged.groupby('id_split').parent_group_id.nunique().to_dict(),
            clean_reference_equals_full_test_metric=abs(summary['clean_parent_macro_mae']-clean['sro']['masked_parent_macro_mae']) < 1e-10,
            clean_reference_parent_count=clean['sro']['parent_count'],
            summary_sha256=digest(path), prediction_sha256=digest(predictions)))
    if len(records) != 20:
        raise ValueError('Expected 20 historical perturbation cases')
    print(json.dumps(dict(source_manifest_sha256=digest(manifest), cases=records,
        conclusion='Inspect observed split membership and clean-reference population; not a label or model rerun'), indent=2))


if __name__ == '__main__':
    main()
