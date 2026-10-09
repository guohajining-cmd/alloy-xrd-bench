"""Read-only checkpoint replay across CPU/CUDA numeric backends."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alloy_xrd_bench.hea_perturbation import _model_from_checkpoint
from alloy_xrd_bench.hea_validation import _predict_sro
from alloy_xrd_bench.hea_validation_models import seed_everything, spectrum_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    seed_everything(payload['seed'])
    saved = pd.read_parquet(args.predictions).iloc[:128]
    manifest = pd.read_parquet(args.dataset/'manifests/task_samples.parquet').set_index('xrd_sample_id')
    selected = manifest.loc[saved.xrd_sample_id].reset_index()
    rows = [SimpleNamespace(xrd_sample_id=r.xrd_sample_id, grid_shard_path=str(args.dataset/r.grid_shard_path),
        grid_row_index=r.grid_row_index) for r in selected.itertuples()]
    features = spectrum_matrix(rows)
    truth = np.stack(saved.sro_predictions)
    report = dict(torch_version=torch.__version__, numpy_version=np.__version__,
        input_sha256=hashlib.sha256(features.tobytes()).hexdigest(), sample_count=len(features), cases=[])
    cases = [('cpu', False, True), ('cpu', False, False)]
    if args.cuda:
        cases += [('cuda:0', True, True), ('cuda:0', False, True)]
    for device, tf32, mkldnn in cases:
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.mkldnn.enabled = mkldnn
        model = _model_from_checkpoint(payload, 128).to(device)
        values = _predict_sro(model, features, batch_size=128, device=torch.device(device))
        difference = np.abs(values-truth)
        report['cases'].append(dict(device=device, tf32=tf32, mkldnn=mkldnn,
            max_absolute_difference=float(difference.max()), mean_absolute_difference=float(difference.mean()),
            q99_absolute_difference=float(np.quantile(difference, .99))))
        del model
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
