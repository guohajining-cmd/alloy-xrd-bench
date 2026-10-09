"""Show all seeds in matched clean/perturbed panels without hiding heavy tails."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tables = []
    for directory in args.results:
        if json.loads((directory/'protocol.json').read_text())['status'] != 'verified':
            raise ValueError('Only completed verified evaluations can be plotted')
        tables.append(pd.read_csv(directory/'scores.csv'))
    table = pd.concat(tables, ignore_index=True)
    models = ['peak-xgboost', 'resnet1d', 'transformer', 'physics-transformer']
    labels = ['Peak-XGBoost', 'ResNet1D', 'Transformer', 'Physics\nTransformer']
    names = dict(id_split='ID', composition_ood_split='Composition OOD',
                 chemistry_ood_split='Chemistry OOD', sro_ood_split='Frozen SRO-OOD')
    splits = [s for s in names if s in set(table.split_column)]
    expected = {(m, seed, split) for m in models for seed in range(2027, 2032) for split in splits}
    keys = list(table[['model', 'seed', 'split_column']].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate model/seed/split matrix')
    values = table[['clean_mae', 'perturbed_mae']].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Log-scale figure requires positive finite MAEs')
    if args.output.exists():
        raise ValueError('Use a fresh figure directory')
    args.output.mkdir(parents=True)
    columns = min(2, len(splits))
    rows = int(np.ceil(len(splits)/columns))
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(rows, columns, figsize=(6.8*columns, 4.8*rows), squeeze=False, sharey=True)
        limits = (float(values.min()*.7), float(values.max()*1.5))
        for index, split in enumerate(splits):
            ax = axes.flat[index]
            part = table[table.split_column.eq(split)]
            if part.parent_count.nunique() != 1 or part.observation_count.nunique() != 1:
                raise ValueError('Models evaluated on different panel sizes')
            for x, model in enumerate(models):
                for jitter, record in zip(np.linspace(-.04, .04, 5), part[part.model.eq(model)].sort_values('seed').itertuples()):
                    xs = [x-.13+jitter, x+.13+jitter]
                    ax.plot(xs, [record.clean_mae, record.perturbed_mae], color='#aaaaaa', lw=.8, zorder=1)
                    ax.scatter(xs[0], record.clean_mae, color='#0072B2', marker='o', s=24, zorder=2)
                    ax.scatter(xs[1], record.perturbed_mae, color='#D55E00', marker='^', s=29, zorder=2)
            ax.set_yscale('log')
            ax.set_ylim(*limits)
            ax.set_xticks(range(4), labels)
            ax.set_xlim(-.5, 3.5)
            ax.grid(axis='y', which='both', color='#dddddd', lw=.6)
            ax.set_title(f"{names[split]}: {int(part.parent_count.iloc[0])} held-out parents", loc='left', fontsize=12)
            ax.set_ylabel('Matched parent-macro MAE (log scale)')
        for ax in list(axes.flat)[len(splits):]:
            ax.set_visible(False)
        fig.legend(handles=[Line2D([], [], marker='o', color='#0072B2', linestyle='none', label='Clean'),
                            Line2D([], [], marker='^', color='#D55E00', linestyle='none', label='Perturbed')],
                   loc='upper center', ncol=2, frameon=False)
        fig.text(.5, .022, 'Each connected pair is one seed on identical states and corrected targets. All five seeds retained.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.11/columns, right=.98, top=.87 if rows==1 else .93,
                            bottom=.19 if rows==1 else .11, hspace=.35, wspace=.18)
        fig.savefig(args.output/'matched_perturbations.png', dpi=300)
        fig.savefig(args.output/'matched_perturbations.pdf')
        plt.close(fig)
    table.to_csv(args.output/'source_data.csv', index=False)
    (args.output/'metadata.json').write_text(json.dumps(dict(log_scale=True, clipped_seeds=False,
        qualification='Frozen synthetic perturbation panel, not full-test or experimental noise validation'), indent=2))


if __name__ == '__main__':
    main()
