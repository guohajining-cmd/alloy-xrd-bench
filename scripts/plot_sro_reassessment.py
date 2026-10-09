"""Plot completed ID A/B/C means with seed SD, keeping mask change explicit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.table)
    frame = frame[frame.split_column == "id_split"]
    models = ["composition-ridge", "peak-xgboost", "resnet1d", "transformer", "physics-transformer"]
    labels = ["Composition\nRidge", "Peak\nXGBoost", "ResNet1D", "Transformer", "Physics\nTransformer"]
    if frame.duplicated(["model", "phase"]).any() or set(zip(frame.model, frame.phase)) != {(m, p) for m in models for p in "ABC"}:
        raise ValueError("Figure requires exactly the complete ID A/B/C table")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh figure directory")
    colors = dict(A="#666666", B="#0072B2", C="#D55E00")
    markers = dict(A="s", B="o", C="^")
    names = dict(A="A: old model / old labels", B="B: old model / corrected labels", C="C: retrained / corrected labels")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
        for ax, phases, title in zip(axes, [("A", "B"), ("B", "C")],
                                   ["a  Scoring-mask correction", "b  Retraining on corrected labels"]):
            ax.set_title(title, loc="left", fontsize=11, pad=12)
            for index, model in enumerate(models):
                points = frame[frame.model == model].set_index("phase")
                values = [points.loc[p, "mae_mean"] for p in phases]
                ax.plot([index-.1, index+.1], values, color="#bdbdbd", linewidth=1, zorder=1)
                for offset, phase in zip([-.1, .1], phases):
                    row = points.loc[phase]
                    ax.errorbar(index+offset, row.mae_mean, yerr=row.seed_sd, color=colors[phase],
                                marker=markers[phase], markersize=6, linewidth=1.2, capsize=3, linestyle="none", zorder=2)
            ax.set_xticks(range(len(models)), labels)
            ax.set_ylim(.28, .69)
            ax.set_xlim(-.5, len(models)-.5)
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Masked parent-macro MAE (lower is better)")
        handles = [Line2D([], [], color=colors[p], marker=markers[p], linestyle="none", label=names[p]) for p in "ABC"]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
        fig.text(.5, .045, "Clean simulated spectra, frozen ID split. Points: seed mean; bars: seed SD (5 runs; 1 deterministic Ridge).",
                 ha="center", fontsize=9)
        fig.text(.5, .012, "A and B use different valid-label masks; only B and C share the corrected evaluation target set.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.82, wspace=.12)
        fig.savefig(out / "sro_ID_ABC.png", dpi=300)
        fig.savefig(out / "sro_ID_ABC.pdf")
        plt.close(fig)
    frame.to_csv(out / "source_data.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        qualification="Provisional revision figure, not a venue-compliance assertion",
        uncertainty="Random-seed SD, not confidence intervals", y_axis_limits=[.28, .69],
        source_table=str(args.table.resolve()),
        alt_text="Two point plots show old predictions rescored with corrected labels, then models retrained using corrected labels. All five models are shown; scoring-mask changes are separate from retraining effects."), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
