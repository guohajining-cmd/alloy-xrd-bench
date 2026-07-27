"""Publication-oriented plots for HEA validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scoring import normalized_sro_score


MODEL_ORDER = ["composition-ridge", "peak-xgboost", "resnet1d", "transformer"]
MODEL_LABELS = {
    "composition-ridge": "Composition Ridge",
    "peak-xgboost": "Peak XGBoost",
    "resnet1d": "1D ResNet",
    "transformer": "Spectrum Transformer",
}
MODEL_COLORS = {
    "composition-ridge": "#6B7280",
    "peak-xgboost": "#0F766E",
    "resnet1d": "#D97706",
    "transformer": "#2563EB",
}


def _ordered_models(aggregates: dict[str, object]) -> list[str]:
    return [name for name in MODEL_ORDER if name in aggregates] + sorted(
        set(aggregates) - set(MODEL_ORDER)
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_validation_plots(summary_path: Path, output_dir: Path) -> dict[str, str]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = summary["model_aggregates"]
    models = _ordered_models(aggregates)
    baseline_mae = float(aggregates["composition-ridge"]["sro_parent_macro_mae_mean"])

    score_rows = []
    for model in models:
        values = aggregates[model]
        mae = float(values["sro_parent_macro_mae_mean"])
        score_rows.append(
            {
                "model": model,
                "display_name": MODEL_LABELS.get(model, model),
                "seed_count": int(values["seed_count"]),
                "sro_parent_macro_mae_mean": mae,
                "sro_parent_macro_mae_std": float(values["sro_parent_macro_mae_std"]),
                "sro_rmse_mean": float(values["sro_rmse_mean"]),
                "sro_rmse_std": float(values["sro_rmse_std"]),
                "sro_skill_score": normalized_sro_score(mae, baseline_mae),
                "qualification": summary["result_qualification"],
            }
        )
    scores = pd.DataFrame(score_rows)
    scores_path = output_dir / "hea_model_scores.csv"
    scores.to_csv(scores_path, index=False)

    labels = scores["display_name"].tolist()
    colors = [MODEL_COLORS.get(model, "#7C3AED") for model in models]
    x = np.arange(len(models))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "sro_parent_macro_mae_mean", "sro_parent_macro_mae_std", "SRO parent-macro MAE", "Lower is better"),
        (axes[0, 1], "sro_skill_score", None, "SRO skill vs composition", "Higher is better (0-100)"),
        (axes[1, 0], "sro_rmse_mean", "sro_rmse_std", "SRO RMSE", "Lower is better"),
    ]
    for axis, value_column, error_column, title, subtitle in panels:
        values = scores[value_column].to_numpy(dtype=float)
        errors = scores[error_column].to_numpy(dtype=float) if error_column else None
        bars = axis.bar(x, values, yerr=errors, capsize=4, color=colors, width=0.68)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(subtitle)
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            text = f"{value:.1f}" if value_column == "sro_skill_score" else f"{value:.3f}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=9)
    axes[0, 1].set_ylim(0, max(30.0, scores["sro_skill_score"].max() * 1.22))

    split_axis = axes[1, 1]
    split_aggregates = summary.get("split_model_aggregates", {})
    if split_aggregates:
        split_names = list(split_aggregates)
        width = 0.8 / max(1, len(models))
        positions = np.arange(len(split_names))
        for index, model in enumerate(models):
            values = [
                split_aggregates.get(split, {}).get(model, {}).get(
                    "sro_parent_macro_mae_mean", np.nan
                )
                for split in split_names
            ]
            split_axis.bar(
                positions + (index - (len(models) - 1) / 2) * width,
                values,
                width=width,
                color=MODEL_COLORS.get(model, "#7C3AED"),
                label=MODEL_LABELS.get(model, model),
            )
        split_axis.set_xticks(positions, [name.replace("_split", "") for name in split_names], rotation=16, ha="right")
        split_axis.set_ylabel("Parent-macro MAE")
        split_axis.set_title("ID and OOD SRO performance", fontsize=12, fontweight="bold")
        split_axis.legend(frameon=False, fontsize=8)
    else:
        split_axis.axis("off")
        split_axis.text(
            0.5, 0.5,
            "Family and occupancy are\ndiagnostic tracks in v1.1",
            ha="center", va="center", fontsize=14, color="#4B5563",
        )
    split_axis.grid(axis="y", alpha=0.22)
    split_axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("SiteXRD-Bench HEA v1.1 SRO Core", fontsize=16, fontweight="bold")
    figure.text(0.5, 0.005, "Error bars: random-seed standard deviation", ha="center", color="#374151")
    comparison_path = output_dir / "hea_model_comparison.png"
    _save_figure(figure, comparison_path)

    seed_rows = pd.DataFrame(summary["models"])
    if "split_column" in seed_rows:
        seed_rows = seed_rows[seed_rows["split_column"] == "id_split"]
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    offsets = np.linspace(-0.18, 0.18, len(models))
    for offset, model in zip(offsets, models):
        subset = seed_rows[seed_rows["model"] == model].sort_values("seed")
        if subset.empty:
            continue
        positions = np.arange(len(subset)) + offset
        axis.scatter(
            positions,
            subset["sro_parent_macro_mae"],
            s=70,
            color=MODEL_COLORS.get(model, "#7C3AED"),
            label=MODEL_LABELS.get(model, model),
            zorder=3,
        )
    neural_seeds = sorted(
        seed_rows.loc[seed_rows["model"].isin(["resnet1d", "transformer"]), "seed"].unique()
    )
    axis.axhline(baseline_mae, color=MODEL_COLORS["composition-ridge"], linestyle="--", linewidth=1.4, label="Composition baseline")
    axis.set_xticks(np.arange(len(neural_seeds)), [str(seed) for seed in neural_seeds])
    axis.set_xlabel("Random seed")
    axis.set_ylabel("SRO parent-macro MAE (lower is better)")
    axis.set_title("Seed Stability on HEA ID-Test", fontsize=15, fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    stability_path = output_dir / "hea_seed_stability.png"
    _save_figure(figure, stability_path)

    gate_rows = [
        ("Parent-group integrity", 100.0 if summary["data_leakage_passed"] else 0.0, "PASS" if summary["data_leakage_passed"] else "FAIL"),
        ("Oracle / shuffle controls", 100.0 if summary["control_gate_passed"] else 0.0, "PASS" if summary["control_gate_passed"] else "FAIL"),
        ("Complete ID/OOD matrix", 100.0 if summary["ood_validation_status"] == "pass" else 0.0, "PASS" if summary["ood_validation_status"] == "pass" else "NOT RUN"),
        ("Matched perturbations", 100.0 if summary["perturbation_validation_status"] == "pass" else 0.0, "PASS" if summary["perturbation_validation_status"] == "pass" else "NOT RUN"),
        ("Family", 50.0, "DIAGNOSTIC"),
        ("Occupancy identifiability", 50.0, "DIAGNOSTIC"),
    ]
    gate_labels = [row[0] for row in gate_rows]
    gate_values = [row[1] for row in gate_rows]
    gate_status = [row[2] for row in gate_rows]
    gate_colors = ["#15803D" if status == "PASS" else "#B91C1C" if status == "FAIL" else "#6B7280" for status in gate_status]
    figure, axis = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    positions = np.arange(len(gate_rows))
    bars = axis.barh(positions, gate_values, color=gate_colors, height=0.62)
    axis.set_yticks(positions, gate_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 112)
    axis.set_xlabel("Gate progress (%)")
    axis.set_title("HEA v1.1 SRO-Core Readiness", fontsize=15, fontweight="bold")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for bar, value, status in zip(bars, gate_values, gate_status):
        axis.text(max(value + 1.5, 2.0), bar.get_y() + bar.get_height() / 2, f"{status}  {value:.1f}%", va="center", fontweight="bold", fontsize=9)
    gates_path = output_dir / "hea_validation_gates.png"
    _save_figure(figure, gates_path)

    return {
        "scores_csv": str(scores_path.resolve()),
        "comparison_png": str(comparison_path.resolve()),
        "stability_png": str(stability_path.resolve()),
        "gates_png": str(gates_path.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(create_validation_plots(args.summary, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
