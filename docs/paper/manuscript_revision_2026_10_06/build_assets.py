"""Bind manuscript tables and fixed-example figures to the corrected evidence."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench import dataset_paper_visuals as visuals

OUT = ROOT / "docs/paper/generated/revision_2026_10_06"
SPLITS = {
    "id_split": "ID",
    "composition_ood_split": "Composition OOD",
    "chemistry_ood_split": "Chemistry OOD",
    "sro_ood_split": "Legacy SRO OOD",
}
MODELS = {
    "composition-ridge": "Composition Ridge",
    "peak-xgboost": "Peak XGBoost",
    "resnet1d": "ResNet1D",
    "transformer": "Spectrum Transformer",
    "physics-transformer": "Physics Transformer (exploratory)",
}
SOURCES: dict[str, str] = {}


def source(relative: str) -> Path:
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def table(name: str, spec: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [r"\small\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")


def tables() -> None:
    df = pd.read_csv(source("output/rebuttal_2026_10_02/full_ABC_summary/leaderboard.csv"))
    df = df.loc[df.phase.eq("C")].copy()
    assert len(df) == 20 and not df.duplicated(["model", "split_column"]).any()
    df.to_csv(OUT / "corrected_hea_results.csv", index=False)
    formal = list(MODELS)[:4]
    rows = []
    for model in formal:
        ranks = [df.loc[df.split_column.eq(split) & df.model.isin(formal)].sort_values("mae_mean").model.tolist().index(model) for split in SPLITS]
        name = MODELS[model]
        if all(rank == 0 for rank in ranks):
            name = r"\textbf{" + name + "}"
        elif all(rank == 1 for rank in ranks):
            name = r"\underline{" + name + "}"
        cells = [name]
        for split in SPLITS:
            subset = df.loc[df.split_column.eq(split) & df.model.isin(formal)]
            row = subset.loc[subset.model.eq(model)].iloc[0]
            rank = subset.sort_values("mae_mean").model.tolist().index(model)
            number = f"{row.mae_mean:.4f}"
            if rank < 2:
                number = (r"\textbf{" if rank == 0 else r"\underline{") + number + "}"
            cells.append(number if row.seed_count == 1 else number + r" $\pm$ " + f"{row.seed_sd:.4f}")
        rows.append(cells)
    table("hea_primary.tex", "lrrrr", ["Model", *SPLITS.values()], rows)
    rows = []
    for split, label in SPLITS.items():
        for model, name in MODELS.items():
            row = df.loc[df.split_column.eq(split) & df.model.eq(model)].iloc[0]
            mae = f"{row.mae_mean:.4f}"
            if model == "physics-transformer" and split in ("chemistry_ood_split", "sro_ood_split"):
                mae = r"\textbf{" + mae + "}"
            rows.append([label, name, mae, f"{row.rmse_mean:.4f}",
                         f"{row.shell1_spearman_mean:.3f}", f"{row.shell2_spearman_mean:.3f}",
                         f"{row.S_SRO:.2f}"])
    table("hea_extended.tex", "llrrrrr", ["Split", "Model", "MAE", "RMSE", r"$\rho_1$", r"$\rho_2$", r"$S_{\mathrm{SRO}}$"], rows)
    controls = json.loads(source("output/rebuttal_2026_10_02/signal_full_matrix_independent/comparisons.json").read_text())
    x_controls = {r["split"]: r for r in controls if r["candidate"] == "control-xgboost-X" and r["reference"] == "control-xgboost-C"}
    controls = [r for r in controls if r["candidate"] == "control-xgboost-CX" and r["reference"] == "control-xgboost-C"]
    assert len(controls) == 4
    rows = []
    for split, label in SPLITS.items():
        row, = [r for r in controls if r["split"] == split]
        assert np.isclose(x_controls[split]["reference_mae"], row["reference_mae"])
        lo, hi = row["delta_ci95_parent"]
        rows.append([label, f"{row['reference_mae']:.4f}", f"{x_controls[split]['candidate_mae']:.4f}", f"{row['candidate_mae']:.4f}",
                     f"{row['delta_mae']:.4f}", f"[{lo:.4f}, {hi:.4f}]", f"{row['relative_reduction_percent']:.2f}\\%"])
    table("signal_controls.tex", "lrrrrrr", ["Split", "C", "X", "C+X", r"$\Delta$ MAE", "Paired 95\\% interval", "Reduction"], rows)
    (OUT / "signal_controls.json").write_text(json.dumps(controls, indent=2), encoding="utf-8")
    fixed = json.loads(source("output/rebuttal_2026_10_02/fixed_cell_full_matrix_independent/comparisons.json").read_text())
    rows = []
    for split, label in SPLITS.items():
        for model in ("control-xgboost-CX", "transformer"):
            row, = [r for r in fixed if r["split"] == split and r["model"] == model]
            lo, hi = row["delta_ci95_parent"]
            rows.append([label, "C+X" if model.endswith("CX") else "Transformer", str(row["parent_count"]),
                         f"{row['relative_reduction_percent']:.2f}\\%", f"[{lo:.4f}, {hi:.4f}]"])
    table("fixed_cell.tex", "llrrr", ["Split", "Predictor", "Parents", "Reduction", r"95\% interval, $\Delta$ MAE"], rows)
    source("output/rebuttal_2026_10_01/corrected_ID/config.json")
    source("output/rebuttal_2026_10_02/signal_controls_20261002/protocol.json")
    source("output/rebuttal_2026_10_02/perturbation_controls_independent/comparisons.json")


def heatmap(fig, axis, values, title):
    limit = max(float(np.nanmax(np.abs(values))), 1e-12)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d4d4d4")
    image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    axis.set(title=title, xlabel="Species-pair target (8 x 8)", ylabel="Shell")
    axis.set_yticks([0, 1], ["1", "2"])
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)


def figures() -> dict:
    old = json.loads(source("docs/paper/generated/dataset_visual_examples.json").read_text())
    selected = old["hea_sro"]
    hea_root = ROOT / "data/benchmarks/siteXRD-bench-v1_1/hea/benchmark_v1"
    task = pd.read_parquet(source(str((hea_root / "manifests/task_samples.parquet").relative_to(ROOT))))
    labels = pd.read_parquet(source("output/rebuttal_2026_10_01/label_revision/labels.parquet")).set_index("structure_state_id", verify_integrity=True)
    source("output/rebuttal_2026_10_01/label_revision/revision.json")
    payload_path = hea_root / "xrd/runs/hea_relaxed_clean_xrd_v1/structure_payloads.parquet"
    payload = pd.read_parquet(source(str(payload_path.relative_to(ROOT)))).set_index("source_row", verify_integrity=True)
    structures, spectra, vectors = [], [], []
    grid = None
    for side in ("left", "right"):
        state = selected[f"{side}_structure_state_id"]
        sample = selected[f"{side}_sample_id"]
        row = task.loc[task.xrd_sample_id.eq(sample)].iloc[0]
        assert row.id_split == "train" and row.structure_state_id == state
        spectrum, this_grid = visuals._read_hea_grid(str(hea_root / row.grid_shard_path), int(row.grid_row_index), sample)
        source(str((hea_root / row.grid_shard_path).relative_to(ROOT)))
        if grid is not None:
            assert np.array_equal(grid, this_grid)
        grid = this_grid
        spectra.append(spectrum)
        structures.append(Structure.from_dict(ast.literal_eval(str(payload.loc[selected["structure_source_rows"][state], "structure_as_dict"]))))
        shell_values = [json.loads(labels.loc[state, f"sro_shell{s}_json"]) for s in (1, 2)]
        vectors.append(np.array([[shell.get(f"{a}->{b}", np.nan) for a in visuals.ELEMENTS for b in visuals.ELEMENTS] for shell in shell_values]))
    delta = vectors[1] - vectors[0]
    distance = float(np.nansum(np.abs(delta)))
    cosine = float(np.dot(*spectra) / (np.linalg.norm(spectra[0]) * np.linalg.norm(spectra[1])))
    assert np.isclose(distance, 26.66666666666667)
    assert np.isclose(cosine, selected["cosine_similarity"])
    fig = plt.figure(figsize=(12.8, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    for i in range(2):
        visuals._plot_structure(fig.add_subplot(gs[0, i]), structures[i], f"Structure {'AB'[i]}")
    axis = fig.add_subplot(gs[0, 2:])
    for i, color in enumerate(("#2563EB", "#DC2626")):
        visuals._plot_pattern(axis, grid, spectra[i], "AB"[i], color)
    axis.set_title(f"Same composition/topology; cosine {cosine:.5f}")
    axis.legend(frameon=False)
    center = int(np.argmax(np.abs(spectra[0] - spectra[1])))
    local = slice(max(0, center - 80), min(len(grid), center + 81))
    axis = fig.add_subplot(gs[1, :2])
    for i, color in enumerate(("#2563EB", "#DC2626")):
        visuals._plot_pattern(axis, grid[local], spectra[i][local], "AB"[i], color)
    axis.set_title("Largest local spectral difference")
    axis.legend(frameon=False)
    heatmap(fig, fig.add_subplot(gs[1, 2:]), delta, f"Corrected SRO(B - A), L1 = {distance:.2f}")
    fig.suptitle("Fixed released example; corrected labels (gray = undefined)", fontsize=13)
    visuals._save_figure(fig, OUT / "hea_sro_example.png")

    # Reconstruct the same released Gamma window from its full-pattern source.
    core = ROOT / "data/releases/gamma-decomp-v1_1/gamma-decomp-core-v1.1"
    gamma = old["gamma_decomp"]
    window = source(str((core / gamma["severe"]["pattern_path"]).relative_to(ROOT)))
    with np.load(window) as doc:
        pattern = {k: doc[k] for k in ("two_theta", "clean", "observed")}
    full_samples = pd.read_parquet(source(str((core / "full/samples.parquet").relative_to(ROOT))))
    full_id = gamma["full_source_sample_ids"]["severe"]
    path = core / full_samples.loc[full_samples.sample_id.eq(full_id), "pattern_path"].iloc[0]
    with np.load(source(str(path.relative_to(ROOT)))) as doc:
        full_grid = doc["two_theta"]
    components = pd.read_parquet(source(str((core / "full/components.parquet").relative_to(ROOT))))
    components = components.loc[components.sample_id.eq(full_id)]
    rebuilt, curves = visuals.project_gamma_components_from_full_grid(pattern["two_theta"], full_grid, components)
    assert np.allclose(rebuilt, pattern["clean"], rtol=1e-6, atol=1e-6)
    pattern["component_curves"] = curves
    fig = plt.figure(figsize=(13.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=(0.8, 1.3, 0.8, 1.3))
    visuals._plot_structure(fig.add_subplot(gs[:, 0]), structures[0], "HEA structure")
    axis = fig.add_subplot(gs[0, 1])
    visuals._plot_pattern(axis, grid, spectra[0], "clean XRD", "#2563EB")
    axis.set_title("Indexed reflections to XRD")
    heatmap(fig, fig.add_subplot(gs[1, 1]), vectors[0], "Corrected SRO; gray = undefined")
    visuals._plot_gamma_prototype(fig.add_subplot(gs[:, 2]), "gamma_prime", r"L1$_2$ phase prototype")
    visuals._plot_gamma_components(fig.add_subplot(gs[:, 3]), pattern, components, "Released component profiles")
    fig.suptitle("Alloy-XRD-Bench: structure, computational targets, and observed profiles", fontsize=13)
    visuals._save_figure(fig, OUT / "dataset_overview.png")
    return {"pair": selected, "selection_policy": "Keep original released pair; do not reselect after correction.",
            "corrected_sro_l1_distance": distance, "cosine_similarity": cosine,
            "common_defined_targets": int(np.isfinite(delta).sum()),
            "masked_targets": "gray, not zero-filled; symmetric color limits cover actual values"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source(str(Path(__file__).relative_to(ROOT)))
    source("alloy_xrd_bench/dataset_paper_visuals.py")
    tables()
    figures_manifest = figures()
    products = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in (".tex", ".csv", ".png", ".json") and p.name != "evidence_manifest.json"}
    (OUT / "evidence_manifest.json").write_text(json.dumps({"sources": SOURCES, "products": products, "figures": figures_manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"tables": 4, "figures": 2, "corrected_pair_distance": figures_manifest["corrected_sro_l1_distance"]}))


if __name__ == "__main__":
    main()
