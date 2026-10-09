"""Generate traceable dataset-first figures for the Alloy-XRD-Bench paper."""

from __future__ import annotations

import argparse
import ast
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image
from pymatgen.core import Structure

from .gamma_overlap_physics import ComponentSpec, render_components
from .hea_validation_data import load_validation_rows


ELEMENTS = ("Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si")
ELEMENT_COLORS = {
    "Al": "#7C8798", "Co": "#2563EB", "Cr": "#16A34A", "Cu": "#D97706",
    "Fe": "#B91C1C", "Mn": "#7C3AED", "Ni": "#0891B2", "Si": "#DB2777",
}
MODEL_LABELS = {
    "composition-ridge": "Composition Ridge",
    "peak-xgboost": "Peak XGBoost",
    "resnet1d": "ResNet1D",
    "transformer": "Spectrum Transformer",
}
VISUAL_SCHEMA_VERSION = "sitexrd-dataset-paper-visuals-v1.1"
HEA_MIN_COSINE_SIMILARITY = 0.995


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("spectra must be one-dimensional arrays with matching shapes")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("spectra must have non-zero norm")
    return float(np.dot(left, right) / denominator)


def select_hea_pair(
    candidates: pd.DataFrame,
    spectra: Mapping[str, np.ndarray],
    *,
    min_cosine_similarity: float = 0.995,
) -> dict[str, object]:
    """Select the largest-SRO-distance train pair with matched composition/topology."""

    required = {
        "sample_id", "structure_state_id", "split", "composition_key",
        "parent_topology", "sro_vector",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"HEA candidates missing columns: {sorted(missing)}")
    train = candidates.loc[candidates["split"].eq("train")].copy()
    accepted: list[dict[str, object]] = []
    for _, group in train.groupby(["composition_key", "parent_topology"], sort=True):
        rows = sorted(group.to_dict(orient="records"), key=lambda row: str(row["sample_id"]))
        for left, right in combinations(rows, 2):
            left_id = str(left["sample_id"])
            right_id = str(right["sample_id"])
            if left_id not in spectra or right_id not in spectra:
                raise ValueError(f"missing spectrum for HEA candidate pair {left_id}, {right_id}")
            cosine = _cosine_similarity(spectra[left_id], spectra[right_id])
            if cosine < min_cosine_similarity:
                continue
            left_sro = np.asarray(left["sro_vector"], dtype=float)
            right_sro = np.asarray(right["sro_vector"], dtype=float)
            if left_sro.shape != right_sro.shape or left_sro.ndim != 1:
                raise ValueError("SRO vectors must be one-dimensional arrays with matching shapes")
            valid = np.isfinite(left_sro) & np.isfinite(right_sro)
            if not valid.any():
                continue
            accepted.append({
                "left_sample_id": left_id,
                "right_sample_id": right_id,
                "left_structure_state_id": str(left["structure_state_id"]),
                "right_structure_state_id": str(right["structure_state_id"]),
                "composition_key": str(left["composition_key"]),
                "parent_topology": str(left["parent_topology"]),
                "split": "train",
                "cosine_similarity": cosine,
                "sro_l1_distance": float(np.abs(left_sro[valid] - right_sro[valid]).sum()),
            })
    if not accepted:
        raise ValueError(
            f"no same-composition train pair meets cosine similarity {min_cosine_similarity:.3f}"
        )
    return sorted(
        accepted,
        key=lambda row: (
            -float(row["sro_l1_distance"]),
            str(row["left_sample_id"]),
            str(row["right_sample_id"]),
        ),
    )[0]


def select_hea_pair_lazy(
    candidates: pd.DataFrame,
    spectrum_loader: Any,
    *,
    min_cosine_similarity: float = 0.995,
) -> dict[str, object]:
    """Rank pairs by SRO distance, then load spectra only until one qualifies."""

    required = {
        "sample_id", "structure_state_id", "split", "composition_key",
        "parent_topology", "sro_vector",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"HEA candidates missing columns: {sorted(missing)}")
    train = candidates.loc[candidates["split"].eq("train")].copy()
    ranked: list[tuple[float, str, str, dict[str, Any], dict[str, Any]]] = []
    for _, group in train.groupby(["composition_key", "parent_topology"], sort=True):
        rows = sorted(group.to_dict(orient="records"), key=lambda row: str(row["sample_id"]))
        if len(rows) < 2:
            continue
        values = np.stack([np.asarray(row["sro_vector"], dtype=float) for row in rows])
        difference = np.abs(values[:, None, :] - values[None, :, :])
        distances = np.nansum(difference, axis=2)
        valid_counts = np.isfinite(difference).sum(axis=2)
        left_indices, right_indices = np.triu_indices(len(rows), k=1)
        for left_index, right_index in zip(left_indices, right_indices):
            if valid_counts[left_index, right_index] == 0:
                continue
            left = rows[int(left_index)]
            right = rows[int(right_index)]
            ranked.append((
                float(distances[left_index, right_index]),
                str(left["sample_id"]),
                str(right["sample_id"]),
                left,
                right,
            ))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    for sro_distance, left_id, right_id, left, right in ranked:
        cosine = _cosine_similarity(
            np.asarray(spectrum_loader(left_id), dtype=float),
            np.asarray(spectrum_loader(right_id), dtype=float),
        )
        if cosine < min_cosine_similarity:
            continue
        return {
            "left_sample_id": left_id,
            "right_sample_id": right_id,
            "left_structure_state_id": str(left["structure_state_id"]),
            "right_structure_state_id": str(right["structure_state_id"]),
            "composition_key": str(left["composition_key"]),
            "parent_topology": str(left["parent_topology"]),
            "split": "train",
            "cosine_similarity": cosine,
            "sro_l1_distance": sro_distance,
        }
    raise ValueError(
        f"no same-composition train pair meets cosine similarity {min_cosine_similarity:.3f}"
    )


def select_gamma_examples(candidates: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Choose median representative resolved and severe examples from train."""

    required = {"sample_id", "split", "difficulty", "component_count", "snr", "phase_weight"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Gamma candidates missing columns: {sorted(missing)}")
    selected: dict[str, dict[str, object]] = {}
    for difficulty in ("resolved", "severe"):
        rows = candidates.loc[
            candidates["split"].eq("train") & candidates["difficulty"].eq(difficulty)
        ].copy()
        if difficulty == "severe" and "scored_component_count" in rows.columns:
            rows = rows.loc[rows["scored_component_count"].astype(int).ge(2)].copy()
        if rows.empty:
            raise ValueError(f"no train samples for Gamma difficulty {difficulty!r}")
        metrics = ["component_count", "snr", "phase_weight"]
        medians = rows[metrics].median(axis=0)
        scales = (rows[metrics] - medians).abs().median(axis=0).replace(0.0, 1.0)
        rows["selection_distance"] = ((rows[metrics] - medians).abs() / scales).sum(axis=1)
        rows = rows.sort_values(["selection_distance", "sample_id"], kind="stable")
        selected[difficulty] = rows.iloc[0].to_dict()
    return selected


def reconstruct_gamma_components(
    two_theta: np.ndarray,
    components: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Render component rows with the benchmark's area-normalized profile."""

    required = {"center_deg", "fwhm_deg", "integrated_intensity", "eta"}
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(f"Gamma components missing columns: {sorted(missing)}")
    specs = [
        ComponentSpec(
            phase=str(row.get("phase", "unknown")),
            hkl=(int(row.get("h", 0)), int(row.get("k", 0)), int(row.get("l", 1))),
            center_deg=float(row["center_deg"]),
            fwhm_deg=float(row["fwhm_deg"]),
            integrated_intensity=float(row["integrated_intensity"]),
            eta=float(row["eta"]),
        )
        for row in components.to_dict(orient="records")
    ]
    return render_components(np.asarray(two_theta, dtype=float), specs)


def project_gamma_components_from_full_grid(
    window_two_theta: np.ndarray,
    full_two_theta: np.ndarray,
    components: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Render area-normalized full-pattern components and crop to a public window."""

    window_grid = np.asarray(window_two_theta, dtype=float)
    full_grid = np.asarray(full_two_theta, dtype=float)
    indices = np.searchsorted(full_grid, window_grid)
    if (
        np.any(indices >= len(full_grid))
        or not np.allclose(full_grid[indices], window_grid, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("Gamma window grid is not an exact subset of its full-pattern grid")
    full_total, full_curves = reconstruct_gamma_components(full_grid, components)
    return full_total[indices], full_curves[:, indices]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _hea_source_root(release_root: Path) -> Path:
    provenance_path = _required_file(
        release_root / "release" / "source_provenance.json", "HEA source provenance"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_root_name = Path(str(provenance["source_dataset_root"])).name
    source_root = (release_root.parent / source_root_name).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"HEA local source dataset root not found: {source_root}")
    return source_root


def _read_hea_data(root: Path) -> dict[str, Any]:
    manifests = root / "manifests"
    task_path = _required_file(manifests / "task_samples.parquet", "HEA manifest")
    composition_path = _required_file(
        manifests / "composition_only_features.parquet", "HEA composition manifest"
    )
    structures_path = _required_file(manifests / "structures.parquet", "HEA structure manifest")
    task_frame = pd.read_parquet(task_path)
    candidate_fidelity = (
        "source_relaxed_dft" if "fidelity" in task_frame.columns else "all_manifest_rows"
    )
    eligible_sample_ids = None
    if candidate_fidelity == "source_relaxed_dft":
        eligible_sample_ids = set(
            task_frame.loc[
                task_frame["fidelity"].astype(str).eq(candidate_fidelity), "xrd_sample_id"
            ].astype(str)
        )
    config_path = Path(__file__).resolve().parents[1] / "configs" / "hea_validation_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_validation_rows(root, config)
    candidates: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    two_theta: np.ndarray | None = None
    local_source_root: Path | None = None
    for row in rows:
        if row.splits["id_split"] != "train":
            continue
        if eligible_sample_ids is not None and row.xrd_sample_id not in eligible_sample_ids:
            continue
        grid_shard_path = Path(row.grid_shard_path).resolve()
        if not grid_shard_path.is_file():
            if local_source_root is None:
                local_source_root = _hea_source_root(root)
            relative_grid_path = grid_shard_path.relative_to(root)
            grid_shard_path = _required_file(
                local_source_root / relative_grid_path,
                f"HEA source XRD shard {relative_grid_path}",
            )
        composition_key = "|".join(f"{element}:{value:.8f}" for element, value in zip(ELEMENTS, row.composition))
        candidates.append({
            "sample_id": row.xrd_sample_id,
            "structure_state_id": row.structure_state_id,
            "split": row.splits["id_split"],
            "composition_key": composition_key,
            "parent_topology": row.family,
            "sro_vector": np.where(row.sro_mask, row.sro_values, np.nan),
            "composition": row.composition,
            "splits": dict(row.splits),
            "grid_shard_path": str(grid_shard_path),
            "grid_row_index": int(row.grid_row_index),
        })
    if not candidates:
        raise ValueError("HEA manifest contains no public training samples")
    frame = pd.DataFrame(candidates)
    candidate_index = frame.set_index("sample_id", verify_integrity=True)

    def load_spectrum(sample_id: str) -> np.ndarray:
        nonlocal two_theta
        if sample_id not in spectra:
            record = candidate_index.loc[sample_id]
            spectrum, grid = _read_hea_grid(
                str(record["grid_shard_path"]), int(record["grid_row_index"]), sample_id
            )
            spectra[sample_id] = spectrum
            if two_theta is None:
                two_theta = grid
            elif not np.array_equal(two_theta, grid):
                raise ValueError(f"HEA selected spectra use different two-theta grids: {sample_id}")
        return spectra[sample_id]

    selected = select_hea_pair_lazy(
        frame, load_spectrum, min_cosine_similarity=HEA_MIN_COSINE_SIMILARITY
    )
    load_spectrum(str(selected["left_sample_id"]))
    load_spectrum(str(selected["right_sample_id"]))
    if two_theta is None:
        raise ValueError("HEA selected pair has no XRD grid")
    selected_rows = frame.set_index("sample_id").loc[
        [selected["left_sample_id"], selected["right_sample_id"]]
    ]
    grid_shard_paths = sorted(
        {Path(value).resolve() for value in selected_rows["grid_shard_path"].tolist()},
        key=str,
    )
    structures = pd.read_parquet(structures_path)
    selected_state_ids = (
        str(selected["left_structure_state_id"]),
        str(selected["right_structure_state_id"]),
    )
    structure_paths: dict[str, Path] = {}
    structure_source_rows: dict[str, int] = {}
    loaded_structures: dict[str, Structure] = {}
    for state_id in selected_state_ids:
        matches = structures.loc[structures["dataset_structure_id"].astype(str).eq(str(state_id))]
        if len(matches) != 1:
            raise ValueError(f"HEA structure_state_id must resolve exactly once: {state_id}")
    artifact_column = next(
        (column for column in ("artifact_path_json", "artifact_path") if column in structures),
        None,
    )
    if artifact_column is not None:
        for state_id in selected_state_ids:
            match = structures.loc[
                structures["dataset_structure_id"].astype(str).eq(state_id)
            ].iloc[0]
            path = _required_file(
                root / str(match[artifact_column]), f"HEA structure artifact {state_id}"
            )
            structure_paths[state_id] = path
            loaded_structures[state_id] = Structure.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
    else:
        if "source_row" not in structures:
            raise ValueError("HEA structures manifest has neither artifact path nor source_row")
        payload_path = _required_file(
            (local_source_root or _hea_source_root(root))
            / "xrd"
            / "runs"
            / "hea_relaxed_clean_xrd_v1"
            / "structure_payloads.parquet",
            "HEA structure payload table",
        )
        payloads = pd.read_parquet(payload_path, columns=["source_row", "structure_as_dict"])
        if payloads["source_row"].duplicated().any():
            raise ValueError("HEA structure payload table contains duplicate source_row")
        payloads = payloads.set_index("source_row")
        for state_id in selected_state_ids:
            match = structures.loc[
                structures["dataset_structure_id"].astype(str).eq(state_id)
            ].iloc[0]
            source_row = int(match["source_row"])
            if source_row not in payloads.index:
                raise ValueError(f"HEA structure payload missing source_row {source_row}")
            structure_doc = ast.literal_eval(str(payloads.loc[source_row, "structure_as_dict"]))
            if not isinstance(structure_doc, dict):
                raise ValueError(f"HEA structure payload is not a mapping for source_row {source_row}")
            structure_paths[state_id] = payload_path
            structure_source_rows[state_id] = source_row
            loaded_structures[state_id] = Structure.from_dict(structure_doc)
    return {
        "task_path": task_path,
        "composition_path": composition_path,
        "structures_path": structures_path,
        "task_frame": task_frame,
        "candidate_fidelity": candidate_fidelity,
        "candidate_frame": frame,
        "spectra": spectra,
        "two_theta": two_theta,
        "selected": selected,
        "structure_paths": structure_paths,
        "structure_source_rows": structure_source_rows,
        "grid_shard_paths": grid_shard_paths,
        "structures": loaded_structures,
    }


def _read_hea_grid(
    path_value: str, row_index: int, expected_sample_id: str
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_value)
    with h5py.File(path, "r") as handle:
        raw_id = handle["xrd_sample_id"][row_index]
        actual_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        if actual_id != expected_sample_id:
            raise ValueError(
                f"HEA grid sample mismatch: expected {expected_sample_id!r}, found {actual_id!r}"
            )
        spectrum = np.asarray(handle["intensity"][row_index], dtype=float)
        grid = np.asarray(handle["two_theta"][:], dtype=float)
    if spectrum.shape != grid.shape or spectrum.ndim != 1:
        raise ValueError(f"HEA spectrum and grid shapes differ for {expected_sample_id}")
    return spectrum, grid


def _read_gamma_data(root: Path) -> dict[str, Any]:
    core = root / "gamma-decomp-core-v1.1"
    provenance = root / "gamma-decomp-provenance-v1.1"
    sample_path = _required_file(core / "window" / "samples.parquet", "Gamma window manifest")
    component_path = _required_file(core / "window" / "components.parquet", "Gamma component manifest")
    slice_path = _required_file(core / "slices" / "overlap_slices.parquet", "Gamma overlap manifest")
    samples = pd.read_parquet(sample_path)
    components = pd.read_parquet(component_path)
    slices = pd.read_parquet(slice_path)
    full_sample_path = _required_file(core / "full" / "samples.parquet", "Gamma full manifest")
    full_component_path = _required_file(
        core / "full" / "components.parquet", "Gamma full component manifest"
    )
    full_samples = pd.read_parquet(full_sample_path)
    full_components = pd.read_parquet(full_component_path)
    edge_path = provenance / "provenance_edges.parquet"
    edges = pd.read_parquet(edge_path) if edge_path.is_file() else pd.DataFrame()
    train = slices.loc[slices["split"].eq("train")].merge(
        samples[["sample_id", "pattern_path"]], on="sample_id", validate="one_to_one"
    )
    component_groups = {str(key): value for key, value in components.groupby("sample_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for record in train.to_dict(orient="records"):
        sample_id = str(record["sample_id"])
        sample_components = component_groups.get(sample_id)
        if sample_components is None or sample_components.empty:
            continue
        pattern_path = core / str(record["pattern_path"])
        with np.load(pattern_path) as artifact:
            clean = np.asarray(artifact["clean"], dtype=float)
            observed = np.asarray(artifact["observed"], dtype=float)
        noise = float(np.std(observed - clean))
        snr = float(np.max(clean) / max(noise, 1e-12))
        total_area = float(sample_components["integrated_intensity"].sum())
        gamma_prime_area = float(
            sample_components.loc[
                sample_components["phase"].eq("gamma_prime"), "integrated_intensity"
            ].sum()
        )
        rows.append({
            **record,
            "component_count": int(record.get("scored_component_count", len(sample_components))),
            "manifest_component_row_count": int(len(sample_components)),
            "snr": snr,
            "phase_weight": gamma_prime_area / max(total_area, 1e-12),
        })
    candidate_frame = pd.DataFrame(rows)
    selected = select_gamma_examples(candidate_frame)
    selected_patterns: dict[str, dict[str, np.ndarray]] = {}
    selected_pattern_paths: dict[str, Path] = {}
    selected_full_pattern_paths: dict[str, Path] = {}
    selected_source_sample_ids: dict[str, str] = {}
    selected_components: dict[str, pd.DataFrame] = {}
    for difficulty, record in selected.items():
        sample_id = str(record["sample_id"])
        sample_row = samples.loc[samples["sample_id"].astype(str).eq(sample_id)].iloc[0]
        pattern_path = core / str(sample_row["pattern_path"])
        selected_pattern_paths[difficulty] = pattern_path.resolve()
        with np.load(pattern_path) as artifact:
            selected_patterns[difficulty] = {
                key: np.asarray(artifact[key], dtype=float) for key in ("two_theta", "clean", "observed")
            }
        selected_components[difficulty] = component_groups[sample_id].copy()
        rebuilt, curves = reconstruct_gamma_components(
            selected_patterns[difficulty]["two_theta"], selected_components[difficulty]
        )
        if not np.allclose(rebuilt, selected_patterns[difficulty]["clean"], rtol=1e-6, atol=1e-6):
            if edges.empty:
                raise ValueError(f"Gamma components do not reconstruct clean pattern for {sample_id}")
            source_edges = edges.loc[
                edges["target_id"].astype(str).eq(sample_id)
                & edges["source_type"].astype(str).eq("mixture_sample")
                & edges["target_type"].astype(str).eq("overlap_window")
                & edges["relation_type"].astype(str).eq("cropped_as")
            ]
            if len(source_edges) != 1:
                raise ValueError(f"Gamma window must resolve to one full source: {sample_id}")
            source_sample_id = str(source_edges.iloc[0]["source_id"])
            full_sample = full_samples.loc[
                full_samples["sample_id"].astype(str).eq(source_sample_id)
            ]
            if len(full_sample) != 1:
                raise ValueError(f"Gamma full source must resolve exactly once: {source_sample_id}")
            full_pattern_path = core / str(full_sample.iloc[0]["pattern_path"])
            with np.load(full_pattern_path) as artifact:
                full_grid = np.asarray(artifact["two_theta"], dtype=float)
                full_clean = np.asarray(artifact["clean"], dtype=float)
            source_components = full_components.loc[
                full_components["sample_id"].astype(str).eq(source_sample_id)
            ].copy()
            full_rebuilt, _ = reconstruct_gamma_components(full_grid, source_components)
            if not np.allclose(full_rebuilt, full_clean, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Gamma full components do not reconstruct clean pattern: {source_sample_id}")
            rebuilt, curves = project_gamma_components_from_full_grid(
                selected_patterns[difficulty]["two_theta"], full_grid, source_components
            )
            if not np.allclose(rebuilt, selected_patterns[difficulty]["clean"], rtol=1e-6, atol=1e-6):
                raise ValueError(f"Gamma full-source projection does not reconstruct window: {sample_id}")
            selected_components[difficulty] = source_components
            selected_full_pattern_paths[difficulty] = full_pattern_path.resolve()
            selected_source_sample_ids[difficulty] = source_sample_id
        selected_patterns[difficulty]["component_curves"] = curves
    evaluation_groups_path = _required_file(
        core / "window" / "evaluation_groups.parquet", "Gamma evaluation groups"
    )
    evaluation_groups = pd.read_parquet(evaluation_groups_path)
    representative_id = str(selected["resolved"]["sample_id"])
    group = evaluation_groups.loc[evaluation_groups["sample_id"].astype(str).eq(representative_id)]
    if len(group) != 1:
        raise ValueError(f"Gamma sample must resolve to one parent pair: {representative_id}")
    pair_id = str(group.iloc[0]["parent_pair_id"])
    pairs_path = _required_file(provenance / "parent_pairs.parquet", "Gamma parent pairs")
    structures_path = _required_file(
        provenance / "parent_structures.parquet", "Gamma parent structures"
    )
    pairs = pd.read_parquet(pairs_path)
    parent_structures = pd.read_parquet(structures_path)
    pair = pairs.loc[pairs["parent_pair_id"].astype(str).eq(pair_id)]
    if len(pair) != 1:
        raise ValueError(f"Gamma parent_pair_id must resolve exactly once: {pair_id}")
    parent_ids = {
        "gamma": str(pair.iloc[0]["gamma_parent_id"]),
        "gamma_prime": str(pair.iloc[0]["gamma_prime_parent_id"]),
    }
    structure_paths: dict[str, Path] = {}
    structures: dict[str, Structure] = {}
    for phase, parent_id in parent_ids.items():
        match = parent_structures.loc[
            parent_structures["parent_structure_id"].astype(str).eq(parent_id)
        ]
        if len(match) != 1:
            raise ValueError(f"Gamma parent_structure_id must resolve exactly once: {parent_id}")
        path = _required_file(
            provenance / str(match.iloc[0]["artifact_path"]), f"Gamma structure artifact {parent_id}"
        )
        structure_paths[phase] = path
        structures[phase] = Structure.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return {
        "core": core,
        "provenance": provenance,
        "sample_path": sample_path,
        "component_path": component_path,
        "full_sample_path": full_sample_path,
        "full_component_path": full_component_path,
        "edge_path": edge_path if edge_path.is_file() else None,
        "slice_path": slice_path,
        "evaluation_groups_path": evaluation_groups_path,
        "pairs_path": pairs_path,
        "structures_path": structures_path,
        "samples": samples,
        "components": components,
        "slices": slices,
        "candidate_frame": candidate_frame,
        "selected": selected,
        "selected_patterns": selected_patterns,
        "selected_pattern_paths": selected_pattern_paths,
        "selected_full_pattern_paths": selected_full_pattern_paths,
        "selected_source_sample_ids": selected_source_sample_ids,
        "selected_components": selected_components,
        "structure_paths": structure_paths,
        "structures": structures,
        "parent_pair_id": pair_id,
        "parent_ids": parent_ids,
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.5)


def _plot_structure(axis: plt.Axes, structure: Structure, title: str) -> None:
    fractional = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    order = np.argsort(fractional[:, 2])
    depth_dx, depth_dy = 0.22, 0.16
    for shift in (0.0, 1.0):
        style = "-" if shift == 0.0 else ":"
        alpha = 0.42 if shift == 0.0 else 0.28
        for value in range(3):
            axis.plot(
                [value + depth_dx * shift, value + depth_dx * shift],
                [0 + depth_dy * shift, 2 + depth_dy * shift],
                color="#334155", linewidth=0.75, alpha=alpha, linestyle=style, zorder=0,
            )
            axis.plot(
                [0 + depth_dx * shift, 2 + depth_dx * shift],
                [value + depth_dy * shift, value + depth_dy * shift],
                color="#334155", linewidth=0.75, alpha=alpha, linestyle=style, zorder=0,
            )
        for corner_x, corner_y in [(0, 0), (2, 0), (2, 2), (0, 2)]:
            axis.plot(
                [corner_x, corner_x + depth_dx],
                [corner_y, corner_y + depth_dy],
                color="#334155", linewidth=0.55, alpha=0.24, zorder=0,
            )

    seen: set[str] = set()
    for tile_x in range(2):
        for tile_y in range(2):
            for index in order:
                symbol = structure[index].specie.symbol
                depth = fractional[index, 2]
                label = symbol if symbol not in seen else None
                seen.add(symbol)
                axis.scatter(
                    fractional[index, 0] + tile_x + depth_dx * depth,
                    fractional[index, 1] + tile_y + depth_dy * depth,
                    s=24 + 30 * depth,
                    color=ELEMENT_COLORS.get(symbol, "#64748B"), edgecolor="white", linewidth=0.38,
                    alpha=0.65 + 0.32 * depth, zorder=2 + depth, label=label,
                )
    symbols = sorted({site.specie.symbol for site in structure})
    legend_columns = min(4, max(1, len(symbols)))
    if symbols:
        axis.legend(
            frameon=False, fontsize=6, ncol=legend_columns, loc="upper center",
            bbox_to_anchor=(0.5, -0.02), handletextpad=0.25, columnspacing=0.55,
        )
    axis.set(xlim=(-0.12, 2.36), ylim=(-0.12, 2.28), aspect="equal", title=title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_gamma_prototype(axis: plt.Axes, phase: str, title: str) -> None:
    """Render compact prototype guides for gamma and gamma-prime phases."""

    depth_dx, depth_dy = 0.18, 0.13
    for shift in (0.0, 1.0):
        style = "-" if shift == 0.0 else ":"
        alpha = 0.42 if shift == 0.0 else 0.26
        for value in range(3):
            axis.plot(
                [value + depth_dx * shift, value + depth_dx * shift],
                [0 + depth_dy * shift, 2 + depth_dy * shift],
                color="#334155", linewidth=0.8, alpha=alpha, linestyle=style, zorder=0,
            )
            axis.plot(
                [0 + depth_dx * shift, 2 + depth_dx * shift],
                [value + depth_dy * shift, value + depth_dy * shift],
                color="#334155", linewidth=0.8, alpha=alpha, linestyle=style, zorder=0,
            )
        for corner_x, corner_y in [(0, 0), (2, 0), (2, 2), (0, 2)]:
            axis.plot(
                [corner_x, corner_x + depth_dx],
                [corner_y, corner_y + depth_dy],
                color="#334155", linewidth=0.55, alpha=0.23, zorder=0,
            )

    if phase == "gamma_prime":
        sites = [
            ("Al", (0.0, 0.0, 0.0), 56),
            ("Al", (1.0, 0.0, 0.0), 56),
            ("Al", (0.0, 1.0, 0.0), 56),
            ("Al", (1.0, 1.0, 0.0), 56),
            ("Al", (0.0, 0.0, 1.0), 42),
            ("Al", (1.0, 0.0, 1.0), 42),
            ("Al", (0.0, 1.0, 1.0), 42),
            ("Al", (1.0, 1.0, 1.0), 42),
            ("Ni", (0.5, 0.5, 0.0), 64),
            ("Ni", (0.5, 0.0, 0.5), 60),
            ("Ni", (0.0, 0.5, 0.5), 60),
            ("Ni", (0.5, 1.0, 0.5), 60),
            ("Ni", (1.0, 0.5, 0.5), 60),
            ("Ni", (0.5, 0.5, 1.0), 50),
        ]
        note = r"L1$_2$: Al corner sublattice, Ni face centers"
    else:
        fcc_basis = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ]
        sites = []
        for index, coord in enumerate(fcc_basis):
            symbol = "Al" if index in {2} else "Ni"
            sites.append((symbol, coord, 56 if coord[2] < 0.5 else 48))
        note = r"FCC $\gamma$: Ni-rich solid solution"

    seen: set[str] = set()
    for tile_x in range(2):
        for tile_y in range(2):
            for symbol, (x, y, z), size in sites:
                if x >= 1.0 and tile_x == 1:
                    continue
                if y >= 1.0 and tile_y == 1:
                    continue
                label = symbol if symbol not in seen else None
                seen.add(symbol)
                axis.scatter(
                    x + tile_x + depth_dx * z,
                    y + tile_y + depth_dy * z,
                    s=size,
                    color=ELEMENT_COLORS.get(symbol, "#64748B"),
                    edgecolor="white", linewidth=0.45,
                    alpha=0.66 + 0.28 * min(float(z), 1.0),
                    zorder=2 + float(z),
                    label=label,
                )
    axis.text(
        0.5, -0.17, note, transform=axis.transAxes,
        ha="center", va="top", fontsize=6.5, color="#334155",
    )
    axis.legend(
        frameon=False, fontsize=6, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.28), handletextpad=0.25, columnspacing=0.65,
    )
    axis.set(xlim=(-0.12, 2.32), ylim=(-0.12, 2.25), aspect="equal", title=title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_pattern(axis: plt.Axes, grid: np.ndarray, intensity: np.ndarray, label: str, color: str) -> None:
    scale = max(float(np.max(intensity)), 1e-12)
    axis.plot(grid, intensity / scale, label=label, color=color, linewidth=1.15)
    axis.set_ylim(bottom=-0.03)
    axis.set_xlabel(r"$2\theta$ (deg)")
    axis.set_ylabel("Normalized intensity")
    _style_axis(axis)


def _format_gamma_component_annotation(row: Mapping[str, Any]) -> str:
    phase = str(row["phase"]).replace("_", "-")
    hkl = f"({int(row['h'])}{int(row['k'])}{int(row['l'])})"
    group = str(row.get("hkl_group_id", "unknown"))
    return (
        f"{phase} {hkl} [group={group}]: center={float(row['center_deg']):.2f}, "
        f"FWHM={float(row['fwhm_deg']):.2f}, area={float(row['integrated_intensity']):.2g}"
    )


def _plot_gamma_components(
    axis: plt.Axes,
    pattern: Mapping[str, np.ndarray],
    components: pd.DataFrame,
    title: str,
) -> None:
    grid = pattern["two_theta"]
    curves = pattern.get("component_curves")
    if curves is None:
        _, curves = reconstruct_gamma_components(grid, components)
    curves = np.asarray(curves, dtype=float)
    colors = {"gamma": "#2563EB", "gamma_prime": "#DC2626"}
    used: set[str] = set()
    scale = max(float(pattern["observed"].max()), 1e-12)
    for curve, row in zip(curves, components.to_dict(orient="records")):
        phase = str(row["phase"])
        hkl = f"({int(row['h'])}{int(row['k'])}{int(row['l'])})"
        label = f"{phase.replace('_', '-')} {hkl}" if phase not in used else None
        used.add(phase)
        axis.plot(grid, curve / scale, color=colors.get(phase, "#64748B"), alpha=0.62, linewidth=0.8, label=label)
    axis.plot(grid, pattern["clean"] / scale, color="#111827", linewidth=1.2, label="clean sum")
    residual = pattern["observed"] - pattern["clean"]
    axis.plot(
        grid, residual / scale, color="#64748B", linewidth=0.75,
        alpha=0.8, label="background + noise",
    )
    axis.plot(grid, pattern["observed"] / scale, color="#F59E0B", linewidth=0.9, alpha=0.85, label="observed")
    axis.set(title=title, xlabel=r"$2\theta$ (deg)", ylabel="Normalized intensity")
    _style_axis(axis)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    visible = components.loc[
        components["center_deg"].astype(float).between(float(grid.min()), float(grid.max()))
    ]
    strongest = visible.sort_values("integrated_intensity", ascending=False).head(2)
    annotations: list[str] = []
    for row in strongest.to_dict(orient="records"):
        center = float(row["center_deg"])
        annotations.append(_format_gamma_component_annotation(row))
        axis.axvline(center, color="#475569", linewidth=0.55, linestyle=":")
    if annotations:
        axis.text(
            0.98, 0.03, "\n".join(annotations), transform=axis.transAxes,
            va="bottom", ha="right", fontsize=6, color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9, "pad": 2},
        )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path, dpi=240, facecolor="white", bbox_inches="tight",
        metadata={"Software": "Alloy-XRD-Bench"},
    )
    plt.close(figure)
    # Some submission-site PDF previewers mishandle PNG alpha masks embedded by
    # pdfTeX, exposing axis text or image coordinates without the raster layer.
    # Flatten to RGB with a white background while preserving the original
    # resolution and traceable generated image content.
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        rgb.save(path, optimize=True)


def _write_dataset_overview(hea: Mapping[str, Any], gamma: Mapping[str, Any], path: Path) -> None:
    figure = plt.figure(figsize=(13.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(0.8, 1.3, 0.8, 1.3))
    left_state = str(hea["selected"]["left_structure_state_id"])
    _plot_structure(figure.add_subplot(grid[:, 0]), hea["structures"][left_state], "HEA structure/order truth")
    hea_axis = figure.add_subplot(grid[0, 1])
    left_id = str(hea["selected"]["left_sample_id"])
    _plot_pattern(hea_axis, hea["two_theta"], hea["spectra"][left_id], "clean XRD", "#2563EB")
    hea_axis.set_title("HEA indexed reflections to XRD")
    sro_axis = figure.add_subplot(grid[1, 1])
    sro = np.asarray(
        hea["candidate_frame"].loc[
            hea["candidate_frame"]["sample_id"].eq(left_id), "sro_vector"
        ].iloc[0], dtype=float,
    )
    sro_axis.imshow(np.nan_to_num(sro).reshape(2, -1), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    sro_axis.set(title="First/second-shell SRO label", xlabel="Species-pair target", ylabel="Shell")
    sro_axis.set_yticks([0, 1], ["1", "2"])
    _plot_gamma_prototype(figure.add_subplot(grid[:, 2]), "gamma_prime", r"L1$_2$ gamma-prime prototype")
    _plot_gamma_components(
        figure.add_subplot(grid[:, 3]), gamma["selected_patterns"]["severe"],
        gamma["selected_components"]["severe"], "Phase/HKL components to observed XRD",
    )
    figure.suptitle("Alloy-XRD-Bench preserves the path from atomic structure to latent labels and observed XRD", fontsize=13)
    _save_figure(figure, path)


def _write_hea_example(hea: Mapping[str, Any], path: Path) -> None:
    figure = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    selected = hea["selected"]
    left_state = str(selected["left_structure_state_id"])
    right_state = str(selected["right_structure_state_id"])
    _plot_structure(figure.add_subplot(grid[0, 0]), hea["structures"][left_state], "Structure A")
    _plot_structure(figure.add_subplot(grid[0, 1]), hea["structures"][right_state], "Structure B")
    full = figure.add_subplot(grid[0, 2:])
    left_id = str(selected["left_sample_id"])
    right_id = str(selected["right_sample_id"])
    _plot_pattern(full, hea["two_theta"], hea["spectra"][left_id], "A", "#2563EB")
    _plot_pattern(full, hea["two_theta"], hea["spectra"][right_id], "B", "#DC2626")
    full.set_title(f"Same composition and topology; cosine similarity {selected['cosine_similarity']:.5f}")
    full.legend(frameon=False)
    difference = np.abs(hea["spectra"][left_id] - hea["spectra"][right_id])
    center_index = int(np.argmax(difference))
    half_window = 80
    start = max(0, center_index - half_window)
    stop = min(len(difference), center_index + half_window + 1)
    zoom = figure.add_subplot(grid[1, :2])
    _plot_pattern(zoom, hea["two_theta"][start:stop], hea["spectra"][left_id][start:stop], "A", "#2563EB")
    _plot_pattern(zoom, hea["two_theta"][start:stop], hea["spectra"][right_id][start:stop], "B", "#DC2626")
    zoom.set_title("Largest local spectral difference")
    zoom.legend(frameon=False)
    heat = figure.add_subplot(grid[1, 2:])
    frame = hea["candidate_frame"].set_index("sample_id")
    left_sro = np.asarray(frame.loc[left_id, "sro_vector"], dtype=float)
    right_sro = np.asarray(frame.loc[right_id, "sro_vector"], dtype=float)
    delta = np.nan_to_num(right_sro - left_sro).reshape(2, -1)
    image = heat.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    heat.set(title=f"SRO(B - A), L1 distance {selected['sro_l1_distance']:.3f}", xlabel="Species-pair target", ylabel="Shell")
    heat.set_yticks([0, 1], ["1", "2"])
    figure.colorbar(image, ax=heat, fraction=0.035, pad=0.02)
    figure.suptitle("HEA-SRO controls composition shortcuts while retaining subtle spectral evidence", fontsize=13)
    _save_figure(figure, path)


def _write_gamma_example(gamma: Mapping[str, Any], path: Path) -> None:
    figure = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(0.72, 1.4, 1.4))
    _plot_gamma_prototype(figure.add_subplot(grid[0, 0]), "gamma", r"FCC $\gamma$ prototype")
    _plot_gamma_prototype(figure.add_subplot(grid[1, 0]), "gamma_prime", r"L1$_2$ $\gamma^\prime$ prototype")
    _plot_gamma_components(
        figure.add_subplot(grid[:, 1]), gamma["selected_patterns"]["resolved"],
        gamma["selected_components"]["resolved"], "Representative resolved window",
    )
    _plot_gamma_components(
        figure.add_subplot(grid[:, 2]), gamma["selected_patterns"]["severe"],
        gamma["selected_components"]["severe"], "Representative severe-overlap window",
    )
    figure.suptitle("Gamma-Decomp stores component truth before phase/HKL profiles are mixed", fontsize=13)
    _save_figure(figure, path)


def _write_dataset_coverage(hea: Mapping[str, Any], gamma: Mapping[str, Any], path: Path) -> None:
    def _thousands(value: float, _position: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000
            return f"{scaled:.0f}k" if float(scaled).is_integer() else f"{scaled:.1f}k"
        return f"{value:.0f}"

    figure = plt.figure(figsize=(12.9, 7.2), constrained_layout=False)
    grid = figure.add_gridspec(
        2, 2,
        width_ratios=(1.0, 1.08),
        height_ratios=(1.0, 1.0),
        hspace=0.44,
        wspace=0.30,
    )
    chemistry_grid = grid[0, 0].subgridspec(
        2, 1,
        height_ratios=(0.12, 0.88),
        hspace=0.02,
    )
    chemistry_title = figure.add_subplot(chemistry_grid[0, 0])
    chemistry_title.set_axis_off()
    chemistry_title.text(
        0.0, 0.45, "(a) HEA chemistry coverage",
        ha="left", va="center", fontweight="bold", transform=chemistry_title.transAxes,
    )
    gamma_grid = grid[1, 0].subgridspec(
        3, 1,
        height_ratios=(0.14, 0.12, 0.74),
        hspace=0.02,
    )
    gamma_title = figure.add_subplot(gamma_grid[0, 0])
    gamma_title.set_axis_off()
    gamma_title.text(
        0.0, 0.55, "(c) Gamma overlap windows",
        ha="left", va="center", fontweight="bold", transform=gamma_title.transAxes,
    )
    gamma_legend_axis = figure.add_subplot(gamma_grid[1, 0])
    gamma_legend_axis.set_axis_off()
    axes = np.asarray([
        [figure.add_subplot(chemistry_grid[1, 0]), figure.add_subplot(grid[0, 1])],
        [figure.add_subplot(gamma_grid[2, 0]), figure.add_subplot(grid[1, 1])],
    ])
    compositions = np.vstack(hea["candidate_frame"]["composition"].to_numpy())
    presence = compositions > 0
    cooccurrence = presence.astype(int).T @ presence.astype(int)
    image = axes[0, 0].imshow(cooccurrence, cmap="Blues", aspect="equal")
    axes[0, 0].set_anchor("C")
    axes[0, 0].set_xticks(range(len(ELEMENTS)), ELEMENTS)
    axes[0, 0].set_yticks(range(len(ELEMENTS)), ELEMENTS)
    axes[0, 0].tick_params(axis="both", labelsize=8)
    for spine in axes[0, 0].spines.values():
        spine.set_visible(False)
    axes[0, 0].set_xticks(np.arange(-0.5, len(ELEMENTS), 1), minor=True)
    axes[0, 0].set_yticks(np.arange(-0.5, len(ELEMENTS), 1), minor=True)
    axes[0, 0].grid(which="minor", color="white", linewidth=1.0)
    axes[0, 0].tick_params(which="minor", bottom=False, left=False)
    chemistry_divider = make_axes_locatable(axes[0, 0])
    chemistry_colorbar_axis = chemistry_divider.append_axes("right", size="5%", pad=0.08)
    colorbar = figure.colorbar(image, cax=chemistry_colorbar_axis)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("samples", fontsize=8)

    task = hea["task_frame"]
    split_columns = ["id_split", "composition_ood_split", "chemistry_ood_split", "sro_ood_split"]
    labels = ["FCC", "BCC", "ID", "Composition", "Chemistry", "SRO"]
    counts = [
        int(task["structure_family"].astype(str).eq("fcc_like").sum()),
        int(task["structure_family"].astype(str).eq("bcc_like").sum()),
        *[int(task[column].astype(str).eq("test").sum()) for column in split_columns],
    ]
    colors = ["#2563EB", "#2563EB", "#D97706", "#D97706", "#D97706", "#D97706"]
    y = np.arange(len(labels))[::-1]
    axes[0, 1].barh(y, counts, color=colors, height=0.62)
    axes[0, 1].set_yticks(y, labels)
    axes[0, 1].set_title("(b) HEA topology and held-out tests", loc="left", fontweight="bold", pad=12)
    axes[0, 1].set_xlabel("structure-XRD pairs")
    axes[0, 1].xaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[0, 1].tick_params(axis="both", labelsize=8)
    max_count = max(counts) if counts else 1
    axes[0, 1].set_xlim(0, max_count * 1.18)
    for yy, count in zip(y, counts):
        axes[0, 1].text(
            count + max_count * 0.02, yy, f"{count / 1000:.1f}k",
            va="center", ha="left", fontsize=7, color="#334155",
        )
    _style_axis(axes[0, 1])

    gamma_counts = pd.crosstab(
        gamma["candidate_frame"]["difficulty"], gamma["candidate_frame"]["component_count"]
    ).reindex(["single", "resolved", "partial", "strong", "severe"], fill_value=0).sort_index(axis=1)
    gamma_counts = gamma_counts.rename(
        columns=lambda value: f"{int(value)} comp" if int(value) == 1 else f"{int(value)} comps"
    )
    component_colors = ["#94A3B8", "#0891B2", "#F59E0B", "#7C3AED"]
    gamma_counts.plot(
        kind="bar", stacked=True, ax=axes[1, 0],
        color=component_colors[: len(gamma_counts.columns)], width=0.72,
    )
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("window count")
    axes[1, 0].tick_params(axis="x", rotation=0)
    axes[1, 0].tick_params(axis="both", labelsize=8)
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(_thousands))
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if axes[1, 0].legend_ is not None:
        axes[1, 0].legend_.remove()
    gamma_legend_axis.legend(
        handles, labels,
        title=None, frameon=False, fontsize=9,
        ncol=min(3, len(labels)), loc="center left",
        bbox_to_anchor=(0.0, 0.5), borderaxespad=0.0,
        handlelength=1.4, columnspacing=1.2,
    )
    _style_axis(axes[1, 0])

    occupancy_path = Path(hea["task_path"]).parent / "occupancy_pairs_calibrated.parquet"
    ratios = pd.read_parquet(occupancy_path)["identifiability_ratio"].astype(float) if occupancy_path.exists() else pd.Series(dtype=float)
    bins = np.linspace(0, 1.0, 45)
    axes[1, 1].hist(ratios.clip(lower=0, upper=1.0), bins=bins, color="#2563EB", alpha=0.88)
    axes[1, 1].axvspan(0, 1.0, color="#2563EB", alpha=0.05, linewidth=0)
    axes[1, 1].axvline(1.0, color="#B91C1C", linestyle="--", linewidth=1.2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("(d) Occupancy identifiability diagnostic", loc="left", fontweight="bold", pad=12)
    axes[1, 1].set_xlabel("signal / parent-specific simulated-noise floor")
    axes[1, 1].set_ylabel("pair count (log)")
    axes[1, 1].tick_params(axis="both", labelsize=8)
    above_threshold = int((ratios >= 1.0).sum())
    total_pairs = int(len(ratios))
    axes[1, 1].text(
        0.97, 0.88, f"{above_threshold:,} / {total_pairs:,}\nabove threshold",
        transform=axes[1, 1].transAxes, ha="right", va="top",
        fontsize=8, color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92, "pad": 3},
    )
    axes[1, 1].text(
        1.0, 0.05, "threshold = 1.0", transform=axes[1, 1].get_xaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="#B91C1C",
    )
    _style_axis(axes[1, 1])
    figure.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.10)
    _save_figure(figure, path)


def _displayed_metric_ranks(
    frame: pd.DataFrame, group: str, metric: str, digits: int, *, ascending: bool
) -> pd.Series:
    # Rank at the printed precision so visually identical results share a rank.
    displayed = frame[metric].map(
        lambda value: float(f"{value:.{digits}f}") if np.isfinite(value) else np.nan
    )
    return displayed.groupby(frame[group]).rank(method="dense", ascending=ascending)


def _format_ranked_result(label: str, value: float, rank: float, digits: int) -> tuple[str, str]:
    metric = f"{value:.{digits}f}" if np.isfinite(value) else "--"
    if rank == 1:
        return f"\\textbf{{{label}}}", f"\\textbf{{{metric}}}"
    if rank == 2:
        metric = f"\\underline{{{metric}}}"
    return label, metric


def _write_compact_results(hea_path: Path, gamma_path: Path, output: Path) -> None:
    hea = pd.read_csv(_required_file(hea_path, "HEA paper results"))
    gamma = pd.read_csv(_required_file(gamma_path, "Gamma paper results"))
    hea = hea.loc[hea["model"].isin(MODEL_LABELS)].copy()
    hea["display_rank"] = _displayed_metric_ranks(
        hea, "split_column", "parent_macro_mae_mean", 4, ascending=True
    )
    gamma["display_rank"] = _displayed_metric_ranks(
        gamma, "task", "component_f1_mean", 3, ascending=False
    )
    lines = [
        "% Generated by alloy_xrd_bench.dataset_paper_visuals; do not edit.",
        "\\begin{minipage}[t]{0.48\\textwidth}",
        "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\textbf{HEA-SRO parent-macro MAE}\\par\\smallskip",
        "\\begin{tabular}{llrr}",
        "\\toprule Split & Model & MAE $\\downarrow$ & SD \\\\",
        "\\midrule",
    ]
    for row in hea.to_dict(orient="records"):
        split = str(row["split_column"]).replace("_split", "").replace("_", "\\_")
        label, metric = _format_ranked_result(
            MODEL_LABELS[str(row["model"])], float(row["parent_macro_mae_mean"]),
            row["display_rank"], 4,
        )
        standard_deviation = float(row["random_seed_std"])
        deviation_text = f"{standard_deviation:.4f}" if np.isfinite(standard_deviation) else "--"
        lines.append(
            f"{split} & {label} & {metric} & {deviation_text} \\\\"
        )
    lines.extend([
        "\\bottomrule", "\\end{tabular}", "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.48\\textwidth}", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\textbf{Gamma-Decomp component F1}\\par\\smallskip",
        "\\begin{tabular}{llrr}",
        "\\toprule View & Model & F1 $\\uparrow$ & SD \\\\", "\\midrule",
    ])
    for row in gamma.to_dict(orient="records"):
        label = str(row.get("model_label", row["model"])).replace("Physics-constrained", "Physics-constr.")
        label, metric = _format_ranked_result(
            label, float(row["component_f1_mean"]), row["display_rank"], 3
        )
        standard_deviation = float(row.get("component_f1_std", float("nan")))
        standard_deviation_text = (
            f"{standard_deviation:.3f}" if np.isfinite(standard_deviation) else "--"
        )
        lines.append(
            f"{str(row['task']).title()} & {label} & {metric} & "
            f"{standard_deviation_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{minipage}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset_paper_visuals(
    hea_release_root: str | Path,
    gamma_release_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 2027,
    hea_results_path: str | Path = "docs/paper/generated/hea_sro_results.csv",
    gamma_results_path: str | Path = "docs/paper/generated/gamma_decomp_results.csv",
) -> dict[str, str]:
    """Generate dataset-first paper figures and their provenance manifest."""

    np.random.default_rng(seed)
    hea_root = Path(hea_release_root).resolve()
    gamma_root = Path(gamma_release_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    hea = _read_hea_data(hea_root)
    gamma = _read_gamma_data(gamma_root)
    paths = {
        "dataset_overview": output / "dataset_overview.png",
        "hea_sro_example": output / "hea_sro_example.png",
        "gamma_decomp_example": output / "gamma_decomp_example.png",
        "dataset_coverage": output / "dataset_coverage.png",
        "examples": output / "dataset_visual_examples.json",
        "compact_results": output / "compact_primary_results.tex",
    }
    _write_dataset_overview(hea, gamma, paths["dataset_overview"])
    _write_hea_example(hea, paths["hea_sro_example"])
    _write_gamma_example(gamma, paths["gamma_decomp_example"])
    _write_dataset_coverage(hea, gamma, paths["dataset_coverage"])
    examples = {
        "schema_version": "sitexrd-dataset-visual-examples-v1.1",
        "seed": int(seed),
        "hea_sro": {
            **hea["selected"],
            "candidate_fidelity": hea["candidate_fidelity"],
            "selection_rule": "source relaxed DFT train, same composition/topology, cosine >= 0.995, maximum SRO L1",
            "structure_paths": {key: str(value) for key, value in hea["structure_paths"].items()},
            "structure_source_rows": hea["structure_source_rows"],
            "spectrum_sha256": {
                str(hea["selected"]["left_sample_id"]): _array_sha256(hea["spectra"][str(hea["selected"]["left_sample_id"])]),
                str(hea["selected"]["right_sample_id"]): _array_sha256(hea["spectra"][str(hea["selected"]["right_sample_id"])]),
            },
        },
        "gamma_decomp": {
            "selection_rule": "train, nearest to difficulty medians for component count, SNR, and phase weight",
            "resolved": gamma["selected"]["resolved"],
            "severe": gamma["selected"]["severe"],
            "full_source_sample_ids": gamma["selected_source_sample_ids"],
            "parent_pair_id": gamma["parent_pair_id"],
            "parent_structure_ids": gamma["parent_ids"],
            "structure_paths": {key: str(value) for key, value in gamma["structure_paths"].items()},
        },
    }
    paths["examples"].write_text(json.dumps(examples, indent=2, sort_keys=True), encoding="utf-8")
    _write_compact_results(Path(hea_results_path), Path(gamma_results_path), paths["compact_results"])
    script_path = Path(__file__).resolve()
    sources = [
        hea["task_path"], hea["composition_path"], hea["structures_path"],
        gamma["sample_path"], gamma["component_path"], gamma["slice_path"],
        gamma["full_sample_path"], gamma["full_component_path"],
        gamma["evaluation_groups_path"], gamma["pairs_path"], gamma["structures_path"],
        *hea["grid_shard_paths"], *gamma["selected_pattern_paths"].values(),
        *gamma["selected_full_pattern_paths"].values(),
        *hea["structure_paths"].values(), *gamma["structure_paths"].values(),
        Path(hea_results_path).resolve(), Path(gamma_results_path).resolve(),
    ]
    if gamma["edge_path"] is not None:
        sources.append(gamma["edge_path"])
    sources = sorted({Path(path).resolve() for path in sources}, key=str)
    manifest_path = output / "dataset_visual_manifest.json"
    manifest = {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "seed": int(seed),
        "script_sha256": _sha256(script_path),
        "configuration": {
            "hea_min_cosine_similarity": HEA_MIN_COSINE_SIMILARITY,
            "hea_candidate_fidelity": hea["candidate_fidelity"],
            "hea_pair_tiebreak": "descending_sro_l1_then_lexicographic_sample_ids",
            "gamma_selection": "minimum_scaled_distance_to_group_medians_then_sample_id",
            "gamma_difficulties": ["resolved", "severe"],
            "structure_projection": "[001]",
            "structure_projection_description": "projected 2x2 cells; Gamma panels use FCC and L1_2 prototype guides",
            "seed": int(seed),
        },
        "sources": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        "artifacts": [
            {"key": key, "path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = manifest_path
    return {key: str(path) for key, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hea-release-root", type=Path, required=True)
    parser.add_argument("--gamma-release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--hea-results", type=Path, default=Path("docs/paper/generated/hea_sro_results.csv"))
    parser.add_argument("--gamma-results", type=Path, default=Path("docs/paper/generated/gamma_decomp_results.csv"))
    args = parser.parse_args(argv)
    result = generate_dataset_paper_visuals(
        args.hea_release_root,
        args.gamma_release_root,
        args.output,
        seed=args.seed,
        hea_results_path=args.hea_results,
        gamma_results_path=args.gamma_results,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
