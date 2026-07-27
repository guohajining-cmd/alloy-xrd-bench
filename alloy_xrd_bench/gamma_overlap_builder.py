"""Deterministic spectrum synthesis and window construction."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gamma_overlap_physics import (
    ComponentSpec,
    overlap_difficulty,
    render_components,
    separation_ratio,
)


def load_overlap_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "config_id",
        "wavelength_angstrom",
        "two_theta_min",
        "two_theta_max",
        "step_deg",
        "window_width_deg",
        "split_ratios",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"overlap config is missing fields: {sorted(missing)}")
    point_count = int(round((config["two_theta_max"] - config["two_theta_min"]) / config["step_deg"])) + 1
    if point_count != 4001:
        raise ValueError("v1 full-pattern grid must contain 4001 points")
    window_count = int(round(config["window_width_deg"] / config["step_deg"])) + 1
    if window_count != 151:
        raise ValueError("v1 window grid must contain 151 points")
    if not np.isclose(sum(float(value) for value in config["split_ratios"].values()), 1.0):
        raise ValueError("split ratios must sum to one")
    return config


def synthesize_mixture(
    *,
    sample_id: str,
    parent_pair_id: str,
    gamma_reflections: Sequence[Mapping[str, Any]],
    gamma_prime_reflections: Sequence[Mapping[str, Any]],
    sample_seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Render component truth, clean mixture, background, and noisy observation."""

    rng = np.random.default_rng(int(sample_seed))
    step = float(config["step_deg"])
    grid = np.round(
        np.arange(
            float(config["two_theta_min"]),
            float(config["two_theta_max"]) + step / 2.0,
            step,
        ),
        8,
    )
    gamma_prime_weight = _uniform(rng, config["gamma_prime_weight"])
    phase_weights = {"gamma": 1.0 - gamma_prime_weight, "gamma_prime": gamma_prime_weight}
    instrument_fwhm = _uniform(rng, config["instrument_fwhm_deg"])
    microstrain = _uniform(rng, config["microstrain_fraction"])
    crystallite_size = _uniform(rng, config["crystallite_size_nm"])
    eta = _uniform(rng, config["pseudo_voigt_eta"])
    zero_shift = _uniform(rng, config["zero_shift_deg"])
    background_fraction = _uniform(rng, config["background_fraction"])
    gaussian_noise_fraction = _uniform(rng, config["gaussian_noise_fraction"])
    poisson_scale = _uniform(rng, config["poisson_scale"])

    components: list[ComponentSpec] = []
    component_rows: list[dict[str, Any]] = []
    for phase, reflections in (
        ("gamma", gamma_reflections),
        ("gamma_prime", gamma_prime_reflections),
    ):
        eligible = [
            row
            for row in reflections
            if bool(row.get("scored", True))
            and str(row.get("reflection_kind", "parent_bragg")) == "parent_bragg"
            and float(row["raw_integrated_intensity"]) > 0
        ]
        total = sum(float(row["raw_integrated_intensity"]) for row in eligible)
        if total <= 0:
            raise ValueError(f"{phase} has no positive parent-Bragg intensity")
        minimum = float(config.get("minimum_phase_relative_area", 0.0))
        for reflection in eligible:
            phase_relative_area = float(reflection["raw_integrated_intensity"]) / total
            if phase_relative_area < minimum:
                continue
            center = float(reflection["center_deg"]) + zero_shift
            fwhm = physically_broadened_fwhm(
                center_deg=center,
                wavelength_angstrom=float(config["wavelength_angstrom"]),
                instrument_fwhm_deg=instrument_fwhm,
                microstrain_fraction=microstrain,
                crystallite_size_nm=crystallite_size,
            )
            area = 100.0 * phase_weights[phase] * phase_relative_area
            spec = ComponentSpec(
                phase=phase,
                hkl=(int(reflection["h"]), int(reflection["k"]), int(reflection["l"])),
                center_deg=center,
                fwhm_deg=fwhm,
                integrated_intensity=area,
                eta=eta,
            )
            component_id = f"component_{sample_id}_{len(components):04d}"
            components.append(spec)
            component_rows.append(
                {
                    "sample_id": sample_id,
                    "component_id": component_id,
                    "parent_pair_id": parent_pair_id,
                    "phase": phase,
                    "h": spec.hkl[0],
                    "k": spec.hkl[1],
                    "l": spec.hkl[2],
                    "center_deg": spec.center_deg,
                    "fwhm_deg": spec.fwhm_deg,
                    "integrated_intensity": spec.integrated_intensity,
                    "eta": spec.eta,
                    "profile_kind": "pseudo_voigt",
                    "score_eligible": True,
                }
            )
    clean, curves = render_components(grid, components)
    peak_height = float(clean.max())
    normalized_x = np.linspace(-1.0, 1.0, len(grid))
    slope = float(rng.uniform(-0.25, 0.25))
    curvature = float(rng.uniform(0.0, 0.25))
    background_shape = np.clip(0.75 + slope * normalized_x + curvature * normalized_x**2, 0.05, None)
    background = peak_height * background_fraction * background_shape
    signal = clean + background
    expected_counts = np.clip(signal / max(float(signal.max()), 1e-12) * poisson_scale, 0.0, None)
    poisson = rng.poisson(expected_counts) / poisson_scale * max(float(signal.max()), 1e-12)
    noise_sigma = gaussian_noise_fraction * peak_height
    gaussian_noise = rng.normal(0.0, noise_sigma, size=len(grid))
    observed = np.clip(poisson + gaussian_noise, 0.0, None)
    for row, curve in zip(component_rows, curves):
        row["score_eligible"] = bool(float(curve.max()) >= float(config["component_detection_sigma"]) * noise_sigma)
    return {
        "sample_id": sample_id,
        "parent_pair_id": parent_pair_id,
        "sample_seed": int(sample_seed),
        "two_theta": grid,
        "clean": clean,
        "background": background,
        "observed": observed,
        "component_curves": curves,
        "component_rows": component_rows,
        "gamma_prime_weight": gamma_prime_weight,
        "instrument_fwhm_deg": instrument_fwhm,
        "microstrain_fraction": microstrain,
        "crystallite_size_nm": crystallite_size,
        "eta": eta,
        "zero_shift_deg": zero_shift,
        "background_fraction": background_fraction,
        "gaussian_noise_fraction": gaussian_noise_fraction,
        "noise_sigma": noise_sigma,
        "poisson_scale": poisson_scale,
    }


def physically_broadened_fwhm(
    *,
    center_deg: float,
    wavelength_angstrom: float,
    instrument_fwhm_deg: float,
    microstrain_fraction: float,
    crystallite_size_nm: float,
) -> float:
    theta = np.deg2rad(float(center_deg) / 2.0)
    size_angstrom = float(crystallite_size_nm) * 10.0
    size_radians = 0.9 * float(wavelength_angstrom) / (size_angstrom * max(float(np.cos(theta)), 1e-8))
    strain_radians = 4.0 * float(microstrain_fraction) * float(np.tan(theta))
    instrument_radians = np.deg2rad(float(instrument_fwhm_deg))
    return float(np.rad2deg(np.sqrt(instrument_radians**2 + size_radians**2 + strain_radians**2)))


def extract_windows(sample: Mapping[str, Any], *, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    components = list(sample["component_rows"])
    if not components:
        return []
    support = float(config["profile_support_fwhm"])
    ordered = sorted(components, key=lambda row: float(row["center_deg"]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_right = -float("inf")
    for row in ordered:
        left = float(row["center_deg"]) - support * float(row["fwhm_deg"])
        right = float(row["center_deg"]) + support * float(row["fwhm_deg"])
        if current and left > current_right:
            groups.append(current)
            current = []
        current.append(row)
        current_right = max(current_right, right) if len(current) > 1 else right
    if current:
        groups.append(current)

    grid = np.asarray(sample["two_theta"], dtype=float)
    point_count = int(round(float(config["window_width_deg"]) / float(config["step_deg"]))) + 1
    windows = []
    for group_index, group in enumerate(groups):
        center = float(np.mean([float(row["center_deg"]) for row in group]))
        start = int(np.argmin(np.abs(grid - (center - float(config["window_width_deg"]) / 2.0))))
        start = max(0, min(start, len(grid) - point_count))
        stop = start + point_count
        left_edge = float(grid[start])
        right_edge = float(grid[stop - 1])
        targets = [row for row in components if left_edge <= float(row["center_deg"]) <= right_edge]
        specs = [
            ComponentSpec(
                str(row["phase"]),
                (int(row["h"]), int(row["k"]), int(row["l"])),
                float(row["center_deg"]),
                float(row["fwhm_deg"]),
                float(row["integrated_intensity"]),
                float(row["eta"]),
            )
            for row in targets
        ]
        ratios = [separation_ratio(left, right) for left, right in combinations(specs, 2)]
        windows.append(
            {
                "window_id": f"window_{sample['sample_id']}_{group_index:03d}",
                "sample_id": str(sample["sample_id"]),
                "parent_pair_id": str(sample["parent_pair_id"]),
                "two_theta_min": left_edge,
                "two_theta_max": right_edge,
                "component_count": len(targets),
                "target_component_ids": [str(row["component_id"]) for row in targets],
                "difficulty": overlap_difficulty(ratios),
                "minimum_separation_ratio": min(ratios) if ratios else None,
                "two_theta": grid[start:stop].copy(),
                "observed": np.asarray(sample["observed"])[start:stop].copy(),
                "clean": np.asarray(sample["clean"])[start:stop].copy(),
            }
        )
    return windows


def assign_pair_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    ratios: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    ratios = ratios or {"train": 0.70, "validation": 0.15, "private_test": 0.15}
    pair_ids = sorted(
        {str(row["parent_pair_id"]) for row in rows},
        key=lambda pair_id: hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest(),
    )
    n_pairs = len(pair_ids)
    train_count = int(round(n_pairs * float(ratios["train"])))
    validation_count = int(round(n_pairs * float(ratios["validation"])))
    if n_pairs >= 3:
        train_count = min(max(train_count, 1), n_pairs - 2)
        validation_count = min(max(validation_count, 1), n_pairs - train_count - 1)
    assignment = {}
    for index, pair_id in enumerate(pair_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "private_test"
        assignment[pair_id] = split
    return [{**dict(row), "split": assignment[str(row["parent_pair_id"])]} for row in rows]


def maximum_cross_split_cosine(
    fingerprints: np.ndarray,
    splits: Sequence[str],
    record_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the nearest cosine pair whose records belong to different splits."""

    matrix = np.asarray(fingerprints, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(splits) or len(matrix) != len(record_ids):
        raise ValueError("fingerprints, splits, and record_ids must have aligned rows")
    if not np.isfinite(matrix).all():
        raise ValueError("fingerprints must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(norms[:, None], np.finfo(float).tiny)
    best = -1.0
    best_ids: list[str] = []
    for left in range(len(matrix)):
        candidates = [right for right in range(left + 1, len(matrix)) if splits[left] != splits[right]]
        if not candidates:
            continue
        similarities = normalized[candidates] @ normalized[left]
        local = int(np.argmax(similarities))
        value = float(similarities[local])
        if value > best:
            best = value
            best_ids = [str(record_ids[left]), str(record_ids[candidates[local]])]
    return {"maximum_cosine_similarity": best, "record_ids": best_ids}


def isolate_pair_split_components(
    rows: Sequence[Mapping[str, Any]],
    fingerprints: np.ndarray,
    *,
    seed: int,
    ratios: Mapping[str, float],
    cosine_threshold: float,
) -> list[dict[str, Any]]:
    """Group provenance-linked or near-duplicate pairs before assigning splits."""

    matrix = np.asarray(fingerprints, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(rows):
        raise ValueError("fingerprints must have one row per parent pair")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in ("source_run_id", "gamma_parent_id"):
        first_by_value: dict[str, int] = {}
        for index, row in enumerate(rows):
            value = str(row[field])
            if value in first_by_value:
                union(first_by_value[value], index)
            else:
                first_by_value[value] = index

    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(norms[:, None], np.finfo(float).tiny)
    similarities = normalized @ normalized.T
    for left, right in np.argwhere(np.triu(similarities >= float(cosine_threshold), k=1)):
        union(int(left), int(right))

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    components = sorted(
        groups.values(),
        key=lambda members: (
            -len(members),
            hashlib.sha256(
                f"{seed}:".encode("utf-8")
                + "|".join(sorted(str(rows[index]["parent_pair_id"]) for index in members)).encode("utf-8")
            ).hexdigest(),
        ),
    )
    split_names = ("train", "validation", "private_test")
    targets = {split: len(rows) * float(ratios[split]) for split in split_names}
    assigned_counts = {split: 0 for split in split_names}
    result = [dict(row) for row in rows]
    for component_index, members in enumerate(components):
        split = max(
            split_names,
            key=lambda name: (targets[name] - assigned_counts[name], -split_names.index(name)),
        )
        component_id = f"split_component_{component_index:04d}"
        for index in members:
            result[index]["split"] = split
            result[index]["split_component_id"] = component_id
        assigned_counts[split] += len(members)
    return result


def select_balanced_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    single_fraction: float = 0.20,
    maximum_overlap_class_ratio: float = 3.0,
) -> list[dict[str, Any]]:
    """Deterministically downsample window classes within each split."""

    if not 0.0 <= single_fraction < 1.0:
        raise ValueError("single_fraction must be in [0, 1)")
    if maximum_overlap_class_ratio < 1.0:
        raise ValueError("maximum_overlap_class_ratio must be at least one")

    selected: list[dict[str, Any]] = []
    split_names = sorted({str(row.get("split", "")) for row in rows})
    for split in split_names:
        split_rows = [dict(row) for row in rows if str(row.get("split", "")) == split]

        def ordered(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                (dict(item) for item in items),
                key=lambda item: hashlib.sha256(
                    f"{seed}:{split}:{item['window_id']}".encode("utf-8")
                ).hexdigest(),
            )

        overlap_groups = {
            difficulty: ordered([row for row in split_rows if str(row["difficulty"]) == difficulty])
            for difficulty in ("resolved", "partial", "strong", "severe")
        }
        positive_counts = [len(group) for group in overlap_groups.values() if group]
        if positive_counts:
            cap = max(1, int(np.floor(min(positive_counts) * maximum_overlap_class_ratio)))
            overlap_selected = [row for group in overlap_groups.values() for row in group[:cap]]
        else:
            overlap_selected = []
        singles = ordered([row for row in split_rows if str(row["difficulty"]) == "single"])
        required_singles = int(
            np.ceil(len(overlap_selected) * single_fraction / max(1.0 - single_fraction, np.finfo(float).eps))
        )
        selected.extend(overlap_selected)
        selected.extend(singles[: max(required_singles, 1 if singles and not overlap_selected else 0)])
    return sorted(selected, key=lambda row: str(row["window_id"]))


def _uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
        raise ValueError("sampling bounds must contain ordered [minimum, maximum]")
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))
