"""Physical primitives for the Ni-Al gamma/gamma-prime overlap benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ComponentSpec:
    phase: str
    hkl: tuple[int, int, int]
    center_deg: float
    fwhm_deg: float
    integrated_intensity: float
    eta: float


def canonical_cubic_hkl(hkl: Sequence[int]) -> tuple[int, int, int]:
    """Canonicalize sign and axis permutations without reducing reflection order."""

    if len(hkl) != 3:
        raise ValueError("hkl must contain three indices")
    values = tuple(sorted((abs(int(value)) for value in hkl), reverse=True))
    if values == (0, 0, 0):
        raise ValueError("(0, 0, 0) is not a diffraction reflection")
    return values


def parent_hkl_from_supercell(
    supercell_hkl: Sequence[int],
    supercell: Sequence[int],
    *,
    tolerance: float = 1e-8,
) -> tuple[int, int, int] | None:
    """Map a diagonal-supercell reflection to an integer parent reflection."""

    if len(supercell_hkl) != 3 or len(supercell) != 3:
        raise ValueError("supercell_hkl and supercell must have length three")
    mapped = []
    for index, repeat in zip(supercell_hkl, supercell):
        if int(repeat) <= 0:
            raise ValueError("supercell repeats must be positive")
        value = float(index) / int(repeat)
        rounded = round(value)
        if abs(value - rounded) > tolerance:
            return None
        mapped.append(int(rounded))
    if mapped == [0, 0, 0]:
        return None
    return canonical_cubic_hkl(mapped)


def pseudo_voigt_profile(
    two_theta: np.ndarray,
    *,
    center_deg: float,
    fwhm_deg: float,
    integrated_intensity: float,
    eta: float,
) -> np.ndarray:
    """Render a discretely area-normalized pseudo-Voigt profile."""

    grid = np.asarray(two_theta, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("two_theta must be a strictly increasing one-dimensional grid")
    if not np.isfinite(center_deg):
        raise ValueError("center_deg must be finite")
    if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
        raise ValueError("fwhm_deg must be finite and positive")
    if not np.isfinite(integrated_intensity) or integrated_intensity < 0:
        raise ValueError("integrated_intensity must be finite and non-negative")
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and in [0, 1]")

    offset = grid - float(center_deg)
    sigma = float(fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gaussian = np.exp(-0.5 * (offset / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    half_width = float(fwhm_deg) / 2.0
    lorentzian = half_width / (np.pi * (offset**2 + half_width**2))
    profile = (1.0 - float(eta)) * gaussian + float(eta) * lorentzian
    discrete_area = float(np.trapezoid(profile, grid))
    if discrete_area <= 0 or not np.isfinite(discrete_area):
        raise ValueError("profile has no finite support on the supplied grid")
    return profile * (float(integrated_intensity) / discrete_area)


def render_components(
    two_theta: np.ndarray,
    components: Iterable[ComponentSpec],
) -> tuple[np.ndarray, np.ndarray]:
    specs = list(components)
    curves = np.vstack(
        [
            pseudo_voigt_profile(
                two_theta,
                center_deg=component.center_deg,
                fwhm_deg=component.fwhm_deg,
                integrated_intensity=component.integrated_intensity,
                eta=component.eta,
            )
            for component in specs
        ]
    ) if specs else np.empty((0, len(two_theta)), dtype=float)
    return curves.sum(axis=0), curves


def separation_ratio(left: ComponentSpec, right: ComponentSpec) -> float:
    denominator = 0.5 * (left.fwhm_deg + right.fwhm_deg)
    if denominator <= 0:
        raise ValueError("component FWHM values must be positive")
    return abs(left.center_deg - right.center_deg) / denominator


def overlap_difficulty(separation_ratios: Sequence[float]) -> str:
    if not separation_ratios:
        return "single"
    minimum = min(float(value) for value in separation_ratios)
    if minimum >= 2.0:
        return "resolved"
    if minimum >= 1.0:
        return "partial"
    if minimum >= 0.5:
        return "strong"
    return "severe"


def hkl_label(hkl: Sequence[int]) -> str:
    return "".join(str(value) for value in canonical_cubic_hkl(hkl))

