"""Diagnostic baseline models for HEA benchmark validation."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from scipy.signal import find_peaks, peak_widths
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import torch
from torch import nn
from torch.nn import functional as F

from .hea_validation_data import ValidationRow


class PerTargetRidge:
    """Independent Ridge regressors fitted only where each target is defined."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.models_: list[Ridge | None] = []
        self.fallbacks_: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> "PerTargetRidge":
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if features.ndim != 2 or targets.ndim != 2 or targets.shape != mask.shape:
            raise ValueError("features, targets, and mask must be aligned matrices")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same row count")
        self.models_ = []
        self.fallbacks_ = np.zeros(targets.shape[1], dtype=float)
        for column in range(targets.shape[1]):
            eligible = mask[:, column]
            if not eligible.any():
                self.models_.append(None)
                continue
            self.fallbacks_[column] = float(np.mean(targets[eligible, column]))
            model = Ridge(alpha=self.alpha)
            model.fit(features[eligible], targets[eligible, column])
            self.models_.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.fallbacks_ is None or not self.models_:
            raise RuntimeError("model has not been fitted")
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be a matrix")
        output = np.tile(self.fallbacks_, (len(features), 1))
        for column, model in enumerate(self.models_):
            if model is not None:
                output[:, column] = model.predict(features)
        return output


class PerTargetExtraTrees:
    """Independent masked ExtraTrees controls for nonlinear composition shortcuts."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
        random_state: int = 2027,
        n_jobs: int = -1,
    ):
        self.parameters = {
            "n_estimators": int(n_estimators),
            "min_samples_leaf": int(min_samples_leaf),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
        }
        self.models_: list[ExtraTreesRegressor | None] = []
        self.fallbacks_: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> "PerTargetExtraTrees":
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if features.ndim != 2 or targets.ndim != 2 or targets.shape != mask.shape:
            raise ValueError("features, targets, and mask must be aligned matrices")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same row count")
        self.models_ = []
        self.fallbacks_ = np.zeros(targets.shape[1], dtype=float)
        for column in range(targets.shape[1]):
            eligible = mask[:, column]
            if not eligible.any():
                self.models_.append(None)
                continue
            self.fallbacks_[column] = float(np.mean(targets[eligible, column]))
            model = ExtraTreesRegressor(**self.parameters)
            model.fit(features[eligible], targets[eligible, column])
            self.models_.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.fallbacks_ is None or not self.models_:
            raise RuntimeError("model has not been fitted")
        features = np.asarray(features, dtype=float)
        output = np.tile(self.fallbacks_, (len(features), 1))
        for column, model in enumerate(self.models_):
            if model is not None:
                output[:, column] = model.predict(features)
        return output


class PerTargetXGBoost:
    """Independent masked XGBoost regressors with finite target fallbacks."""

    def __init__(self, **parameters: object):
        self.parameters = dict(parameters)
        self.models_: list[object | None] = []
        self.fallbacks_: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> "PerTargetXGBoost":
        from xgboost import XGBRegressor

        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if features.ndim != 2 or targets.ndim != 2 or targets.shape != mask.shape:
            raise ValueError("features, targets, and mask must be aligned matrices")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same row count")
        self.models_ = []
        self.fallbacks_ = np.zeros(targets.shape[1], dtype=float)
        for column in range(targets.shape[1]):
            eligible = mask[:, column]
            if not eligible.any():
                self.models_.append(None)
                continue
            self.fallbacks_[column] = float(np.mean(targets[eligible, column]))
            model = XGBRegressor(**self.parameters)
            model.fit(features[eligible], targets[eligible, column], verbose=False)
            self.models_.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.fallbacks_ is None or not self.models_:
            raise RuntimeError("model has not been fitted")
        features = np.asarray(features, dtype=float)
        output = np.tile(self.fallbacks_, (len(features), 1))
        for column, model in enumerate(self.models_):
            if model is not None:
                output[:, column] = model.predict(features)
        return output


def peak_features(
    intensity: np.ndarray, two_theta: np.ndarray, *, max_peaks: int = 32
) -> np.ndarray:
    """Extract fixed peak position, intensity, width, and global features."""
    intensity = np.asarray(intensity, dtype=float)
    two_theta = np.asarray(two_theta, dtype=float)
    if intensity.ndim != 1 or intensity.shape != two_theta.shape:
        raise ValueError("intensity and two_theta must have the same one-dimensional shape")
    if max_peaks <= 0 or not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError("max_peaks must be positive and intensity finite and non-negative")
    if not np.isfinite(two_theta).all() or np.any(np.diff(two_theta) <= 0):
        raise ValueError("two_theta must be finite and strictly increasing")

    peak_indices, _ = find_peaks(intensity)
    if len(peak_indices):
        strongest = np.argsort(intensity[peak_indices])[::-1][:max_peaks]
        peak_indices = peak_indices[strongest]
        widths_in_samples = peak_widths(intensity, peak_indices, rel_height=0.5)[0]
        step = float(np.median(np.diff(two_theta)))
        widths = widths_in_samples * step
    else:
        widths = np.empty(0, dtype=float)
    positions = two_theta[peak_indices]
    heights = intensity[peak_indices]

    def padded(values: np.ndarray) -> np.ndarray:
        return np.pad(values, (0, max_peaks - len(values)), constant_values=0.0)

    total = float(np.trapezoid(intensity, two_theta))
    weight_sum = float(intensity.sum())
    centroid = float(np.dot(intensity, two_theta) / weight_sum) if weight_sum else 0.0
    return np.concatenate(
        [padded(positions), padded(heights), padded(widths), [total, centroid]]
    ).astype(np.float32)


def peak_feature_matrix(
    rows: Sequence[ValidationRow], *, max_peaks: int = 32
) -> np.ndarray:
    """Extract peak features while opening each HDF5 shard only once."""
    grouped: dict[str, list[tuple[int, ValidationRow]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        grouped[row.grid_shard_path].append((output_index, row))
    output = np.empty((len(rows), 3 * max_peaks + 2), dtype=np.float32)
    for shard_path, indexed_rows in grouped.items():
        with h5py.File(Path(shard_path), "r") as handle:
            two_theta = np.asarray(handle["two_theta"], dtype=np.float32)
            for output_index, row in indexed_rows:
                if not 0 <= row.grid_row_index < len(handle["xrd_sample_id"]):
                    raise IndexError(f"grid row index out of range for {row.xrd_sample_id!r}")
                raw_id = handle["xrd_sample_id"][row.grid_row_index]
                actual_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                if actual_id != row.xrd_sample_id:
                    raise ValueError(
                        f"sample ID mismatch: expected {row.xrd_sample_id!r}, found {actual_id!r}"
                    )
                intensity = np.asarray(
                    handle["intensity"][row.grid_row_index], dtype=np.float32
                )
                output[output_index] = peak_features(
                    intensity, two_theta, max_peaks=max_peaks
                )
    return output


def spectrum_matrix(rows: Sequence[ValidationRow]) -> np.ndarray:
    """Load spectra by shard and apply frozen log/max normalization."""
    grouped: dict[str, list[tuple[int, ValidationRow]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        grouped[row.grid_shard_path].append((output_index, row))
    output = np.empty((len(rows), 4001), dtype=np.float32)
    for shard_path, indexed_rows in grouped.items():
        with h5py.File(Path(shard_path), "r") as handle:
            for output_index, row in indexed_rows:
                raw_id = handle["xrd_sample_id"][row.grid_row_index]
                actual_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                if actual_id != row.xrd_sample_id:
                    raise ValueError(
                        f"sample ID mismatch: expected {row.xrd_sample_id!r}, found {actual_id!r}"
                    )
                intensity = np.asarray(
                    handle["intensity"][row.grid_row_index], dtype=np.float32
                )
                if intensity.shape != (4001,) or not np.isfinite(intensity).all():
                    raise ValueError(f"invalid spectrum for {row.xrd_sample_id!r}")
                transformed = np.log1p(intensity)
                scale = float(transformed.max())
                output[output_index] = transformed / scale if scale > 0 else transformed
    return output


def seed_everything(seed: int) -> None:
    """Seed supported RNGs and request deterministic PyTorch kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def masked_mse(
    prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != truth.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, truth, and mask must have identical shapes")
    if not torch.any(mask):
        raise ValueError("masked MSE requires at least one valid target")
    return torch.square(prediction[mask] - truth[mask]).mean()


def normalize_reconstruction_target(spectra: torch.Tensor) -> torch.Tensor:
    """Normalize each benchmark pattern to the unit scale used by the forward head."""
    scale = spectra.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return spectra / scale


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.skip(inputs))


class ResidualSpectrumEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        channels = (16, 32, 64, 128)
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(_ResidualBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(channels[-1], embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, points)")
        encoded = self.blocks(inputs)
        return self.projection(self.pool(encoded).squeeze(-1))


class ResNet1D(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = ResidualSpectrumEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


class SpectrumTransformer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        patch_size: int = 32,
        model_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_points: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv1d(
            1, model_dim, kernel_size=patch_size, stride=patch_size
        )
        max_tokens = math.ceil(max_points / patch_size)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, output_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, points)")
        remainder = inputs.shape[-1] % self.patch_size
        if remainder:
            inputs = F.pad(inputs, (0, self.patch_size - remainder))
        tokens = self.patch_embedding(inputs).transpose(1, 2)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("spectrum exceeds configured maximum point count")
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.norm(encoded.mean(dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(inputs))


class PhysicsConstrainedTransformer(nn.Module):
    """SRO predictor with a differentiable pseudo-Voigt forward head."""

    def __init__(self, output_dim: int, *, peak_count: int = 16):
        super().__init__()
        self.peak_count = int(peak_count)
        self.backbone = SpectrumTransformer(output_dim)
        model_dim = self.backbone.head.in_features
        self.sro_head = nn.Linear(model_dim, output_dim)
        self.peak_head = nn.Linear(model_dim, self.peak_count * 4)
        self.register_buffer("two_theta", torch.linspace(20.0, 100.0, 4001))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.backbone.encode(inputs)
        raw = self.peak_head(latent).reshape(-1, self.peak_count, 4)
        center = 20.0 + 80.0 * raw[..., 0].sigmoid()
        fwhm = 0.03 + 0.77 * raw[..., 1].sigmoid()
        area = raw[..., 2].sigmoid()
        eta = raw[..., 3].sigmoid()
        grid = self.two_theta.to(dtype=inputs.dtype)[None, None, :]
        center_grid = center[..., None]
        sigma = fwhm[..., None] / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        gaussian = torch.exp(-0.5 * torch.square((grid - center_grid) / sigma)) / (
            sigma * math.sqrt(2.0 * math.pi)
        )
        gamma = fwhm[..., None] / 2.0
        lorentzian = (gamma / math.pi) / (torch.square(grid - center_grid) + torch.square(gamma))
        profile = area[..., None] * (
            (1.0 - eta[..., None]) * gaussian + eta[..., None] * lorentzian
        )
        reconstruction = profile.sum(dim=1)
        scale = reconstruction.amax(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "sro": self.sro_head(latent),
            "reconstruction": reconstruction / scale,
            "peak_parameters": torch.stack((center, fwhm, area, eta), dim=-1),
        }


class SiameseRankNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = ResidualSpectrumEncoder(embedding_dim)
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.score(self.encoder(inputs)).squeeze(-1)
