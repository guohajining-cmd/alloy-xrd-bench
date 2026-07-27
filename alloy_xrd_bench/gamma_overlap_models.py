"""DETR-style set models for public gamma/gamma-prime decomposition tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F

from .hea_validation_models import ResidualSpectrumEncoder, SpectrumTransformer


class GammaSetPredictor(nn.Module):
    """Encode a spectrum and decode an unordered, fixed-capacity component set."""

    def __init__(
        self,
        *,
        encoder_name: str,
        class_count: int,
        query_count: int,
        point_count: int,
        overlap_mask: bool,
        embedding_dim: int = 128,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.input_projection = (
            nn.Identity() if input_channels == 1 else nn.Conv1d(input_channels, 1, kernel_size=1)
        )
        if encoder_name == "resnet1d":
            self.encoder = ResidualSpectrumEncoder(embedding_dim)
        elif encoder_name == "spectrum_transformer":
            self.encoder = SpectrumTransformer(
                embedding_dim,
                model_dim=embedding_dim,
                max_points=max(4096, point_count),
            )
        else:
            raise ValueError("encoder_name must be resnet1d or spectrum_transformer")
        self.query_embedding = nn.Parameter(torch.empty(query_count, embedding_dim))
        nn.init.normal_(self.query_embedding, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=4 * embedding_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.existence_head = nn.Linear(embedding_dim, 1)
        self.class_head = nn.Linear(embedding_dim, class_count)
        self.parameter_head = nn.Linear(embedding_dim, 5)
        self.mask_head = nn.Linear(embedding_dim, point_count) if overlap_mask else None

    def forward(self, spectra: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(self.input_projection(spectra))
        queries = self.query_embedding.unsqueeze(0).expand(spectra.shape[0], -1, -1)
        decoded = self.decoder(queries, embedding.unsqueeze(1))
        result = {
            "existence_logits": self.existence_head(decoded).squeeze(-1),
            "class_logits": self.class_head(decoded),
            "parameters": torch.sigmoid(self.parameter_head(decoded)),
        }
        if self.mask_head is not None:
            result["overlap_mask_logits"] = self.mask_head(embedding)
        return result


def hungarian_set_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    class_weight: float = 1.0,
    parameter_weight: float = 1.0,
    existence_weight: float = 1.0,
    mask_weight: float = 1.0,
    reconstruction_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Match target components to queries and compute permutation-invariant losses."""

    existence_logits = outputs["existence_logits"]
    class_logits = outputs["class_logits"]
    parameters = outputs["parameters"]
    if len(targets) != existence_logits.shape[0]:
        raise ValueError("target batch size does not match model output")
    existence_target = torch.zeros_like(existence_logits)
    class_losses: list[torch.Tensor] = []
    parameter_losses: list[torch.Tensor] = []
    matches: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_index, target in enumerate(targets):
        target_classes = target["classes"].to(class_logits.device, dtype=torch.long)
        target_parameters = target["parameters"].to(parameters.device, dtype=parameters.dtype)
        if len(target_classes) == 0:
            matches.append((np.array([], dtype=int), np.array([], dtype=int)))
            continue
        if len(target_classes) > existence_logits.shape[1]:
            raise ValueError("target component count exceeds query capacity")
        class_cost = -class_logits[batch_index].log_softmax(-1)[:, target_classes]
        parameter_cost = torch.cdist(parameters[batch_index], target_parameters, p=1) / 5.0
        existence_cost = -existence_logits[batch_index].sigmoid().unsqueeze(1)
        cost = class_cost + parameter_cost + existence_cost
        query_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        matches.append((query_indices, target_indices))
        query_tensor = torch.as_tensor(query_indices, device=class_logits.device)
        target_tensor = torch.as_tensor(target_indices, device=class_logits.device)
        existence_target[batch_index, query_tensor] = 1.0
        class_losses.append(
            F.cross_entropy(class_logits[batch_index, query_tensor], target_classes[target_tensor])
        )
        parameter_losses.append(
            F.l1_loss(parameters[batch_index, query_tensor], target_parameters[target_tensor])
        )
    existence_loss = F.binary_cross_entropy_with_logits(existence_logits, existence_target)
    zero = existence_logits.sum() * 0.0
    class_loss = torch.stack(class_losses).mean() if class_losses else zero
    parameter_loss = torch.stack(parameter_losses).mean() if parameter_losses else zero
    mask_loss = zero
    if "overlap_mask_logits" in outputs and all("overlap_mask" in target for target in targets):
        mask_target = torch.stack([
            target["overlap_mask"].to(outputs["overlap_mask_logits"].device, dtype=outputs["overlap_mask_logits"].dtype)
            for target in targets
        ])
        mask_loss = F.binary_cross_entropy_with_logits(outputs["overlap_mask_logits"], mask_target)
    reconstruction_loss = zero
    if reconstruction_weight and all(
        {"grid_deg", "intensity", "intensity_scale"} <= set(target) for target in targets
    ):
        reconstruction_losses = []
        for batch_index, target in enumerate(targets):
            grid = target["grid_deg"].to(parameters.device, dtype=parameters.dtype)
            scale = target["intensity_scale"].to(parameters.device, dtype=parameters.dtype)
            predicted = parameters[batch_index]
            center = 20.0 + 80.0 * predicted[:, 0]
            fwhm = 0.03 + 1.17 * predicted[:, 1]
            area = 100.0 * predicted[:, 2]
            eta = predicted[:, 3]
            delta = (grid.unsqueeze(0) - center.unsqueeze(1)) / fwhm.unsqueeze(1)
            gaussian = torch.sqrt(torch.tensor(4.0 * np.log(2.0) / np.pi, device=grid.device)) / fwhm.unsqueeze(1)
            gaussian = gaussian * torch.exp(-4.0 * np.log(2.0) * delta.square())
            lorentzian = (2.0 / (np.pi * fwhm)).unsqueeze(1) / (1.0 + 4.0 * delta.square())
            profiles = area.unsqueeze(1) * (
                eta.unsqueeze(1) * lorentzian + (1.0 - eta).unsqueeze(1) * gaussian
            )
            existence = existence_logits[batch_index].sigmoid().unsqueeze(1)
            reconstruction = (existence * profiles).sum(0) / scale.clamp_min(1e-8)
            target_intensity = target["intensity"].to(reconstruction.device, dtype=reconstruction.dtype)
            reconstruction_losses.append(F.l1_loss(reconstruction, target_intensity))
        reconstruction_loss = torch.stack(reconstruction_losses).mean()
    total = (
        existence_weight * existence_loss
        + class_weight * class_loss
        + parameter_weight * parameter_loss
        + mask_weight * mask_loss
        + reconstruction_weight * reconstruction_loss
    )
    return {
        "loss": total,
        "existence_loss": existence_loss,
        "class_loss": class_loss,
        "parameter_loss": parameter_loss,
        "overlap_mask_loss": mask_loss,
        "reconstruction_loss": reconstruction_loss,
        "matches": matches,
    }
