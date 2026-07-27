"""Training workflow for public gamma decomposition set baselines."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .gamma_overlap_models import GammaSetPredictor, hungarian_set_loss
from .gamma_overlap_public import _json_compatible, evaluate_public_submission
from .gamma_overlap_scoring import hkl_group_id


def build_public_class_catalog(release_dir: str | Path, task: str) -> list[tuple[str, int, int, int]]:
    root = Path(release_dir).resolve()
    labels = pd.read_parquet(root / f"{task}/components.parquet")
    train = labels.loc[labels["split"].astype(str).eq("train") & labels["score_eligible"].astype(bool)]
    catalog = sorted({(str(r.phase), int(r.h), int(r.k), int(r.l)) for r in train.itertuples()})
    known = set(catalog)
    unknown = {
        (str(r.phase), int(r.h), int(r.k), int(r.l))
        for r in labels.loc[labels["score_eligible"].astype(bool)].itertuples()
        if (str(r.phase), int(r.h), int(r.k), int(r.l)) not in known
    }
    if unknown:
        raise ValueError(f"validation/test contain phase/HKL classes absent from train: {sorted(unknown)}")
    return catalog


class GammaSetDataset(Dataset):
    def __init__(self, release_dir: str | Path, task: str, split: str, catalog: Sequence[tuple[str, int, int, int]], max_samples: int | None = None, minimum_relative_intensity: float = 0.0, spectrum_view: str = "observed"):
        self.root = Path(release_dir).resolve()
        self.task = task
        samples = pd.read_parquet(self.root / f"{task}/samples.parquet")
        selected = samples.loc[samples["split"].astype(str).eq(split)].sort_values("sample_id")
        self.all_sample_ids = selected["sample_id"].astype(str).tolist()
        if max_samples is not None:
            selected = selected.head(int(max_samples))
        self.samples = selected.to_dict(orient="records")
        labels = pd.read_parquet(self.root / f"{task}/components.parquet")
        labels = labels.loc[labels["split"].astype(str).eq(split) & labels["score_eligible"].astype(bool)]
        self.labels = {str(key): group.to_dict(orient="records") for key, group in labels.groupby("sample_id")}
        self.class_index = {value: index for index, value in enumerate(catalog)}
        self.minimum_relative_intensity = float(minimum_relative_intensity)
        if not 0.0 <= self.minimum_relative_intensity < 1.0:
            raise ValueError("minimum_relative_intensity must be in [0, 1)")
        if spectrum_view not in {"observed", "clean"}:
            raise ValueError("spectrum_view must be observed or clean")
        if spectrum_view == "clean" and split != "train":
            raise ValueError("clean spectrum view is available only for the train split")
        self.spectrum_view = spectrum_view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(self.root / str(sample["pattern_path"])) as artifact:
            theta = np.asarray(artifact["two_theta"], dtype=np.float32)
            if self.spectrum_view not in artifact.files:
                raise ValueError(f"pattern artifact does not contain {self.spectrum_view!r}")
            intensity = np.asarray(artifact[self.spectrum_view], dtype=np.float32)
        scale = max(float(np.quantile(intensity, 0.99)), 1e-8)
        model_intensity = intensity / scale
        if self.minimum_relative_intensity:
            model_intensity = model_intensity.copy()
            model_intensity[model_intensity < self.minimum_relative_intensity] = 0.0
        x = np.stack([(theta - 60.0) / 40.0, model_intensity], axis=0)
        rows = self.labels.get(str(sample["sample_id"]), [])
        classes = torch.tensor([
            self.class_index[(str(row["phase"]), int(row["h"]), int(row["k"]), int(row["l"]))]
            for row in rows
        ], dtype=torch.long)
        parameters = torch.tensor([
            [
                np.clip((float(row["center_deg"]) - 20.0) / 80.0, 0.0, 1.0),
                np.clip((float(row["fwhm_deg"]) - 0.03) / 1.17, 0.0, 1.0),
                np.clip(float(row["integrated_intensity"]) / 100.0, 0.0, 1.0),
                np.clip(float(row["eta"]), 0.0, 1.0),
                1.0,
            ]
            for row in rows
        ], dtype=torch.float32).reshape(-1, 5)
        target: dict[str, torch.Tensor] = {
            "classes": classes,
            "parameters": parameters,
            "grid_deg": torch.from_numpy(theta),
            "intensity": torch.from_numpy(intensity / scale),
            "intensity_scale": torch.tensor(scale, dtype=torch.float32),
        }
        if self.task == "full":
            count = np.zeros(len(theta), dtype=np.int16)
            for row in rows:
                radius = 1.5 * float(row["fwhm_deg"])
                count += np.abs(theta - float(row["center_deg"])) <= radius
            target["overlap_mask"] = torch.from_numpy((count >= 2).astype(np.float32))
        return {"sample_id": str(sample["sample_id"]), "x": torch.from_numpy(x), "target": target}


def _collate(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, list[Mapping[str, torch.Tensor]], list[str]]:
    return torch.stack([row["x"] for row in rows]), [row["target"] for row in rows], [str(row["sample_id"]) for row in rows]


def train_set_baseline(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    task: str,
    encoder_name: str,
    evaluation_split: str = "validation",
    seed: int = 2027,
    maximum_epochs: int = 50,
    patience: int = 10,
    batch_size: int | None = None,
    embedding_dim: int = 128,
    reconstruction_weight: float = 0.1,
    mask_weight: float = 1.0,
    class_weight: float = 1.0,
    parameter_weight: float = 1.0,
    existence_weight: float = 1.0,
    device_name: str | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_evaluation_samples: int | None = None,
    minimum_relative_intensity: float = 0.0,
    training_spectrum_view: str = "observed",
) -> dict[str, Any]:
    if task not in {"full", "window"}:
        raise ValueError("task must be full or window")
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    _seed_everything(seed)
    root = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_public_class_catalog(root, task)
    train_data = GammaSetDataset(root, task, "train", catalog, max_train_samples, minimum_relative_intensity, training_spectrum_view)
    validation_data = GammaSetDataset(root, task, "validation", catalog, max_validation_samples, minimum_relative_intensity)
    evaluation_data = (
        validation_data
        if evaluation_split == "validation" and max_evaluation_samples is None
        else GammaSetDataset(root, task, evaluation_split, catalog, max_evaluation_samples, minimum_relative_intensity)
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(batch_size or (64 if task == "full" else 256))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=_collate)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    evaluation_loader = DataLoader(evaluation_data, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    model = GammaSetPredictor(
        encoder_name=encoder_name,
        class_count=len(catalog),
        query_count=20 if task == "full" else 3,
        point_count=4001 if task == "full" else 151,
        overlap_mask=task == "full",
        embedding_dim=embedding_dim,
        input_channels=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    history = []
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(int(maximum_epochs)):
        model.train()
        losses = []
        for spectra, targets, _ in train_loader:
            spectra = spectra.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = hungarian_set_loss(
                model(spectra), targets,
                class_weight=class_weight,
                parameter_weight=parameter_weight,
                existence_weight=existence_weight,
                reconstruction_weight=reconstruction_weight,
                mask_weight=mask_weight if task == "full" else 0.0,
            )
            result["loss"].backward()
            optimizer.step()
            losses.append(float(result["loss"].detach().cpu()))
        predictions = _predict(model, validation_loader, catalog, device)
        epoch_submission = output / "epoch_validation_submission"
        _write_submission(validation_data, predictions, epoch_submission)
        score = evaluate_public_submission(root, task, "validation", epoch_submission, output / "epoch_validation_score")
        f1 = float(score["metrics"]["component_f1"])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_component_f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "catalog": catalog,
            "task": task,
            "encoder_name": encoder_name,
            "seed": seed,
            "embedding_dim": embedding_dim,
            "reconstruction_weight": reconstruction_weight,
            "mask_weight": mask_weight,
            "class_weight": class_weight,
            "parameter_weight": parameter_weight,
            "existence_weight": existence_weight,
            "minimum_relative_intensity": minimum_relative_intensity,
            "training_spectrum_view": training_spectrum_view,
        },
        output / "best_model.pt",
    )
    validation_predictions = _predict(model, validation_loader, catalog, device)
    validation_submission = output / "validation_submission"
    _write_submission(validation_data, validation_predictions, validation_submission)
    validation_score = evaluate_public_submission(
        root, task, "validation", validation_submission, output / "validation_score"
    )
    if evaluation_split == "validation":
        final_score = validation_score
    else:
        predictions = _predict(model, evaluation_loader, catalog, device)
        submission = output / f"{evaluation_split}_submission"
        _write_submission(evaluation_data, predictions, submission)
        final_score = evaluate_public_submission(
            root, task, evaluation_split, submission, output / f"{evaluation_split}_score"
        )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = {
        "status": "pass", "task": task, "encoder_name": encoder_name, "seed": seed,
        "evaluation_split": evaluation_split,
        "loss_weights": {
            "class": class_weight,
            "parameter": parameter_weight,
            "existence": existence_weight,
            "mask": mask_weight,
            "reconstruction": reconstruction_weight,
        },
        "minimum_relative_intensity": minimum_relative_intensity,
        "training_spectrum_view": training_spectrum_view,
        "epochs_completed": len(history), "best_validation_component_f1": best_f1,
        "validation_metrics": validation_score["metrics"],
        "evaluation_metrics": final_score["metrics"],
    }
    (output / "training_summary.json").write_text(
        json.dumps(_json_compatible(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


@torch.no_grad()
def _predict(model: GammaSetPredictor, loader: DataLoader, catalog: Sequence[tuple[str, int, int, int]], device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for spectra, _, sample_ids in loader:
        output = model(spectra.to(device))
        existence = output["existence_logits"].sigmoid().cpu()
        classes = output["class_logits"].argmax(-1).cpu()
        parameters = output["parameters"].cpu()
        for batch_index, sample_id in enumerate(sample_ids):
            for query_index in range(existence.shape[1]):
                confidence = float(existence[batch_index, query_index])
                if confidence < 0.5:
                    continue
                phase, h, k, l = catalog[int(classes[batch_index, query_index])]
                values = parameters[batch_index, query_index]
                rows.append({
                    "sample_id": sample_id, "component_id": f"query_{query_index:02d}",
                    "phase": phase, "h": h, "k": k, "l": l,
                    "hkl_group_id": hkl_group_id(phase, h, k, l),
                    "center_deg": 20.0 + 80.0 * float(values[0]),
                    "fwhm_deg": 0.03 + 1.17 * float(values[1]),
                    "integrated_intensity": 100.0 * float(values[2]),
                    "eta": float(values[3]), "confidence": confidence,
                })
    return rows


def _write_submission(dataset: GammaSetDataset, predictions: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id", "center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence"]
    frame = pd.DataFrame(predictions).reindex(columns=columns)
    frame.to_parquet(output / "submission_components.parquet", index=False)
    counts = frame.groupby("sample_id").size() if len(frame) else {}
    sample_ids = dataset.all_sample_ids
    pd.DataFrame({"sample_id": sample_ids, "predicted_component_count": [int(counts.get(value, 0)) for value in sample_ids]}).to_parquet(
        output / "submission_samples.parquet", index=False
    )


def _seed_everything(seed: int) -> None:
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
