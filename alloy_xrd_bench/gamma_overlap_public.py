"""Public packaging, loading, and evaluation for the gamma overlap benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .gamma_overlap_physics import canonical_cubic_hkl
from .gamma_overlap_scoring import (
    build_scoring_components,
    hkl_group_id,
    score_component_predictions,
)
from .gamma_overlap_baseline import build_training_candidate_catalog


COMPONENT_COLUMNS = [
    "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
    "center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence",
]


def _public_split(value: object) -> str:
    return "test" if str(value) == "private_test" else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_artifacts(source: Path, destination: Path, rows: pd.DataFrame, path_column: str) -> pd.DataFrame:
    destination.mkdir(parents=True, exist_ok=True)
    result = rows.copy()
    public_paths = []
    hashes = []
    for record in result.to_dict(orient="records"):
        relative = str(record[path_column])
        source_path = source / relative
        target = destination / source_path.name
        with np.load(source_path) as artifact:
            payload = {
                "two_theta": np.asarray(artifact["two_theta"]),
                "observed": np.asarray(artifact["observed"]),
            }
            if str(record["split"]) == "train" and "clean" in artifact.files:
                payload["clean"] = np.asarray(artifact["clean"])
            np.savez_compressed(target, **payload)
        public_paths.append(target.relative_to(destination.parent.parent).as_posix())
        hashes.append(_sha256(target))
    result["artifact_path"] = public_paths
    result["artifact_sha256"] = hashes
    return result


INSTRUMENT_FIELDS = [
    "instrument_fwhm_deg", "microstrain_fraction", "crystallite_size_nm",
    "zero_shift_deg", "background_fraction", "gaussian_noise_fraction", "poisson_scale",
]


def _write_instrument_slices(core: Path, samples: pd.DataFrame) -> None:
    output = core / "slices"
    output.mkdir()
    available = [field for field in INSTRUMENT_FIELDS if field in samples.columns]
    train = samples.loc[samples["split"].astype(str).eq("train")]
    minimum_outside_fields = min(2, len(available)) if available else 0
    thresholds: dict[str, Any] = {
        "fitted_split": "train",
        "quantiles": [0.1, 0.9],
        "minimum_outside_fields": minimum_outside_fields,
        "fields": {},
    }
    for field in available:
        thresholds["fields"][field] = {
            "lower": float(train[field].astype(float).quantile(0.1)),
            "upper": float(train[field].astype(float).quantile(0.9)),
        }
    rows = []
    for row in samples.to_dict(orient="records"):
        outside = 0
        for field, limits in thresholds["fields"].items():
            value = float(row[field])
            outside += int(value < limits["lower"] or value > limits["upper"])
        rows.append({
            "sample_id": str(row["sample_id"]),
            "split": _public_split(row["split"]),
            "instrument_slice": (
                "not_available"
                if not available
                else "instrument_ood"
                if outside >= minimum_outside_fields
                else "instrument_id"
            ),
            "severity_score": float(outside / len(available)) if available else 0.0,
        })
    thresholds["status"] = "pass" if available else "not_available"
    (output / "instrument_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(rows).to_parquet(output / "instrument_slices.parquet", index=False)


def export_public_release(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    include_provenance: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    core = output / "gamma-decomp-core-v1.1"
    if core.exists():
        raise FileExistsError(f"public release already exists: {core}")
    (core / "full").mkdir(parents=True)
    (core / "window").mkdir()

    full_samples = pd.read_parquet(source / "mixture_samples.parquet").copy()
    _write_instrument_slices(core, full_samples)
    full_samples["split"] = full_samples["split"].map(_public_split)
    full_samples = _copy_artifacts(
        source, core / "full" / "patterns", full_samples, "pattern_path"
    )
    full_groups = full_samples[["sample_id", "parent_pair_id", "split"]].copy()

    raw_components = pd.concat(
        [
            pd.read_parquet(source / "components.parquet"),
            pd.read_parquet(source / "private/components.parquet"),
        ],
        ignore_index=True,
    )
    raw_components["split"] = raw_components["split"].map(_public_split)
    components = pd.DataFrame(build_scoring_components(raw_components.to_dict(orient="records")))
    label_columns = [
        "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
        "allowed_hkls_json", "degenerate_group", "merged_integrated_intensity",
        "center_deg", "fwhm_deg", "integrated_intensity", "eta", "score_eligible", "split",
    ]
    components[label_columns].to_parquet(core / "full" / "components.parquet", index=False)

    public_windows = pd.read_parquet(source / "overlap_windows.parquet")
    windows = pd.concat(
        [
            public_windows.loc[public_windows["split"].astype(str).ne("private_test")],
            pd.read_parquet(source / "private/overlap_windows_truth.parquet"),
        ],
        ignore_index=True,
    )
    windows["split"] = windows["split"].map(_public_split)
    windows = _copy_artifacts(source, core / "window" / "patterns", windows, "window_path")
    instrument_slices_path = core / "slices" / "instrument_slices.parquet"
    full_instrument_slices = pd.read_parquet(instrument_slices_path)
    window_instrument_slices = (
        windows[["window_id", "sample_id", "split"]]
        .merge(
            full_instrument_slices[["sample_id", "instrument_slice", "severity_score"]],
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
        .drop(columns="sample_id")
        .rename(columns={"window_id": "sample_id"})
    )
    pd.concat([full_instrument_slices, window_instrument_slices], ignore_index=True).to_parquet(
        instrument_slices_path, index=False
    )
    windows["sample_id"] = windows["window_id"].astype(str)
    windows["task_id"] = "gamma_window_decomp_v1"
    windows["pattern_path"] = windows["artifact_path"]
    window_groups = windows[["sample_id", "parent_pair_id", "split"]].copy()
    component_by_id = {
        str(row["component_id"]): row for row in raw_components.to_dict(orient="records")
    }
    window_truth_rows = []
    scored_counts = []
    context_ids_json = []
    for window in windows.to_dict(orient="records"):
        target_ids = [str(value) for value in json.loads(str(window["target_component_ids_json"]))]
        projected = []
        for source_component_id in target_ids:
            component = dict(component_by_id[source_component_id])
            component["sample_id"] = str(window["sample_id"])
            component["component_id"] = f"{window['sample_id']}::{source_component_id}"
            component["split"] = str(window["split"])
            projected.append(component)
        grouped_window = build_scoring_components(projected)
        scored_counts.append(sum(bool(row["score_eligible"]) for row in grouped_window))
        context_ids_json.append(json.dumps([
            str(row["component_id"]) for row in grouped_window if not bool(row["score_eligible"])
        ]))
        window_truth_rows.extend(grouped_window)
    windows["scored_component_count"] = scored_counts
    windows["context_component_ids_json"] = context_ids_json
    windows[[
        "sample_id",
        "split",
        "difficulty",
        "minimum_separation_ratio",
        "scored_component_count",
        "context_component_ids_json",
    ]].to_parquet(core / "slices" / "overlap_slices.parquet", index=False)
    window_truth = pd.DataFrame(window_truth_rows)
    window_truth[label_columns].to_parquet(core / "window" / "components.parquet", index=False)

    pairs = pd.read_parquet(source / "parent_pairs.parquet").copy()
    pairs["split"] = pairs["split"].map(_public_split)
    full_samples["task_id"] = "gamma_full_decomp_v1"
    full_samples["pattern_path"] = full_samples["artifact_path"]
    full_samples = full_samples[
        ["sample_id", "task_id", "split", "point_count", "pattern_path", "artifact_sha256"]
    ]
    full_samples.to_parquet(core / "full" / "samples.parquet", index=False)
    window_columns = [
        "sample_id", "task_id", "split", "pattern_path", "artifact_sha256",
    ]
    windows["point_count"] = 151
    window_columns.insert(3, "point_count")
    windows[window_columns].to_parquet(core / "window" / "samples.parquet", index=False)
    full_groups.to_parquet(core / "full" / "evaluation_groups.parquet", index=False)
    window_groups.to_parquet(core / "window" / "evaluation_groups.parquet", index=False)
    baseline_dir = core / "baselines"
    baseline_dir.mkdir()
    baseline_catalog = (
        components.loc[components["split"].astype(str).eq("train")]
        .groupby(["phase", "h", "k", "l", "hkl_group_id"], as_index=False)["center_deg"]
        .median()
        .to_dict(orient="records")
    )
    pd.DataFrame(baseline_catalog).to_parquet(
        baseline_dir / "training_candidate_catalog.parquet", index=False
    )
    if include_provenance:
        provenance = output / "gamma-decomp-provenance-v1.1"
        provenance.mkdir(parents=True)
        for name in (
            "parent_structures.parquet", "parent_pairs.parquet", "phase_patterns.parquet",
            "phase_reflections.parquet", "satellite_reflections.parquet", "provenance_edges.parquet",
        ):
            shutil.copy2(source / name, provenance / name)
        if (source / "structures").exists():
            shutil.copytree(source / "structures", provenance / "structures")
        if (source / "audit").exists():
            shutil.copytree(source / "audit", provenance / "source_audit")
    metadata = {
        "schema_version": "nickel-gamma-overlap-public-v1.1",
        "source_release_id": json.loads((source / "summary.json").read_text()).get("release_id"),
        "tasks": ["full", "window"],
        "splits": ["train", "validation", "test"],
        "private_truth_included": True,
    }
    (core / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    def split_counts(rows: pd.DataFrame) -> dict[str, int]:
        counts = rows["split"].value_counts().to_dict()
        return {split: int(counts.get(split, 0)) for split in ("train", "validation", "test")}

    registry = {
        "release_id": "gamma-decomp-v1.1",
        "independent_track": True,
        "test_labels_public": True,
        "allowed_model_inputs": ["sample_id", "two_theta", "observed_intensity", "task_id"],
        "training_ablation_only_inputs": ["clean_intensity"],
        "tasks": {
            "gamma_full_decomp_v1": {"alias": "full", "point_count": 4001, "query_count": 20, "split_counts": split_counts(full_samples)},
            "gamma_window_decomp_v1": {"alias": "window", "point_count": 151, "query_count": 3, "split_counts": split_counts(windows)},
        },
    }
    (core / "task_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True))
    (core / "dataset_card.md").write_text(
        """# Gamma-Decomp v1

Gamma-Decomp v1 is the independent Ni-Al Track B of Alloy-XRD-Bench. It contains synthetic Cu K-alpha powder XRD patterns generated from audited gamma and gamma-prime parent structures.

## Tasks

- `gamma_full_decomp_v1`: predict the unordered physical reflection-component set for a 20-100 degree pattern and locate overlap regions.
- `gamma_window_decomp_v1`: classify local windows by scored component count and predict the unordered component set.

Each component is labeled by phase, canonical cubic HKL, center, FWHM, integrated intensity, pseudo-Voigt eta, and score eligibility. Weak context components are published but excluded from leaderboard matching.

## Inputs And Splits

Allowed model inputs are sample ID, task ID, two-theta, and observed intensity. Parent structures, parent-pair IDs, source runs, phase fractions, generation parameters, and random seeds are not model inputs. Internal `private_test` is published as `test`; test labels are public for reproducibility, but test must not be used for tuning.

Train artifacts additionally include `clean` intensity only for the frozen no-perturbation-training ablation. Validation and test artifacts never expose `clean`; it is not an allowed input for main benchmark submissions.

## Evaluation

The evaluator uses phase/HKL-gated Hungarian matching and parent-pair macro averaging. It reports component F1, count and phase/HKL accuracy, peak-parameter errors, profile and forward-reconstruction errors, overlap-region F1, and window single/multiple F1.

## Scope

This release evaluates overlapping-peak decomposition, not elemental site occupancy, lattice parameters, phase-fraction ranking, or gamma/gamma-prime lattice mismatch. It is synthetic Ni-Al data and is not presented as experimental superalloy diffraction.
""",
        encoding="utf-8",
    )
    split_dir = core / "splits"
    split_dir.mkdir()
    split_rows = pd.concat(
        [
            full_samples[["sample_id", "split"]].assign(task="full"),
            windows[["sample_id", "split"]].assign(task="window"),
        ],
        ignore_index=True,
    )
    for split in ("train", "validation", "test"):
        split_rows.loc[split_rows["split"].eq(split)].to_parquet(split_dir / f"{split}.parquet", index=False)
    schema_dir = core / "schemas"
    schema_dir.mkdir()
    (schema_dir / "submission_samples.schema.json").write_text(json.dumps({
        "type": "object", "required": ["sample_id", "predicted_component_count"],
        "additionalProperties": False,
    }, indent=2))
    (schema_dir / "submission_components.schema.json").write_text(json.dumps({
        "type": "object", "required": COMPONENT_COLUMNS, "additionalProperties": False,
    }, indent=2))
    report = validate_public_release(core)
    audit_dir = core / "audits"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_checksums(core)
    if include_provenance:
        _write_checksums(output / "gamma-decomp-provenance-v1.1")
    report["core_dir"] = str(core)
    return report


def validate_public_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir).resolve()
    issues = []
    if (root / "private").exists():
        issues.append("private_truth_present")
    overlap_slices_path = root / "slices" / "overlap_slices.parquet"
    if not overlap_slices_path.exists():
        issues.append("window:missing_overlap_slice_manifest")
    for task in ("full", "window"):
        samples_path = root / task / "samples.parquet"
        components_path = root / task / "components.parquet"
        if not samples_path.exists() or not components_path.exists():
            issues.append(f"{task}:missing_manifest")
            continue
        samples = pd.read_parquet(samples_path)
        allowed_columns = {
            "sample_id", "task_id", "split", "point_count", "pattern_path", "artifact_sha256"
        }
        if set(samples.columns) != allowed_columns:
            issues.append(f"{task}:invalid_public_sample_fields")
        if set(samples["split"].astype(str)) != {"train", "validation", "test"}:
            issues.append(f"{task}:invalid_splits")
        if task == "window" and overlap_slices_path.exists():
            overlap_slices = pd.read_parquet(overlap_slices_path)
            if (
                overlap_slices["sample_id"].astype(str).duplicated().any()
                or set(overlap_slices["sample_id"].astype(str))
                != set(samples["sample_id"].astype(str))
            ):
                issues.append("window:invalid_overlap_slice_coverage")
        if samples["sample_id"].astype(str).duplicated().any():
            issues.append(f"{task}:duplicate_samples")
        components = pd.read_parquet(components_path)
        if not set(components["sample_id"].astype(str)) <= set(samples["sample_id"].astype(str)):
            issues.append(f"{task}:orphan_components")
        if components.empty or not components["split"].astype(str).eq("test").any():
            issues.append(f"{task}:test_truth_missing")
        for row in samples.itertuples():
            artifact = root / row.pattern_path
            if not artifact.exists() or _sha256(artifact) != row.artifact_sha256:
                issues.append(f"{task}:artifact_integrity")
                break
        groups = pd.read_parquet(root / task / "evaluation_groups.parquet")
        if groups.groupby("parent_pair_id")["split"].nunique().max() > 1:
            issues.append(f"{task}:parent_pair_split_leakage")
        for row in samples.itertuples():
            artifact = root / row.pattern_path
            if not artifact.exists():
                continue
            with np.load(artifact) as pattern:
                expected_arrays = (
                    {"two_theta", "observed", "clean"}
                    if str(row.split) == "train"
                    else {"two_theta", "observed"}
                )
                if set(pattern.files) != expected_arrays:
                    issues.append(f"{task}:{row.split}:invalid_pattern_arrays")
                    break
        instrument_slices_path = root / "slices" / "instrument_slices.parquet"
        if not instrument_slices_path.exists():
            issues.append(f"{task}:missing_instrument_slice_manifest")
        else:
            instrument_slices = pd.read_parquet(instrument_slices_path)
            task_instrument = instrument_slices.loc[
                instrument_slices["sample_id"].astype(str).isin(samples["sample_id"].astype(str))
            ]
            if (
                task_instrument["sample_id"].astype(str).duplicated().any()
                or set(task_instrument["sample_id"].astype(str))
                != set(samples["sample_id"].astype(str))
            ):
                issues.append(f"{task}:invalid_instrument_slice_coverage")
    report = {"status": "pass" if not issues else "fail", "issues": issues}
    return report


class GammaOverlapDataset:
    def __init__(self, release_dir: str | Path, *, task: str, split: str):
        if task not in {"full", "window"} or split not in {"train", "validation", "test"}:
            raise ValueError("invalid task or split")
        self.root = Path(release_dir).resolve()
        self.task = task
        samples = pd.read_parquet(self.root / task / "samples.parquet")
        self.rows = samples.loc[samples["split"].astype(str).eq(split)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        with np.load(self.root / str(row["pattern_path"])) as artifact:
            x = np.column_stack((artifact["two_theta"], artifact["observed"])).astype(np.float32)
        return {"sample_id": str(row["sample_id"]), "x": x}


def make_submission_template(
    release_dir: str | Path,
    *,
    task: str,
    split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    samples = pd.read_parquet(root / task / "samples.parquet")
    samples = samples.loc[samples["split"].astype(str).eq(split)].copy()
    template = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].astype(str),
            "predicted_component_count": 0,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    template.to_parquet(output / "submission_samples.parquet", index=False)
    pd.DataFrame(columns=COMPONENT_COLUMNS).to_parquet(
        output / "submission_components.parquet", index=False
    )
    return {"sample_count": len(template), "output_dir": str(output)}


def make_oracle_submission(
    release_dir: str | Path,
    *,
    task: str,
    split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write an exact public-label submission for evaluator control tests."""
    root = Path(release_dir).resolve()
    output = Path(output_dir).resolve()
    samples = pd.read_parquet(root / task / "samples.parquet")
    samples = samples.loc[samples["split"].astype(str).eq(split)].copy()
    truth = pd.read_parquet(root / task / "components.parquet")
    truth = truth.loc[
        truth["split"].astype(str).eq(split) & truth["score_eligible"].astype(bool)
    ].copy()
    oracle = truth[[
        "sample_id", "component_id", "phase", "h", "k", "l", "hkl_group_id",
        "center_deg", "fwhm_deg", "integrated_intensity", "eta",
    ]].copy()
    oracle["component_id"] = "oracle::" + oracle["component_id"].astype(str)
    oracle["confidence"] = 1.0
    output.mkdir(parents=True, exist_ok=True)
    oracle.to_parquet(output / "submission_components.parquet", index=False)
    counts = oracle.groupby("sample_id").size()
    pd.DataFrame({
        "sample_id": samples["sample_id"].astype(str),
        "predicted_component_count": samples["sample_id"].astype(str).map(counts).fillna(0).astype(int),
    }).to_parquet(output / "submission_samples.parquet", index=False)
    return {
        "sample_count": len(samples),
        "component_count": len(oracle),
        "output_dir": str(output),
    }


def evaluate_public_submission(
    release_dir: str | Path,
    task: str,
    split: str,
    submission_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(release_dir).resolve()
    submission = Path(submission_dir).resolve()
    output = Path(output_dir).resolve()
    task_samples = pd.read_parquet(root / task / "samples.parquet")
    task_samples = task_samples.loc[task_samples["split"].astype(str).eq(split)].copy()
    id_column = "sample_id"
    expected = task_samples[id_column].astype(str).tolist()
    group_rows = pd.read_parquet(root / task / "evaluation_groups.parquet")
    group_rows = group_rows.loc[group_rows["split"].astype(str).eq(split)]
    parents = dict(zip(group_rows["sample_id"].astype(str), group_rows["parent_pair_id"].astype(str)))
    sample_predictions = pd.read_parquet(submission / "submission_samples.parquet")
    if set(sample_predictions.columns) != {"sample_id", "predicted_component_count"}:
        raise ValueError("submission_samples must contain exactly sample_id and predicted_component_count")
    if sample_predictions["sample_id"].astype(str).duplicated().any() or set(sample_predictions["sample_id"].astype(str)) != set(expected):
        raise ValueError("submission must provide exact sample coverage")
    predictions = pd.read_parquet(submission / "submission_components.parquet")
    missing = set(COMPONENT_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"submission components missing fields: {sorted(missing)}")
    if set(predictions.columns) != set(COMPONENT_COLUMNS):
        raise ValueError("submission components contain unexpected fields")
    if predictions.duplicated(["sample_id", "component_id"]).any():
        raise ValueError("submission contains duplicate component IDs")
    if not set(predictions["sample_id"].astype(str)).issubset(expected):
        raise ValueError("submission contains unexpected sample IDs")
    actual_counts = predictions.groupby(predictions["sample_id"].astype(str)).size()
    declared = sample_predictions.set_index(sample_predictions["sample_id"].astype(str))["predicted_component_count"]
    if any(int(declared[sample_id]) != int(actual_counts.get(sample_id, 0)) for sample_id in expected):
        raise ValueError("predicted component counts do not match component rows")
    numeric = ["center_deg", "fwhm_deg", "integrated_intensity", "eta", "confidence"]
    if len(predictions) and not np.isfinite(predictions[numeric].astype(float).to_numpy()).all():
        raise ValueError("component parameters must be finite")
    if len(predictions) and (
        (predictions["fwhm_deg"].astype(float) <= 0).any()
        or (predictions["integrated_intensity"].astype(float) < 0).any()
    ):
        raise ValueError("FWHM must be positive and integrated intensity non-negative")
    if len(predictions) and (
        not predictions["eta"].astype(float).between(0, 1).all()
        or not predictions["confidence"].astype(float).between(0, 1).all()
    ):
        raise ValueError("eta and confidence must be in [0, 1]")
    training_labels = pd.read_parquet(root / task / "components.parquet")
    training_labels = training_labels.loc[training_labels["split"].astype(str).eq("train")]
    legal = set(training_labels["hkl_group_id"].astype(str))
    if any(
        str(row.hkl_group_id) not in legal
        or str(row.hkl_group_id) != hkl_group_id(str(row.phase), int(row.h), int(row.k), int(row.l))
        for row in predictions.itertuples()
    ):
        raise ValueError("submission contains phase/HKL outside the training catalog")
    truth = pd.read_parquet(root / task / "components.parquet")
    truth = truth.loc[truth["split"].astype(str).eq(split) & truth["score_eligible"].astype(bool)]
    metrics = score_component_predictions(
        truth.to_dict(orient="records"),
        predictions.to_dict(orient="records"),
        expected_sample_ids=expected,
        expected_parent_ids=parents,
    )
    per_parent = []
    for parent_id, parent_samples in group_rows.groupby("parent_pair_id"):
        ids = parent_samples["sample_id"].astype(str).tolist()
        truth_rows = truth.loc[truth["sample_id"].astype(str).isin(ids)].to_dict(orient="records")
        prediction_rows = predictions.loc[predictions["sample_id"].astype(str).isin(ids)].to_dict(orient="records")
        result = score_component_predictions(
            truth_rows,
            prediction_rows,
            expected_sample_ids=ids,
            expected_parent_ids={sample_id: str(parent_id) for sample_id in ids},
        )
        per_parent.append({"parent_pair_id": str(parent_id), **result})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_parent).to_parquet(output / "per_parent_pair.parquet", index=False)
    per_difficulty = []
    if task == "window":
        overlap_slices_path = root / "slices" / "overlap_slices.parquet"
        if not overlap_slices_path.exists():
            raise ValueError("window evaluation requires slices/overlap_slices.parquet")
        overlap_slices = pd.read_parquet(overlap_slices_path)
        overlap_slices = overlap_slices.loc[
            overlap_slices["split"].astype(str).eq(split)
        ]
        if set(overlap_slices["sample_id"].astype(str)) != set(expected):
            raise ValueError("overlap slice manifest must provide exact sample coverage")
        for difficulty, difficulty_samples in overlap_slices.groupby("difficulty", dropna=False):
            ids = difficulty_samples[id_column].astype(str).tolist()
            difficulty_truth = truth.loc[truth["sample_id"].astype(str).isin(ids)].to_dict(orient="records")
            difficulty_predictions = predictions.loc[predictions["sample_id"].astype(str).isin(ids)].to_dict(orient="records")
            result = score_component_predictions(
                difficulty_truth,
                difficulty_predictions,
                expected_sample_ids=ids,
                expected_parent_ids={sample_id: parents[sample_id] for sample_id in ids},
            )
            per_difficulty.append({"difficulty": str(difficulty), **result})
    pd.DataFrame(per_difficulty or [{"difficulty": "not_applicable"}]).to_parquet(
        output / "per_difficulty.parquet", index=False
    )
    instrument_slices_path = root / "slices" / "instrument_slices.parquet"
    if not instrument_slices_path.exists():
        raise ValueError("evaluation requires slices/instrument_slices.parquet")
    instrument_slices = pd.read_parquet(instrument_slices_path)
    instrument_slices = instrument_slices.loc[
        instrument_slices["split"].astype(str).eq(split)
        & instrument_slices["sample_id"].astype(str).isin(expected)
    ]
    if set(instrument_slices["sample_id"].astype(str)) != set(expected):
        raise ValueError("instrument slice manifest must provide exact sample coverage")
    per_instrument_slice = []
    for instrument_slice, slice_samples in instrument_slices.groupby("instrument_slice"):
        ids = slice_samples["sample_id"].astype(str).tolist()
        slice_truth = truth.loc[truth["sample_id"].astype(str).isin(ids)].to_dict(orient="records")
        slice_predictions = predictions.loc[
            predictions["sample_id"].astype(str).isin(ids)
        ].to_dict(orient="records")
        result = score_component_predictions(
            slice_truth,
            slice_predictions,
            expected_sample_ids=ids,
            expected_parent_ids={sample_id: parents[sample_id] for sample_id in ids},
        )
        per_instrument_slice.append({"instrument_slice": str(instrument_slice), **result})
    pd.DataFrame(per_instrument_slice).to_parquet(
        output / "per_instrument_slice.parquet", index=False
    )
    scorecard = {"schema_version": "nickel-gamma-overlap-score-v1", "task": task, "split": split, "sample_count": len(expected), "metrics": metrics}
    (output / "scorecard.json").write_text(
        json.dumps(_json_compatible(scorecard), indent=2, sort_keys=True, allow_nan=False)
    )
    (output / "submission_validation.json").write_text(json.dumps({
        "status": "pass", "sample_count": len(sample_predictions), "component_count": len(predictions),
    }, indent=2, sort_keys=True))
    return scorecard


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value
