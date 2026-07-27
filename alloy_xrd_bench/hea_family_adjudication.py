"""Expert adjudication workflow for HEA family labels.

Automated PTM/CNA evidence is exported for review but is never interpreted as
an expert decision by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .hea_validation_metrics import family_classification_metrics


FAMILY_LABELS = frozenset(
    {
        "fcc_like",
        "l12_like",
        "bcc_like",
        "b2_like",
        "hcp_like",
        "complex",
        "unknown",
    }
)
ALLOWED_DECISIONS = frozenset({"accept_current", "accept_proposed", "exclude"})
ADJUDICATION_COLUMNS = (
    "structure_state_id",
    "current_label",
    "proposed_label",
    "ptm_evidence",
    "cna_evidence",
    "reviewer_id",
    "review_date",
    "decision",
    "expert_label",
    "notes",
)


class AdjudicationValidationError(ValueError):
    """Raised when an adjudication file is incomplete or inconsistent."""


DOUBLE_BLIND_COLUMNS = (
    "structure_state_id",
    "blinded_sample_id",
    "reviewer_id",
    "formula",
    "chemical_system",
    "classification_confidence",
    "template_mapping_rms",
    "expert_label",
    "review_date",
    "notes",
)


def select_stratified_review_sample(
    structures: pd.DataFrame, *, sample_size: int = 200, seed: int = 2027
) -> pd.DataFrame:
    """Select a deterministic family-balanced review sample with risk coverage."""

    frame = _normalise_ids(structures)
    required = {
        "structure_family",
        "classification_confidence",
        "template_mapping_rms",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise AdjudicationValidationError(
            f"review sampling missing columns: {sorted(missing)}"
        )
    if sample_size <= 0 or sample_size > len(frame):
        raise AdjudicationValidationError("invalid review sample size")
    families = sorted(frame["structure_family"].astype(str).unique())
    if not families:
        raise AdjudicationValidationError("no structure families available")
    base, remainder = divmod(sample_size, len(families))
    rng = pd.Series(range(len(frame)), index=frame.index).map(
        lambda value: int.from_bytes(
            f"{seed}:{value}".encode("utf-8"), "little", signed=False
        )
    )
    frame = frame.assign(_tie_key=rng)
    selected = []
    for family_index, family in enumerate(families):
        quota = base + int(family_index < remainder)
        group = frame.loc[frame["structure_family"].astype(str).eq(family)].copy()
        if len(group) < quota:
            raise AdjudicationValidationError(
                f"family {family!r} has {len(group)} rows, requires {quota}"
            )
        group["_risk"] = (
            1.0 - group["classification_confidence"].astype(float).clip(0.0, 1.0)
        ) + group["template_mapping_rms"].astype(float).clip(lower=0.0)
        group = group.sort_values(
            ["_risk", "_tie_key", "structure_state_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        risk_count = quota // 2
        high_risk = group.iloc[:risk_count]
        remaining = group.iloc[risk_count:].sort_values(
            ["_tie_key", "structure_state_id"], kind="stable"
        )
        selected.append(pd.concat([high_risk, remaining.iloc[: quota - risk_count]]))
    result = pd.concat(selected, ignore_index=True)
    return result.drop(columns=["_tie_key", "_risk"], errors="ignore")


def build_double_blind_review(
    selected: pd.DataFrame, *, reviewer_ids: Iterable[str]
) -> pd.DataFrame:
    """Create two label-blinded review rows for every selected structure."""

    frame = _normalise_ids(selected)
    reviewers = [str(value).strip() for value in reviewer_ids]
    if len(reviewers) != 2 or len(set(reviewers)) != 2 or any(not value for value in reviewers):
        raise AdjudicationValidationError("exactly two independent reviewer IDs are required")
    rows = []
    for record in frame.to_dict(orient="records"):
        for slot, reviewer in enumerate(reviewers, start=1):
            rows.append(
                {
                    "structure_state_id": str(record["structure_state_id"]),
                    "blinded_sample_id": f"blind_{record['structure_state_id']}_{slot}",
                    "reviewer_id": reviewer,
                    "formula": str(record.get("formula", "")),
                    "chemical_system": str(record.get("chemical_system", "")),
                    "classification_confidence": float(
                        record.get("classification_confidence", 0.0)
                    ),
                    "template_mapping_rms": float(
                        record.get("template_mapping_rms", 0.0)
                    ),
                    "expert_label": pd.NA,
                    "review_date": pd.NA,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows, columns=DOUBLE_BLIND_COLUMNS)


def finalize_double_blind_review(
    structures: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    resolutions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Validate independent reviews, resolve conflicts, and compute release gates."""

    source = _normalise_ids(structures)
    missing = set(DOUBLE_BLIND_COLUMNS).difference(reviews.columns)
    if missing:
        raise AdjudicationValidationError(
            f"missing double-blind columns: {sorted(missing)}"
        )
    review = reviews.loc[:, DOUBLE_BLIND_COLUMNS].copy()
    unknown = set(review["structure_state_id"].astype(str)).difference(
        source["structure_state_id"].astype(str)
    )
    if unknown:
        raise AdjudicationValidationError(
            f"review contains unknown structure IDs: {sorted(unknown)[:5]}"
        )
    counts = review.groupby("structure_state_id").size()
    reviewers = review.groupby("structure_state_id")["reviewer_id"].nunique()
    if not counts.eq(2).all() or not reviewers.eq(2).all():
        raise AdjudicationValidationError(
            "every structure requires exactly two independent reviewers"
        )
    labels = review["expert_label"].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any() or (~labels.isin(FAMILY_LABELS)).any():
        raise AdjudicationValidationError("every review requires a valid expert_label")
    dates = review["review_date"].astype("string").str.strip()
    if dates.isna().any() or dates.eq("").any():
        raise AdjudicationValidationError("every review requires review_date")

    ordered = review.sort_values(
        ["structure_state_id", "reviewer_id"], kind="stable"
    )
    pivot = ordered.pivot(
        index="structure_state_id", columns="reviewer_id", values="expert_label"
    )
    reviewer_columns = list(pivot.columns)
    agreement = pivot[reviewer_columns[0]].eq(pivot[reviewer_columns[1]])
    disagreements = set(pivot.index[~agreement])
    resolution_map: dict[str, str] = {}
    if resolutions is not None and len(resolutions):
        required = {
            "structure_state_id",
            "consensus_label",
            "adjudicator_id",
            "adjudication_date",
            "notes",
        }
        missing_resolution = required.difference(resolutions.columns)
        if missing_resolution:
            raise AdjudicationValidationError(
                f"resolution missing columns: {sorted(missing_resolution)}"
            )
        if resolutions["structure_state_id"].duplicated().any():
            raise AdjudicationValidationError("duplicate disagreement resolution")
        resolution_ids = set(resolutions["structure_state_id"].astype(str))
        if not resolution_ids.issubset(disagreements):
            raise AdjudicationValidationError("resolution provided for a non-disagreement")
        if (~resolutions["consensus_label"].isin(FAMILY_LABELS)).any():
            raise AdjudicationValidationError("invalid consensus_label")
        for record in resolutions.to_dict(orient="records"):
            structure_id = str(record["structure_state_id"])
            adjudicator = str(record["adjudicator_id"]).strip()
            original_reviewers = set(
                review.loc[
                    review["structure_state_id"].astype(str).eq(structure_id),
                    "reviewer_id",
                ].astype(str)
            )
            if not adjudicator or adjudicator in original_reviewers:
                raise AdjudicationValidationError(
                    "disagreement requires an independent adjudicator"
                )
            resolution_map[structure_id] = str(record["consensus_label"])
    unresolved = disagreements.difference(resolution_map)
    if unresolved:
        raise AdjudicationValidationError(
            f"unresolved disagreement count: {len(unresolved)}"
        )

    consensus = {
        structure_id: (
            str(row.iloc[0]) if row.iloc[0] == row.iloc[1] else resolution_map[str(structure_id)]
        )
        for structure_id, row in pivot.iterrows()
    }
    reviewed_source = source.loc[
        source["structure_state_id"].astype(str).isin(consensus)
    ].copy()
    reviewed_source["expert_consensus_label"] = reviewed_source[
        "structure_state_id"
    ].astype(str).map(consensus)
    truth = reviewed_source["expert_consensus_label"].astype(str).tolist()
    prediction = reviewed_source["structure_family"].astype(str).tolist()
    classes = sorted(set(truth) | set(prediction))
    metrics = family_classification_metrics(truth, prediction, classes)
    kappa = float(
        cohen_kappa_score(
            pivot[reviewer_columns[0]], pivot[reviewer_columns[1]], labels=classes
        )
    )
    bcc_fcc_pass = all(
        metrics["per_class"].get(label, {}).get(metric, 0.0) >= 0.95
        for label in ("bcc_like", "fcc_like")
        if label in classes
        for metric in ("precision", "recall")
    )
    complex_pass = (
        "complex" not in classes
        or metrics["per_class"]["complex"]["recall"] >= 0.90
    )
    overall_accuracy = float(
        np.mean(np.asarray(truth, dtype=object) == np.asarray(prediction, dtype=object))
    )
    summary = {
        "reviewed_structure_count": len(consensus),
        "review_row_count": len(review),
        "reviewer_count": int(review["reviewer_id"].nunique()),
        "disagreement_count": len(disagreements),
        "pending_disagreement_count": 0,
        "cohen_kappa": kappa,
        "overall_accuracy": overall_accuracy,
        "family_metrics": metrics,
        "review_gate_passed": bool(
            overall_accuracy >= 0.95
            and kappa >= 0.80
            and bcc_fcc_pass
            and complex_pass
        ),
    }
    labels_v2 = source.copy()
    labels_v2["structure_family_v2"] = labels_v2["structure_state_id"].astype(str).map(
        consensus
    ).fillna(labels_v2["structure_family"])
    return {
        "summary": summary,
        "expert_review": review.reset_index(drop=True),
        "consensus": reviewed_source.reset_index(drop=True),
        "structure_labels_v2": labels_v2.reset_index(drop=True),
        "resolutions": resolutions.copy() if resolutions is not None else pd.DataFrame(),
    }


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _normalise_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    id_column = _first_column(result, ("structure_state_id", "dataset_structure_id"))
    if id_column is None:
        raise AdjudicationValidationError("missing structure/state ID column")
    if id_column != "structure_state_id":
        result = result.rename(columns={id_column: "structure_state_id"})
    return result


def _fraction_evidence(row: pd.Series, prefix: str) -> str:
    direct = f"{prefix}_evidence"
    if direct in row.index and pd.notna(row[direct]):
        return str(row[direct])
    values = {
        name.removeprefix(f"{prefix}_").removesuffix("_fraction"): float(row[name])
        for name in row.index
        if name.startswith(f"{prefix}_")
        and name.endswith("_fraction")
        and pd.notna(row[name])
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def build_adjudication_export(
    review_queue: pd.DataFrame, evidence: pd.DataFrame
) -> pd.DataFrame:
    """Build a pending human-review table from failed and conflict rows."""

    queue = _normalise_ids(review_queue)
    evidence_frame = _normalise_ids(evidence)

    if "review_status" in queue.columns:
        queue = queue.loc[queue["review_status"].eq("fail")].copy()
    if "review_decision" in evidence_frame.columns:
        conflicts = evidence_frame.loc[
            evidence_frame["review_decision"].eq("conflict")
        ].copy()
    else:
        conflicts = evidence_frame.iloc[0:0].copy()

    ids = list(dict.fromkeys(queue["structure_state_id"].tolist() + conflicts["structure_state_id"].tolist()))
    queue_by_id = queue.drop_duplicates("structure_state_id").set_index("structure_state_id")
    evidence_by_id = evidence_frame.drop_duplicates("structure_state_id").set_index("structure_state_id")

    rows = []
    for structure_id in ids:
        queue_row = queue_by_id.loc[structure_id] if structure_id in queue_by_id.index else None
        evidence_row = (
            evidence_by_id.loc[structure_id]
            if structure_id in evidence_by_id.index
            else None
        )
        current_label = _value_from_rows(
            (queue_row, evidence_row), ("current_label", "structure_family")
        )
        proposed_label = _value_from_rows(
            (queue_row,), ("proposed_label", "reviewer_label")
        )
        if not proposed_label or proposed_label not in FAMILY_LABELS:
            proposed_label = "unknown"
        evidence_values = evidence_row if evidence_row is not None else queue_row
        rows.append(
            {
                "structure_state_id": structure_id,
                "current_label": current_label,
                "proposed_label": proposed_label,
                "ptm_evidence": _fraction_evidence(evidence_values, "ptm"),
                "cna_evidence": _fraction_evidence(evidence_values, "cna"),
                "reviewer_id": pd.NA,
                "review_date": pd.NA,
                "decision": "pending",
                "expert_label": pd.NA,
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=ADJUDICATION_COLUMNS)


def _value_from_rows(
    rows: Iterable[pd.Series | None], columns: Iterable[str]
) -> str | None:
    for row in rows:
        if row is None:
            continue
        for column in columns:
            if column in row.index and pd.notna(row[column]):
                return str(row[column])
    return None


def validate_adjudications(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate completed independent expert decisions and return a copy."""

    missing = set(ADJUDICATION_COLUMNS).difference(frame.columns)
    if missing:
        raise AdjudicationValidationError(
            f"missing adjudication columns: {sorted(missing)}"
        )
    result = frame.loc[:, ADJUDICATION_COLUMNS].copy()
    if result["structure_state_id"].duplicated().any():
        raise AdjudicationValidationError("duplicate structure_state_id values")
    if result["structure_state_id"].isna().any() or result[
        "structure_state_id"
    ].astype(str).str.strip().eq("").any():
        raise AdjudicationValidationError("empty structure_state_id")

    for column in ("current_label", "proposed_label"):
        invalid = ~result[column].isin(FAMILY_LABELS)
        if invalid.any():
            raise AdjudicationValidationError(f"invalid {column} label")

    decisions = result["decision"].astype("string").str.strip()
    if decisions.eq("pending").any() or decisions.isna().any():
        raise AdjudicationValidationError("pending adjudication decision")
    if (~decisions.isin(ALLOWED_DECISIONS)).any():
        raise AdjudicationValidationError("invalid adjudication decision")

    for column in ("reviewer_id", "review_date"):
        values = result[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise AdjudicationValidationError(f"empty {column}")

    accepted = ~decisions.eq("exclude")
    expert_labels = result["expert_label"].astype("string").str.strip()
    if expert_labels.loc[accepted].isna().any() or expert_labels.loc[
        accepted
    ].eq("").any():
        raise AdjudicationValidationError(
            "expert_label is required for every accepted decision"
        )
    invalid_expert = accepted & ~expert_labels.isin(FAMILY_LABELS)
    if invalid_expert.any():
        raise AdjudicationValidationError("invalid expert_label")
    wrong_current = decisions.eq("accept_current") & (
        expert_labels != result["current_label"].astype("string")
    )
    wrong_proposed = decisions.eq("accept_proposed") & (
        expert_labels != result["proposed_label"].astype("string")
    )
    if wrong_current.any() or wrong_proposed.any():
        raise AdjudicationValidationError(
            "expert_label must match the explicitly accepted label"
        )
    return result


def apply_adjudications(
    manifest: pd.DataFrame,
    adjudications: pd.DataFrame,
    *,
    label_column: str = "structure_family",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply completed decisions to a copied manifest and retain provenance."""

    validated = validate_adjudications(adjudications)
    result = _normalise_ids(manifest)
    if label_column not in result.columns:
        raise AdjudicationValidationError(f"manifest missing {label_column}")
    unknown_ids = set(validated["structure_state_id"]).difference(
        result["structure_state_id"]
    )
    if unknown_ids:
        raise AdjudicationValidationError(
            f"adjudications contain IDs absent from manifest: {sorted(unknown_ids)[:5]}"
        )

    decisions = validated.set_index("structure_state_id")
    excluded_ids = set(
        validated.loc[validated["decision"].eq("exclude"), "structure_state_id"]
    )
    result = result.loc[~result["structure_state_id"].isin(excluded_ids)].copy()
    accepted = decisions.loc[~decisions["decision"].eq("exclude"), "expert_label"]
    result[label_column] = result["structure_state_id"].map(accepted).fillna(
        result[label_column]
    )
    return result.reset_index(drop=True), validated.reset_index(drop=True)


def _write_export(args: argparse.Namespace) -> None:
    output = Path(args.output)
    allowed_root = Path(args.output_root)
    try:
        output.resolve().relative_to(allowed_root.resolve())
    except ValueError as error:
        raise AdjudicationValidationError(
            f"output must resolve beneath {allowed_root}"
        ) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = build_adjudication_export(
        pd.read_parquet(args.review_queue), pd.read_parquet(args.evidence)
    )
    exported.to_parquet(output, index=False)
    print(json.dumps({"output": str(output), "rows": len(exported)}))


def _write_blind_export(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    structures = pd.read_parquet(args.structures)
    selected = select_stratified_review_sample(
        structures, sample_size=args.sample_size, seed=args.seed
    )
    review = build_double_blind_review(
        selected, reviewer_ids=[args.reviewer_a, args.reviewer_b]
    )
    manifests = output_root / "manifests"
    manifests.mkdir(exist_ok=True)
    selected_path = manifests / "expert_review_sample_v1.parquet"
    review_path = manifests / "expert_review_v1.parquet"
    selected.to_parquet(selected_path, index=False)
    review.to_parquet(review_path, index=False)
    print(
        json.dumps(
            {
                "status": "pass",
                "selected_count": len(selected),
                "review_row_count": len(review),
                "sample_path": str(selected_path),
                "review_path": str(review_path),
            },
            sort_keys=True,
        )
    )


def _finalize_blind(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    manifests = output_root / "manifests"
    reports = output_root / "reports"
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    resolutions = (
        pd.read_parquet(args.resolutions) if args.resolutions is not None else None
    )
    result = finalize_double_blind_review(
        pd.read_parquet(args.structures),
        pd.read_parquet(args.reviews),
        resolutions=resolutions,
    )
    result["expert_review"].to_parquet(
        manifests / "expert_review_v1.parquet", index=False
    )
    result["structure_labels_v2"].to_parquet(
        manifests / "structure_labels_v2.parquet", index=False
    )
    result["consensus"].to_parquet(
        manifests / "expert_review_consensus_v1.parquet", index=False
    )
    summary = result["summary"]
    (reports / "label_error_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    gate = {
        "status": "pass" if summary["review_gate_passed"] else "fail",
        "label_gate_passed": bool(summary["review_gate_passed"]),
        "benchmark_ready": False,
        "benchmark_ready_reason": "remaining_data_and_model_gates_not_evaluated",
        "review": summary,
    }
    (reports / "hea_scientific_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(gate, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--review-queue", required=True)
    export.add_argument("--evidence", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--output-root", default="/home/GJM/data/hea/benchmark_v1_1"
    )
    export.set_defaults(handler=_write_export)
    blind_export = subparsers.add_parser("blind-export")
    blind_export.add_argument("--structures", required=True)
    blind_export.add_argument("--output-root", required=True)
    blind_export.add_argument("--sample-size", type=int, default=200)
    blind_export.add_argument("--seed", type=int, default=2027)
    blind_export.add_argument("--reviewer-a", required=True)
    blind_export.add_argument("--reviewer-b", required=True)
    blind_export.set_defaults(handler=_write_blind_export)
    blind_finalize = subparsers.add_parser("blind-finalize")
    blind_finalize.add_argument("--structures", required=True)
    blind_finalize.add_argument("--reviews", required=True)
    blind_finalize.add_argument("--resolutions")
    blind_finalize.add_argument("--output-root", required=True)
    blind_finalize.set_defaults(handler=_finalize_blind)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
