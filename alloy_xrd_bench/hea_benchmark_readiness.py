"""Machine-readable scientific readiness gate for the HEA benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


EVIDENCE = {
    "canonical": "canonical_audit.json",
    "evaluator": "evaluator_smoke.json",
    "controls": "controls.json",
    "predictions": "prediction_completeness.json",
    "family": "family_support.json",
    "identifiability": "identifiability.json",
    "model_gain": "model_comparison.json",
    "ood": "ood_summary.json",
    "perturbation": "perturbation_summary.json",
    "bootstrap": "bootstrap_summary.json",
    "release_package": "release_package.json",
}


def _read(root: Path, key: str) -> dict[str, Any] | None:
    path = root / EVIDENCE[key]
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def assess_readiness(root: Path) -> dict[str, Any]:
    evidence = {key: _read(root, key) for key in EVIDENCE}
    core_gates = {
        "canonical_immutable": bool(
            evidence["canonical"]
            and evidence["canonical"].get("status") == "pass"
            and int(evidence["canonical"].get("canonical_count", 0)) == 84019
        ),
        "evaluator": bool(
            evidence["evaluator"] and evidence["evaluator"].get("status") == "pass"
        ),
        "controls": bool(
            evidence["controls"]
            and evidence["controls"].get("control_gate_passed")
        ),
        "prediction_matrix": bool(
            evidence["predictions"]
            and evidence["predictions"].get("status") == "pass"
            and int(evidence["predictions"].get("required_run_count", 0)) == 84
            and int(evidence["predictions"].get("validated_run_count", 0)) == 84
            and int(evidence["predictions"].get("missing_count", 0)) == 0
            and int(evidence["predictions"].get("duplicate_count", 0)) == 0
            and int(evidence["predictions"].get("unexpected_count", 0)) == 0
        ),
        "ood": bool(evidence["ood"] and evidence["ood"].get("status") == "pass"),
        "perturbation": bool(
            evidence["perturbation"]
            and evidence["perturbation"].get("status") == "pass"
            and int(evidence["perturbation"].get("seed_count", 0)) >= 5
        ),
        "bootstrap": bool(
            evidence["bootstrap"]
            and evidence["bootstrap"].get("status") == "pass"
            and int(evidence["bootstrap"].get("replicates", 0)) >= 2000
            and int(evidence["bootstrap"].get("seed", -1)) == 2027
        ),
        "release_package": bool(
            evidence["release_package"]
            and evidence["release_package"].get("status") == "pass"
            and int(evidence["release_package"].get("checksum_errors", 0)) == 0
            and int(evidence["release_package"].get("license_errors", 0)) == 0
        ),
    }
    codes = {
        "canonical_immutable": "CANONICAL_AUDIT_FAILED",
        "evaluator": "EVALUATOR_NOT_VALIDATED",
        "controls": "CONTROL_GATE_FAILED",
        "prediction_matrix": "PREDICTION_MATRIX_INCOMPLETE",
        "ood": "OOD_EVALUATION_MISSING",
        "perturbation": "PERTURBATION_EVALUATION_MISSING",
        "bootstrap": "BOOTSTRAP_EVIDENCE_MISSING",
        "release_package": "RELEASE_PACKAGE_INCOMPLETE",
    }
    failure_codes = [codes[name] for name, passed in core_gates.items() if not passed]
    core_ready = all(core_gates.values())

    family = evidence["family"] or {}
    pending_reviews = int(family.get("pending_review_count", 0))
    if not family:
        family_status = "diagnostic_not_generated"
    elif pending_reviews:
        family_status = "diagnostic_pending_review"
    else:
        family_status = "diagnostic_only"

    identifiability = evidence["identifiability"] or {}
    identifiable_groups = int(
        identifiability.get("identifiable_parent_group_count", 0)
    )
    if not identifiability:
        occupancy_status = "diagnostic_not_generated"
    elif identifiable_groups == 0:
        occupancy_status = "diagnostic_non_identifiable"
    else:
        occupancy_status = "diagnostic_identifiable_subset"
    experimental_noise_calibrated = bool(
        identifiability.get("experimental_noise_calibrated", False)
    )
    return {
        "schema_version": "sitexrd-hea-readiness-v1.1",
        "track": "hea",
        "release_scope": "sro_core",
        "gamma_included": False,
        "gates": core_gates,
        "failure_codes": failure_codes,
        "core_benchmark_ready": core_ready,
        "benchmark_release_ready": core_ready,
        "full_production_allowed": core_ready,
        "diagnostic_family_status": family_status,
        "diagnostic_occupancy_status": occupancy_status,
        "experimental_noise_calibrated": experimental_noise_calibrated,
        "model_gain_is_release_gate": False,
        "decision": "GO" if core_ready else "NO_GO",
        "evidence_files": EVIDENCE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = assess_readiness(args.evidence_root.resolve())
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["benchmark_release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
