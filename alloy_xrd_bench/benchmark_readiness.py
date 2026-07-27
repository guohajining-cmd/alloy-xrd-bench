"""Combine the independent HEA-SRO and Gamma-Decomp release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"readiness evidence must be a JSON object: {path}")
    return value


def assess_benchmark_readiness(
    hea_readiness_path: str | Path,
    gamma_readiness_path: str | Path,
    output_path: str | Path,
    *,
    paper_path: str | Path | None = None,
    hea_paper_manifest_path: str | Path | None = None,
    gamma_paper_manifest_path: str | Path | None = None,
    release_smoke_path: str | Path | None = None,
    environment_lock_path: str | Path | None = None,
    external_sanity_path: str | Path | None = None,
    publication_record_path: str | Path | None = None,
) -> dict[str, Any]:
    hea_path = Path(hea_readiness_path).resolve()
    gamma_path = Path(gamma_readiness_path).resolve()
    hea = _read(hea_path)
    gamma = _read(gamma_path)
    hea_ready = bool(hea.get("core_benchmark_ready", False))
    gamma_ready = bool(gamma.get("gamma_benchmark_ready", False))
    failure_codes = []
    if not hea_ready:
        failure_codes.append("HEA_TRACK_NOT_READY")
    if not gamma_ready:
        failure_codes.append("GAMMA_TRACK_NOT_READY")
    submission_checks: dict[str, bool] = {}
    if paper_path is not None:
        paper_text = Path(paper_path).read_text(encoding="utf-8")
        submission_checks["paper_frozen"] = not any(
            marker in paper_text for marker in ("\\workingnote", "TODO", "Working Draft")
        )
    else:
        submission_checks["paper_frozen"] = False
    for name, path in (
        ("hea_paper_artifacts", hea_paper_manifest_path),
        ("gamma_paper_artifacts", gamma_paper_manifest_path),
    ):
        if path is None:
            submission_checks[name] = False
            continue
        manifest = _read(Path(path))
        submission_checks[name] = bool(
            manifest.get("artifacts")
            and not manifest.get("missing_scorecards", [])
            and manifest.get("complete_experiment_matrix", True)
        )
    if release_smoke_path is None:
        submission_checks["cpu_release_smoke"] = False
    else:
        submission_checks["cpu_release_smoke"] = _read(Path(release_smoke_path)).get("status") == "pass"
    submission_checks["environment_lock"] = bool(
        environment_lock_path is not None and Path(environment_lock_path).is_file()
        and Path(environment_lock_path).stat().st_size > 0
    )
    if external_sanity_path is None:
        submission_checks["external_sanity"] = False
    else:
        sanity = _read(Path(external_sanity_path))
        submission_checks["external_sanity"] = bool(
            sanity.get("status") == "pass"
            and sanity.get("exact_latent_truth_claimed") is False
        )
    if publication_record_path is None:
        submission_checks["publication_record"] = False
    else:
        publication = _read(Path(publication_record_path))
        submission_checks["publication_record"] = bool(
            publication.get("dataset_doi") and publication.get("code_tag")
        )
    submission_failure_codes = [
        f"{name.upper()}_NOT_READY"
        for name, passed in submission_checks.items()
        if not passed
    ]
    report = {
        "schema_version": "sitexrd-benchmark-readiness-v1.1",
        "release_scope": "dual_track",
        "benchmark_ready": hea_ready and gamma_ready,
        "failure_codes": failure_codes,
        "tracks": {
            "hea_sro": {
                "ready": hea_ready,
                "failure_codes": list(hea.get("failure_codes", [])),
                "diagnostic_family_status": hea.get("diagnostic_family_status"),
                "diagnostic_occupancy_status": hea.get("diagnostic_occupancy_status"),
                "evidence_path": str(hea_path),
            },
            "gamma_decomp": {
                "ready": gamma_ready,
                "failure_codes": list(gamma.get("failure_codes", [])),
                "evidence_path": str(gamma_path),
            },
        },
        "score_aggregation": "independent_track_metrics_no_composite_score",
        "submission_checks": submission_checks,
        "submission_failure_codes": submission_failure_codes,
        "kdd_submission_ready": bool(
            hea_ready and gamma_ready and all(submission_checks.values())
        ),
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hea-readiness", required=True)
    parser.add_argument("--gamma-readiness", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper")
    parser.add_argument("--hea-paper-manifest")
    parser.add_argument("--gamma-paper-manifest")
    parser.add_argument("--release-smoke")
    parser.add_argument("--environment-lock")
    parser.add_argument("--external-sanity")
    parser.add_argument("--publication-record")
    parser.add_argument("--require-submission", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = assess_benchmark_readiness(
            args.hea_readiness,
            args.gamma_readiness,
            args.output,
            paper_path=args.paper,
            hea_paper_manifest_path=args.hea_paper_manifest,
            gamma_paper_manifest_path=args.gamma_paper_manifest,
            release_smoke_path=args.release_smoke,
            environment_lock_path=args.environment_lock,
            external_sanity_path=args.external_sanity,
            publication_record_path=args.publication_record,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    ready = report["kdd_submission_ready"] if args.require_submission else report["benchmark_ready"]
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
