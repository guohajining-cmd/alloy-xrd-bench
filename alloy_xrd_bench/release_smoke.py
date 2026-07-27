"""Run a CPU-only end-to-end smoke check for both public benchmark tracks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence

from .evaluate import score_submission
from .gamma_overlap_baseline import run_public_no_spectrum_prior
from .gamma_overlap_public import validate_public_release
from .hea_validation import train_composition_ridge


def run_release_smoke(
    hea_dataset: str | Path,
    hea_config: str | Path,
    gamma_release: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    hea = Path(hea_dataset).resolve()
    config = Path(hea_config).resolve()
    gamma = Path(gamma_release).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    gamma_audit = validate_public_release(gamma)
    if gamma_audit.get("status") != "pass":
        raise ValueError("Gamma-Decomp release audit failed")

    hea_model = train_composition_ridge(
        hea, config, output / "hea_model", seed=2027, split_column="id_split"
    )
    hea_submission = Path(str(hea_model["submission_path"]))
    hea_score = score_submission(
        hea,
        hea_submission,
        output / "hea_score",
        config,
        bootstrap_replicates=32,
        bootstrap_seed=2027,
    )
    gamma_score = run_public_no_spectrum_prior(
        release_dir=gamma,
        output_dir=output / "gamma_no_spectrum",
        task="window",
        split="validation",
    )
    report = {
        "schema_version": "sitexrd-release-smoke-v1.1",
        "status": "pass",
        "qualification": "engineering-smoke",
        "device_requirement": "cpu",
        "python": platform.python_version(),
        "hea": {
            "model": "composition-ridge",
            "split": "id_split:test",
            "submission_path": str(hea_submission),
            "score_status": hea_score["status"],
        },
        "gamma_decomp": {
            "model": "no-spectrum",
            "task": "window",
            "split": "validation",
            "release_audit": gamma_audit["status"],
            "score_status": gamma_score["status"],
        },
    }
    (output / "release_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hea-dataset", required=True)
    parser.add_argument("--hea-config", required=True)
    parser.add_argument("--gamma-release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_release_smoke(
            args.hea_dataset, args.hea_config, args.gamma_release, args.output
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
