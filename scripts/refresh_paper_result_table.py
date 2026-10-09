"""Refresh table formatting without implying a rerun of historical experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from alloy_xrd_bench.dataset_paper_visuals import _write_compact_results
import alloy_xrd_bench.dataset_paper_visuals as visuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=Path("docs/paper/generated"))
    args = parser.parse_args()
    root = args.generated_dir
    hea = root / "hea_sro_results.csv"
    gamma = root / "gamma_decomp_results.csv"
    output = root / "compact_primary_results.tex"
    _write_compact_results(hea, gamma, output)
    paths = [hea, gamma, output, Path(visuals.__file__)]
    manifest = {
        "schema_version": "paper-table-format-v1",
        "scope": "Formatting only; historical v1.1 inputs, no experimental rerun.",
        "ranking": "Within split/view; best bold, second distinct displayed value underlined; ties share rank.",
        "significance_claim": False,
        "provenance_note": "Supersedes only the compact_results hash in dataset_visual_manifest.json; figures were not regenerated.",
        "files": [{"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    (root / "table_format_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
