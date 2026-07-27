from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class GammaOverlapCliTests(unittest.TestCase):
    def test_all_commands_run_on_tiny_physical_release(self):
        from alloy_xrd_bench.gamma_overlap import main

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            cfg = root / "tiny.cfg"
            cfg.write_text(_ideal_l12_cfg(), encoding="utf-8")
            config = Path("configs/nickel_gamma_overlap_v1.json").resolve()

            ingest_dir = root / "ingest"
            screen_dir = root / "screen"
            self.assertEqual(0, _run(main, "ingest", "--cfg", str(cfg), "--output", str(ingest_dir)))
            self.assertEqual(0, _run(main, "screen", "--cfg", str(cfg), "--output", str(screen_dir)))
            self.assertEqual(1, json.loads((ingest_dir / "ingest_summary.json").read_text())["accepted_core_count"])

            pilot = root / "pilot"
            build_args = (
                "--root", str(root), "--cfg", str(cfg), "--config", str(config),
                "--pair-count", "1", "--variants-per-pair", "1", "--gamma-count", "3",
            )
            self.assertEqual(0, _run(main, "pilot", "--output", str(pilot), *build_args))
            self.assertEqual(0, _run(main, "audit", "--output", str(pilot), "--config", str(config)))
            for name in (
                "parent_structures.parquet", "parent_pairs.parquet", "phase_patterns.parquet",
                "mixture_samples.parquet", "components.parquet", "overlap_windows.parquet",
                "splits.parquet", "provenance_edges.parquet",
            ):
                self.assertTrue((pilot / name).exists(), name)
            edges = pd.read_parquet(pilot / "provenance_edges.parquet")
            self.assertTrue(
                {"paired_as", "augmented_as", "decomposed_into", "cropped_as"}
                <= set(edges["relation_type"])
            )
            self.assertIn("pattern_sha256", pd.read_parquet(pilot / "mixture_samples.parquet").columns)
            window_manifest = pd.read_parquet(pilot / "overlap_windows.parquet")
            self.assertIn("window_sha256", window_manifest.columns)
            self.assertEqual(len(window_manifest), len(list((pilot / "windows").glob("*.npz"))))

            truth = pd.read_parquet(pilot / "components.parquet")
            predictions = truth.copy()
            predictions["component_id"] = "prediction_" + predictions["component_id"].astype(str)
            predictions["confidence"] = 1.0
            prediction_path = root / "predictions.parquet"
            predictions.to_parquet(prediction_path, index=False)
            self.assertEqual(
                0,
                _run(main, "score", "--truth", str(pilot / "components.parquet"), "--predictions", str(prediction_path)),
            )

            baseline_path = root / "baseline.parquet"
            self.assertEqual(
                0,
                _run(main, "baseline", "--release", str(pilot), "--output", str(baseline_path),
                     "--split", "train", "--max-windows", "1"),
            )
            self.assertTrue(baseline_path.exists())

            production = root / "production"
            self.assertEqual(0, _run(main, "generate", "--output", str(production), *build_args))
            self.assertEqual("production", json.loads((production / "summary.json").read_text())["release_status"])
            pilot_hash = pd.read_parquet(pilot / "mixture_samples.parquet").iloc[0]["pattern_sha256"]
            production_hash = pd.read_parquet(production / "mixture_samples.parquet").iloc[0]["pattern_sha256"]
            self.assertEqual(pilot_hash, production_hash)
            sample = pd.read_parquet(production / "mixture_samples.parquet").iloc[0]
            pattern_path = production / str(sample["pattern_path"])
            pattern_path.write_bytes(pattern_path.read_bytes() + b"tampered")
            self.assertEqual(1, _run(main, "audit", "--output", str(production), "--config", str(config)))


def _run(main, *args: str) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return main(list(args))


def _ideal_l12_cfg() -> str:
    from pymatgen.core import Lattice, Structure

    unit = Structure(
        Lattice.cubic(3.57),
        ["Al", "Ni", "Ni", "Ni"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    structure = unit * (3, 3, 3)
    lines = [
        "BEGIN_CFG", " Size", f"    {len(structure)}", " Supercell",
        "    10.71 0.0 0.0", "    0.0 10.71 0.0", "    0.0 0.0 10.71",
        " AtomData:  id type cartes_x cartes_y cartes_z fx fy fz",
    ]
    for index, (site, coords) in enumerate(zip(structure, structure.cart_coords), start=1):
        atom_type = 0 if str(site.specie) == "Al" else 1
        lines.append(
            f"    {index} {atom_type} {coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f} 0.0 0.0 0.0"
        )
    lines.extend([" Energy", "    -100.0", " PlusStress:  xx yy zz yz xz xy", "    0 0 0 0 0 0", "END_CFG", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
