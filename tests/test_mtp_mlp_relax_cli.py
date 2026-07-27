import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from alloy_xrd_bench.mlip import verify_potential
from alloy_xrd_bench.potentials import register_potential
from alloy_xrd_bench.relax import mtp_mlp_relax_initial_structures
from alloy_xrd_bench.structures import generate_initial_structures


class MtpMlpRelaxationTests(unittest.TestCase):
    def test_mtp_mlp_relaxation_writes_per_potential_manifests(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            fake_mlp = root / "fake_mlp.py"
            fake_mlp.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "def rewrite_cfg(src, dst):",
                        "    lines = Path(src).read_text(encoding='utf-8').splitlines()",
                        "    out = []",
                        "    in_atoms = False",
                        "    skip_next = False",
                        "    for line in lines:",
                        "        stripped = line.strip()",
                        "        if skip_next:",
                        "            if out[-1].strip() == 'Energy':",
                        "                out.append('       -1.250000000000')",
                        "            elif out[-1].startswith(' PlusStress') or out[-1].strip().startswith('PlusStress'):",
                        "                out.append('         0.00000     0.00000     0.00000     0.00000     0.00000     0.00000')",
                        "            skip_next = False",
                        "            continue",
                        "        if stripped.startswith('AtomData:'):",
                        "            in_atoms = True",
                        "            out.append(line)",
                        "            continue",
                        "        if in_atoms and stripped and not stripped.startswith(('Energy', 'PlusStress', 'Feature', 'END_CFG')):",
                        "            parts = line.split()",
                        "            if len(parts) >= 5:",
                        "                out.append(f'{int(parts[0]):14d} {int(parts[1]):4d} {float(parts[2]):14.8f} {float(parts[3]):13.8f} {float(parts[4]):13.8f}     0.010000    0.000000    0.000000')",
                        "                continue",
                        "        if stripped == 'Energy':",
                        "            in_atoms = False",
                        "            out.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        if stripped.startswith('PlusStress'):",
                        "            out.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        out.append(line)",
                        "    Path(dst).write_text('\\n'.join(out) + '\\n', encoding='utf-8')",
                        "",
                        "args = sys.argv[1:]",
                        "if args[0] == 'relax':",
                        "    rewrite_cfg(args[2], args[3])",
                        "    settings = next((a.split('=', 1)[1] for a in args if a.startswith('--relaxation_settings=')), '')",
                        "    if settings:",
                        "        log = None",
                        "        for line in Path(settings).read_text(encoding='utf-8').splitlines():",
                        "            if line.strip().startswith('relax:log'):",
                        "                log = line.split()[-1]",
                        "        if log:",
                        "            Path(log).write_text('Relaxation: starting relaxation cfg#1\\ncfg 1\\titr 0\\tfrc 0.010000\\nRelaxation of cfg#1 completed successfully\\n', encoding='utf-8')",
                        "elif args[0] == 'calculate_efs':",
                        "    output = next(a.split('=', 1)[1] for a in args if a.startswith('--output_filename='))",
                        "    rewrite_cfg(args[2], output)",
                        "else:",
                        "    raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )

            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "CrCoNi",
                                "phase": "fcc",
                                "composition": {"Cr": 1, "Co": 1, "Ni": 2},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.58,
                                "seed": 7,
                                "generation_method": "pair_balance_occupancy",
                            },
                            {
                                "alloy_system": "Al",
                                "phase": "fcc",
                                "composition": {"Al": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 4.05,
                                "seed": 8,
                                "generation_method": "deterministic_random_occupancy",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            generate_initial_structures(root=root, recipes_path=recipe_path)
            model_file = root / "model.mtp"
            model_file.write_text("MTP\nversion = 1.1.0\nspecies_count = 3\n", encoding="utf-8")
            register_potential(
                root=root,
                potential_id="pot_fake_mtp",
                family="MTP",
                engine="mtp_mlp",
                elements=["Cr", "Co", "Ni"],
                phases=["fcc"],
                source_url="https://example.org/fake-mtp",
                doi="10.0000/fake-mtp",
                license_name="MIT",
                redistribution_policy="redistributable_with_attribution",
                citation_keys=["fake_mtp"],
                file_path=model_file,
            )
            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["type_map"] = {"0": "Cr", "1": "Co", "2": "Ni"}
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            rows = mtp_mlp_relax_initial_structures(
                root=root,
                potential_id="pot_fake_mtp",
                mlp_command=[sys.executable, str(fake_mlp)],
                fmax=0.05,
                steps=1000,
                relax_cell=False,
            )

            manifest_dir = root / "data" / "potentials" / "pot_fake_mtp" / "generated" / "manifests"
            relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
            relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")
            edges = pd.read_parquet(manifest_dir / "provenance_edges.parquet")
            relaxed_json_path = root / relaxed.iloc[0]["artifact_path_json"]
            relaxed_cif_path = root / relaxed.iloc[0]["artifact_path_cif"]
            relaxed_json_exists = relaxed_json_path.exists()
            relaxed_cif_exists = relaxed_cif_path.exists()

        self.assertEqual(1, len(rows))
        self.assertEqual(1, len(relaxations))
        self.assertEqual("mtp_mlp", relaxations.iloc[0]["engine"])
        self.assertLessEqual(float(relaxations.iloc[0]["fmax"]), 0.05)
        self.assertEqual("pass", relaxed.iloc[0]["quality_tier"])
        self.assertTrue(relaxed_json_exists)
        self.assertTrue(relaxed_cif_exists)
        self.assertIn("used_by", set(edges["relation_type"]))

    def test_verify_mtp_mlp_uses_external_command_without_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            fake_mlp = root / "fake_mlp.py"
            fake_mlp.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "def rewrite_cfg(src, dst):",
                        "    text = Path(src).read_text(encoding='utf-8')",
                        "    lines = []",
                        "    skip_next = False",
                        "    in_atoms = False",
                        "    for line in text.splitlines():",
                        "        stripped = line.strip()",
                        "        if skip_next:",
                        "            lines.append('       -2.500000000000' if lines[-1].strip() == 'Energy' else '         0.00000     0.00000     0.00000     0.00000     0.00000     0.00000')",
                        "            skip_next = False",
                        "            continue",
                        "        if stripped.startswith('AtomData:'):",
                        "            in_atoms = True",
                        "            lines.append(line)",
                        "            continue",
                        "        if in_atoms and stripped and not stripped.startswith(('Energy', 'PlusStress', 'Feature', 'END_CFG')):",
                        "            parts = line.split()",
                        "            if len(parts) >= 5:",
                        "                lines.append(f'{int(parts[0]):14d} {int(parts[1]):4d} {float(parts[2]):14.8f} {float(parts[3]):13.8f} {float(parts[4]):13.8f}     0.010000    0.000000    0.000000')",
                        "                continue",
                        "        if stripped == 'Energy':",
                        "            in_atoms = False",
                        "            lines.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        if stripped.startswith('PlusStress'):",
                        "            lines.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        lines.append(line)",
                        "    Path(dst).write_text('\\n'.join(lines) + '\\n', encoding='utf-8')",
                        "",
                        "args = sys.argv[1:]",
                        "if args[0] == 'calculate_efs':",
                        "    output = next(a.split('=', 1)[1] for a in args if a.startswith('--output_filename='))",
                        "    rewrite_cfg(args[2], output)",
                        "elif args[0] == 'relax':",
                        "    rewrite_cfg(args[2], args[3])",
                        "else:",
                        "    raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "CrCoNi",
                                "phase": "fcc",
                                "composition": {"Cr": 1, "Co": 1, "Ni": 2},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.58,
                                "seed": 7,
                                "generation_method": "pair_balance_occupancy",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            generate_initial_structures(root=root, recipes_path=recipe_path)
            initial_id = pd.read_parquet(root / "data" / "shared" / "manifests" / "initial_structures.parquet").iloc[0][
                "initial_structure_id"
            ]
            model_file = root / "model.mtp"
            model_file.write_text("MTP\nversion = 1.1.0\nspecies_count = 3\n", encoding="utf-8")
            register_potential(
                root=root,
                potential_id="pot_fake_mtp",
                family="MTP",
                engine="mtp_mlp",
                elements=["Cr", "Co", "Ni"],
                phases=["fcc"],
                source_url="https://example.org/fake-mtp",
                doi="10.0000/fake-mtp",
                license_name="MIT",
                redistribution_policy="redistributable_with_attribution",
                citation_keys=["fake_mtp"],
                file_path=model_file,
            )
            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["type_map"] = {"0": "Cr", "1": "Co", "2": "Ni"}
            catalog["potentials"][0]["mlp_command"] = [sys.executable, str(fake_mlp)]
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            report = verify_potential(
                root=root,
                potential_id="pot_fake_mtp",
                reference_initial_id=str(initial_id),
                smoke_initial_ids=[str(initial_id)],
            )
            manifest_path = (
                root
                / "data"
                / "potentials"
                / "pot_fake_mtp"
                / "generated"
                / "manifests"
                / "relaxations.parquet"
            )

        self.assertEqual("pass", report["status"])
        self.assertTrue(report["load"]["ok"])
        self.assertEqual("pass", report["reference_inference"]["status"])
        self.assertEqual("pass", report["smoke_relaxations"][0]["status"])
        self.assertFalse(manifest_path.exists())

    def test_mtp_mlp_respects_declared_phase_scope(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            fake_mlp = root / "fake_mlp.py"
            fake_mlp.write_text(
                "\n".join(
                    [
                        "import sys",
                        "from pathlib import Path",
                        "",
                        "def rewrite_cfg(src, dst):",
                        "    text = Path(src).read_text(encoding='utf-8')",
                        "    out = []",
                        "    skip_next = False",
                        "    in_atoms = False",
                        "    for line in text.splitlines():",
                        "        stripped = line.strip()",
                        "        if skip_next:",
                        "            out.append('       -3.000000000000' if out[-1].strip() == 'Energy' else '         0.00000     0.00000     0.00000     0.00000     0.00000     0.00000')",
                        "            skip_next = False",
                        "            continue",
                        "        if stripped.startswith('AtomData:'):",
                        "            in_atoms = True",
                        "            out.append(line)",
                        "            continue",
                        "        if in_atoms and stripped and not stripped.startswith(('Energy', 'PlusStress', 'Feature', 'END_CFG')):",
                        "            parts = line.split()",
                        "            if len(parts) >= 5:",
                        "                out.append(f'{int(parts[0]):14d} {int(parts[1]):4d} {float(parts[2]):14.8f} {float(parts[3]):13.8f} {float(parts[4]):13.8f}     0.010000    0.000000    0.000000')",
                        "                continue",
                        "        if stripped == 'Energy':",
                        "            in_atoms = False",
                        "            out.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        if stripped.startswith('PlusStress'):",
                        "            out.append(line)",
                        "            skip_next = True",
                        "            continue",
                        "        out.append(line)",
                        "    Path(dst).write_text('\\n'.join(out) + '\\n', encoding='utf-8')",
                        "",
                        "args = sys.argv[1:]",
                        "if args[0] == 'calculate_efs':",
                        "    output = next(a.split('=', 1)[1] for a in args if a.startswith('--output_filename='))",
                        "    rewrite_cfg(args[2], output)",
                        "elif args[0] == 'relax':",
                        "    rewrite_cfg(args[2], args[3])",
                        "else:",
                        "    raise SystemExit(2)",
                    ]
                ),
                encoding="utf-8",
            )
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "Ni-Al",
                                "phase": "fcc",
                                "composition": {"Ni": 9, "Al": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.58,
                                "seed": 23,
                                "generation_method": "pair_balance_occupancy",
                            },
                            {
                                "alloy_system": "Ni-Al",
                                "phase": "l12",
                                "composition": {"Ni": 3, "Al": 1},
                                "sublattice_composition": {
                                    "corner": {"Al": 1},
                                    "face": {"Ni": 3},
                                },
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.58,
                                "seed": 24,
                                "generation_method": "l12_sublattice_occupancy",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            generate_initial_structures(root=root, recipes_path=recipe_path)
            initial_rows = pd.read_parquet(root / "data" / "shared" / "manifests" / "initial_structures.parquet")
            fcc_id = str(initial_rows[initial_rows["phase"] == "fcc"].iloc[0]["initial_structure_id"])
            model_file = root / "model.mtp"
            model_file.write_text("MTP\nversion = 1.1.0\nspecies_count = 2\n", encoding="utf-8")
            register_potential(
                root=root,
                potential_id="pot_fake_ni3al_mtp",
                family="MTP",
                engine="mtp_mlp",
                elements=["Al", "Ni"],
                phases=["l12"],
                source_url="https://example.org/fake-ni3al-mtp",
                doi="10.0000/fake-ni3al-mtp",
                license_name="CC-BY-4.0",
                redistribution_policy="redistributable_with_attribution",
                citation_keys=["fake_ni3al_mtp"],
                file_path=model_file,
            )
            catalog_path = root / "data" / "registry" / "potentials_catalog.yml"
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
            catalog["potentials"][0]["type_map"] = {"0": "Al", "1": "Ni"}
            catalog["potentials"][0]["mlp_command"] = [sys.executable, str(fake_mlp)]
            catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

            rows = mtp_mlp_relax_initial_structures(
                root=root,
                potential_id="pot_fake_ni3al_mtp",
                mlp_command=[sys.executable, str(fake_mlp)],
                fmax=0.05,
                steps=1000,
                relax_cell=False,
            )
            report = verify_potential(
                root=root,
                potential_id="pot_fake_ni3al_mtp",
                reference_initial_id=fcc_id,
                smoke_initial_ids=[fcc_id],
            )
            manifest_dir = root / "data" / "potentials" / "pot_fake_ni3al_mtp" / "generated" / "manifests"
            relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")

        self.assertEqual(1, len(rows))
        self.assertEqual(["l12"], relaxations["initial_structure_id"].str.extract(r"_([a-z0-9]+)_", expand=False).tolist())
        self.assertEqual("skipped_unsupported_phase", report["reference_inference"]["status"])
        self.assertEqual("skipped_unsupported_phase", report["smoke_relaxations"][0]["status"])


if __name__ == "__main__":
    unittest.main()
