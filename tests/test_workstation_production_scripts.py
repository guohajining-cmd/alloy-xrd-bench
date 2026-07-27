import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


class WorkstationProductionScriptTests(unittest.TestCase):
    def test_environment_pins_python_311_and_xrd_dependencies(self):
        environment = yaml.safe_load(
            (ROOT / "environment-workstation.yml").read_text(encoding="utf-8")
        )
        dependencies = {str(value) for value in environment["dependencies"] if isinstance(value, str)}
        self.assertIn("python=3.11", dependencies)
        for package in ("pymatgen", "numpy", "pandas", "pyarrow", "h5py", "ase", "matplotlib"):
            self.assertIn(package, dependencies)

    def test_slurm_submission_uses_cpu_array_and_resume(self):
        submit = (ROOT / "scripts" / "workstation" / "submit_hea_xrd.sh").read_text(encoding="utf-8")
        worker = (ROOT / "scripts" / "workstation" / "hea_xrd_worker.sbatch").read_text(encoding="utf-8")
        self.assertIn("shard_count", submit)
        self.assertIn("%24", submit)
        self.assertIn("--cpus-per-task=1", worker)
        self.assertIn("--partition=hpc", worker)
        self.assertIn("--shard-index", worker)
        self.assertIn("--resume", worker)
        self.assertNotIn("--gres=gpu", worker)
        self.assertNotIn("guojingming", submit.lower() + worker.lower())

    def test_smoke_and_finalize_scripts_enforce_audit(self):
        smoke = (ROOT / "scripts" / "workstation" / "smoke_hea_xrd.sh").read_text(encoding="utf-8")
        finalize = (ROOT / "scripts" / "workstation" / "finalize_hea_xrd.sh").read_text(encoding="utf-8")
        self.assertIn("--max-structures 16", smoke)
        self.assertIn("simulate-finalize", smoke)
        self.assertIn("audit", smoke)
        self.assertIn("simulate-finalize", finalize)
        self.assertIn("audit", finalize)
        self.assertIn('cd "$DATASET_DIR/xrd/runs/$RUN_ID"', finalize)


if __name__ == "__main__":
    unittest.main()
