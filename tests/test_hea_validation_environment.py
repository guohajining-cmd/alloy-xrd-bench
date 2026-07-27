from pathlib import Path
import tomllib
import unittest

import yaml


class ValidationEnvironmentTests(unittest.TestCase):
    def test_environment_has_training_and_io_dependencies(self):
        environment = yaml.safe_load(Path("environment-validation.yml").read_text())
        dependencies = environment["dependencies"]
        rendered = "\n".join(
            item if isinstance(item, str) else str(item) for item in dependencies
        )

        self.assertEqual("sitexrd-validate", environment["name"])
        for package in [
            "python=3.11",
            "pytorch",
            "pytorch-cuda=12.4",
            "scikit-learn",
            "xgboost",
            "pandas",
            "pyarrow",
            "h5py",
            "scipy",
            "matplotlib",
            "pymatgen",
        ]:
            self.assertIn(package, rendered)

    def test_project_exposes_hea_validation_command(self):
        project = tomllib.loads(Path("pyproject.toml").read_text())

        self.assertEqual(
            "alloy_xrd_bench.hea_validation:main",
            project["project"]["scripts"]["alloy-xrd-hea-validation"],
        )


if __name__ == "__main__":
    unittest.main()
