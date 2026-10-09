import json

import pandas as pd
import pytest

from scripts.build_corrected_sro_manifest import complete_shell, corrected_labels


def test_complete_shell_fills_only_defined_zero_pairs():
    fixed, added, undefined = complete_shell('{"Fe->Al": -1.0}', ["Fe", "Al"])
    assert json.loads(fixed) == {"Fe->Al": -1.0, "Fe->Fe": 1.0}
    assert (added, undefined) == (1, 2)


def test_complete_shell_rejects_foreign_elements():
    with pytest.raises(ValueError, match="invalid SRO pair"):
        complete_shell('{"Fe->Cu": 0.0}', ["Fe", "Al"])


def test_corrected_sidecar_preserves_rows_and_parent_mapping():
    tasks = pd.DataFrame([
        {"structure_state_id": "source", "parent_structure_id": "source", "parent_group_id": "p",
         "fidelity": "source_relaxed_dft", "parent_lattice": "bcc", "sro_shell1_json": '{"Fe->Al": -1.0}',
         "sro_shell2_json": "{}"},
        {"structure_state_id": "counterfactual", "parent_structure_id": "source", "parent_group_id": "p",
         "fidelity": "fixed_lattice_counterfactual", "parent_lattice": "bcc",
         "sro_shell1_json": '{"Al->Fe": -1.0}', "sro_shell2_json": "{}"},
    ])
    result = corrected_labels(tasks, {"source": ["Fe", "Al"]})
    assert result.structure_state_id.tolist() == ["source", "counterfactual"]
    assert result.added_shell1_keys.tolist() == [1, 1]
    assert result.undefined_shell2_keys.tolist() == [4, 4]
    assert json.loads(result.loc[1, "corrected_shell1_json"])["Al->Al"] == 1.0
