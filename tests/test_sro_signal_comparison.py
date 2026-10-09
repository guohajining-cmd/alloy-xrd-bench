import pandas as pd
import pytest

from scripts.run_sro_signal_comparison import select_sample


def test_frozen_selection_is_parent_disjoint_and_deterministic():
    frame = pd.DataFrame([
        {"structure_state_id": f"s{i}", "parent_group_id": f"p{i}", "fidelity": "source_relaxed_dft",
         "parent_lattice": "fcc", "nelements": 2,
         "id_split": ("train", "validation", "test")[i // 2]}
        for i in range(6)
    ])
    counts = {"train": 1, "validation": 1, "test": 1}
    first = select_sample(frame, "id_split", 5, counts)
    second = select_sample(frame.sample(frac=1, random_state=7), "id_split", 5, counts)
    assert first.structure_state_id.tolist() == second.structure_state_id.tolist()
    assert first.parent_group_id.is_unique
    assert first.id_split.tolist() == ["train", "validation", "test"]


def test_selection_rejects_parent_leakage():
    frame = pd.DataFrame([
        {"structure_state_id": "a", "parent_group_id": "p", "fidelity": "source_relaxed_dft",
         "parent_lattice": "fcc", "nelements": 2, "id_split": "train"},
        {"structure_state_id": "b", "parent_group_id": "p", "fidelity": "source_relaxed_dft",
         "parent_lattice": "fcc", "nelements": 2, "id_split": "test"},
    ])
    with pytest.raises(ValueError, match="crosses split"):
        select_sample(frame, "id_split", 5, {"train": 1, "validation": 1, "test": 1})
