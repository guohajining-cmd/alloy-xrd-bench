import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from scripts.audit_sro_independent import ase_sro, implied_zero_pairs


def test_independent_b2_periodic_images_and_empty_shell():
    structure = Structure(Lattice.cubic(3), ["Fe", "Al"], [[0, 0, 0], [.5, .5, .5]])
    values, stats = ase_sro(structure, "bcc")
    assert values[0] == pytest.approx({"Fe->Fe": 1, "Fe->Al": -1, "Al->Fe": -1, "Al->Al": 1})
    assert values[1] == pytest.approx({"Fe->Fe": -1, "Fe->Al": 1, "Al->Fe": 1, "Al->Al": -1})
    assert stats[0]["coordination_min"] == stats[0]["coordination_max"] == 8
    assert stats[1]["coordination_min"] == stats[1]["coordination_max"] == 6


def test_independent_reference_preserves_undefined_centers():
    structure = Structure(Lattice.cubic(20), ["Fe", "Fe", "Al"], [[0, 0, 0], [.04, 0, 0], [.5, .5, .5]])
    values, _ = ase_sro(structure, "bcc")
    assert "Fe->Al" in values[0]
    assert values[0]["Fe->Al"] == 1
    assert "Al->Fe" not in values[0]


def test_implied_zero_pairs_does_not_fill_undefined_center():
    assert implied_zero_pairs({"Fe->Fe": -.5}, ["Fe", "Al"]) == ["Fe->Al"]
    assert implied_zero_pairs({}, ["Fe", "Al"]) == []


def test_reference_sum_rule_and_supercell_invariance():
    structure = Structure(Lattice.cubic(4), ["Au", "Cu", "Cu", "Cu"],
                          [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    values, _ = ase_sro(structure, "fcc")
    repeated, _ = ase_sro(structure * [2, 2, 2], "fcc")
    for shell, other in zip(values, repeated):
        assert shell == pytest.approx(other)
        for center in ["Au", "Cu"]:
            assert np.isclose(.25 * shell[f"{center}->Au"] + .75 * shell[f"{center}->Cu"], 0)
