import json

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.occupancy_pipeline import _sro, _warren_cowley


def b2():
    return Structure(Lattice.cubic(3), ["Fe", "Al"], [[0, 0, 0], [.5, .5, .5]])


def test_b2_includes_zero_count_pairs_in_both_shells():
    first, second = _sro(b2(), "bcc")
    assert first == pytest.approx({"Fe->Fe": 1, "Fe->Al": -1, "Al->Fe": -1, "Al->Al": 1})
    assert second == pytest.approx({"Fe->Fe": -1, "Fe->Al": 1, "Al->Fe": 1, "Al->Al": -1})


def test_l12_analytical_values_are_not_clipped():
    structure = Structure(Lattice.cubic(4), ["Au", "Cu", "Cu", "Cu"],
                          [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    first, second = _sro(structure, "fcc")
    assert first == pytest.approx({"Au->Au": 1, "Au->Cu": -1 / 3, "Cu->Au": -1 / 3, "Cu->Cu": 1 / 9})
    assert second == pytest.approx({"Au->Au": -3, "Au->Cu": 1, "Cu->Au": 1, "Cu->Cu": -1 / 3})


def test_empty_shell_remains_undefined():
    assert _warren_cowley(b2(), 0, .1) == {}


def test_parent_and_single_species_gates_are_preserved():
    assert _sro(b2(), "unknown") == ({}, {})
    pure = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
    assert _sro(pure, "bcc") == ({}, {})


def test_invariance_to_order_translation_and_cell_replication():
    base = b2()
    expected = _sro(base, "bcc")
    reordered = Structure(base.lattice, list(reversed(base.species)), base.frac_coords[::-1])
    translated = base.copy()
    translated.translate_sites(range(len(base)), [.21, .32, .43])
    for structure in (reordered, translated, base * [2, 2, 2]):
        actual = _sro(structure, "bcc")
        for left, right in zip(actual, expected):
            assert left == pytest.approx(right)


def test_zero_count_label_is_valid_but_absent_species_is_masked():
    first, second = _sro(b2(), "bcc")
    targets = ["shell1:Fe->Fe", "shell1:Fe->Al", "shell2:Fe->Al", "shell1:Fe->Ni"]
    values, mask = vectorize_sro(json.dumps(first), json.dumps(second), targets)
    np.testing.assert_array_equal(mask, [True, True, True, False])
    np.testing.assert_allclose(values[:3], [1, -1, 1])
