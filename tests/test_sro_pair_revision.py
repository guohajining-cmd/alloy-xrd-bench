import pytest
from scripts.audit_sro_pair_revision import pair_change


def test_pair_change_uses_only_defined_targets():
    value = pair_change(({"Fe->Al": -1.}, {}), ({"Fe->Al": 1.}, {}))
    assert value["common_targets"] == 1
    assert value["changed_targets"] == 1
    assert value["mean_abs_change"] == 2.


def test_missing_shell_is_not_zero_sro_change():
    value = pair_change(({}, {}), ({}, {}))
    assert value["common_targets"] == 0
    assert value["mean_abs_change"] is None


def test_mask_change_is_reported_separately():
    value = pair_change(({"Fe->Al": 1.}, {}), ({"Fe->Fe": 1.}, {}))
    assert value["mask_symmetric_difference"] == 2
    assert value["max_abs_change"] is None


def test_nonfinite_pair_fails():
    with pytest.raises(ValueError):
        pair_change(({"Fe->Al": float("nan")}, {}), ({"Fe->Al": 1.}, {}))
