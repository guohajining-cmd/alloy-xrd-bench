import numpy as np
import pytest
from scripts.assess_sro_fixed_cell_predictions import pair_scores


def test_only_change_is_scored_and_masked_entries_are_ignored():
    p, c = np.array([[.3, 0.]]), np.array([[.9, 5.]])
    mask = np.array([[True, False]])
    no_change = pair_scores(p, c, mask, mask, p, p)
    np.testing.assert_allclose(no_change.change_error, [.6])
    np.testing.assert_allclose(no_change.change_error, no_change.zero_change_error)
    perfect = pair_scores(p, c, mask, mask, p+2, c+2)
    np.testing.assert_allclose(perfect.change_error, [0.], atol=1e-15)
    assert perfect.changed_target_count.iloc[0] == 1


def test_pair_masks_cannot_change():
    x, mask = np.ones((1,2)), np.ones((1,2), bool)
    with pytest.raises(ValueError, match="mask"):
        pair_scores(x,x,mask,np.array([[True,False]]),x,x)
