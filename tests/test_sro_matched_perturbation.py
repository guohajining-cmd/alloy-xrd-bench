import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_sro_corrected_perturbations import matched_test_observations, paired_errors
from scripts.verify_sro_perturbation_predictions import aggregate_errors


def frames():
    tasks = pd.DataFrame(dict(structure_state_id=['s0', 's1'], parent_group_id=['p0', 'p1'],
        xrd_sample_id=['c0', 'c1'], grid_shard_path=['g', 'g'], grid_row_index=[0, 1],
        id_split=['train', 'test'], sro_shell1_json=['{}', '{}'], sro_shell2_json=['{}', '{}']))
    obs = pd.DataFrame(dict(structure_state_id=['s0', 's1'], parent_group_id=['p0', 'p1'],
        xrd_sample_id=['o0', 'o1'], source_xrd_sample_id=['c0', 'c1'], id_split=['train', 'test'],
        grid_shard_path=['o', 'o'], grid_row_index=[0, 1]))
    return tasks, obs


def test_only_matching_test_states_are_selected():
    tasks, obs = frames()
    selected = matched_test_observations(obs, tasks, 'id_split')
    assert selected.xrd_sample_id.tolist() == ['o1']
    assert selected.clean_xrd_sample_id.tolist() == ['c1']


@pytest.mark.parametrize('column,value', [('parent_group_id', 'wrong'),
    ('source_xrd_sample_id', 'wrong'), ('id_split', 'train')])
def test_observation_identity_or_split_mismatch_rejected(column, value):
    tasks, obs = frames()
    obs.loc[1, column] = value
    with pytest.raises(ValueError):
        matched_test_observations(obs, tasks, 'id_split')


def test_parent_mean_does_not_overweight_repeated_views_or_states():
    frame = pd.DataFrame(dict(parent_group_id=['p', 'p', 'p', 'q'],
                             structure_state_id=['a', 'a', 'b', 'c']))
    truth = np.zeros((4, 1))
    pert = np.array([[2.], [4.], [1.], [6.]])
    table = paired_errors(truth, np.ones_like(truth, dtype=bool), truth, pert, frame)
    assert table.loc['p', 'perturbed'] == 2.
    assert table.loc['q', 'perturbed'] == 6.
    assert table.clean.eq(0).all()
    _, independent = aggregate_errors(truth, pert, np.ones_like(truth, dtype=bool), frame)
    np.testing.assert_array_equal(independent.to_numpy(), table.perturbed.to_numpy())


def test_local_rescoring_rejects_nonfinite_predictions():
    frame = pd.DataFrame(dict(parent_group_id=['p'], structure_state_id=['s']))
    with pytest.raises(ValueError):
        aggregate_errors(np.zeros((1, 1)), np.array([[np.inf]]), np.ones((1, 1), bool), frame)
