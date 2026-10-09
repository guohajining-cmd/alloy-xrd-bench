import numpy as np
import pandas as pd
import pytest
from scripts.verify_sro_perturbed_increment import observation_parent_errors


def test_unequal_observation_counts_do_not_reweight_states_or_parents():
    frame=pd.DataFrame(dict(parent_group_id=["p","p","p","q"],structure_state_id=["a","a","b","c"]))
    truth=np.zeros((4,1))
    values=np.array([[1.],[3.],[6.],[8.]])
    actual=observation_parent_errors(truth,np.ones_like(truth,bool),values,frame)
    assert actual.to_dict()=={"p":4.,"q":8.}


def test_zero_mask_is_not_an_error_free_prediction():
    frame=pd.DataFrame(dict(parent_group_id=["p"],structure_state_id=["a"]))
    x=np.zeros((1,1))
    with pytest.raises(ValueError):
        observation_parent_errors(x,np.zeros_like(x,bool),x,frame)
