from pathlib import Path

import pandas as pd
import pytest

from alloy_xrd_bench.hea_perturbation import validate_heldout_scope, aggregate_perturbation_reports


def test_unbound_clean_aggregate_is_rejected():
    frame = pd.DataFrame(dict(parent_group_id=['p'], id_split=['test']))
    with pytest.raises(ValueError, match='clean-prediction'):
        validate_heldout_scope(frame, frame, 'id_split', Path('full-test.metrics.json'))


def test_missing_or_non_test_assignments_cannot_be_marked_pass():
    tasks = pd.DataFrame(dict(parent_group_id=['p', 'q'], id_split=['train', 'test']))
    with pytest.raises(ValueError, match='test'):
        validate_heldout_scope(tasks, tasks, 'id_split', None)
    with pytest.raises(ValueError, match='split'):
        validate_heldout_scope(tasks.drop(columns='id_split'), tasks, 'id_split', None)
    validate_heldout_scope(tasks, tasks.iloc[[1]], 'id_split', None)


def test_cross_split_parent_is_rejected():
    tasks = pd.DataFrame(dict(parent_group_id=['p', 'p'], id_split=['train', 'test']))
    with pytest.raises(ValueError, match='Parent'):
        validate_heldout_scope(tasks, tasks.iloc[[1]], 'id_split', None)


def test_legacy_pass_summary_cannot_be_reused_as_verified_evidence(tmp_path):
    import json
    (tmp_path/'perturbation_summary.json').write_text(json.dumps(dict(status='pass', seed=2027,
        perturbed_parent_macro_mae=.4, clean_to_perturbed_retention=.9)))
    report = aggregate_perturbation_reports(tmp_path)
    assert report['status'] == 'fail'
    assert report['reason'] == 'unverified_matched_evaluation_scope'
