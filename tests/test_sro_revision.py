import hashlib
import json

import pandas as pd
import pytest

from alloy_xrd_bench.sro_revision import apply_label_revision, check_label_correction


def test_zero_pair_is_added_without_changing_old_values():
    old = {"Fe->Al": -1., "Al->Fe": -1.}
    new = {**old, "Fe->Fe": 1., "Al->Al": 1.}
    assert check_label_correction(old, new) == 2


@pytest.mark.parametrize("old,new", [
    ({"Fe->Al": -1.}, {"Fe->Al": -.9}),
    ({"Fe->Al": -1.}, {}),
    ({"Fe->Al": -1.}, {"Fe->Al": -1., "Fe->Fe": 0.}),
    ({}, {"Fe->Al": 1.}),
])
def test_unexplained_label_changes_are_rejected(old, new):
    with pytest.raises(ValueError):
        check_label_correction(old, new)


def test_full_recompute_worker_on_analytic_b2():
    from pymatgen.core import Structure, Lattice
    from scripts.build_sro_label_revision import recompute

    structure = Structure(Lattice.cubic(3.), ["Fe", "Al"], [[0, 0, 0], [.5, .5, .5]])
    row = dict(structure_state_id="s", parent_group_id="p", xrd_sample_id="x", structure_sha256="h",
               parent_lattice="bcc", sro_shell1_json='{"Fe->Al": -1, "Al->Fe": -1}',
               sro_shell2_json='{"Fe->Fe": -1, "Al->Al": -1}')
    result = recompute((row, json.dumps(structure.as_dict()), "json"))
    assert result["added_shell1"] == result["added_shell2"] == 2
    assert json.loads(result["sro_shell1_json"])["Fe->Fe"] == 1
    assert json.loads(result["sro_shell2_json"])["Fe->Al"] == 1


def artifact(tmp_path):
    source = tmp_path / "source.parquet"
    row = dict(structure_state_id="s", parent_group_id="p", xrd_sample_id="x", structure_sha256="h",
               id_split="test", sro_shell1_json='{"Fe->Al": -1}', sro_shell2_json="{}")
    manifest = pd.DataFrame([row])
    manifest.to_parquet(source, index=False)
    revision = tmp_path / "revision"
    revision.mkdir()
    labels = manifest.drop(columns="id_split").copy()
    labels.loc[0, "sro_shell1_json"] = '{"Fe->Al": -1, "Fe->Fe": 1}'
    labels.to_parquet(revision / "labels.parquet", index=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta = dict(schema_version="sro-label-revision-v1", status="verified", source_manifest_sha256=digest(source),
                labels_sha256=digest(revision / "labels.parquet"), sro_targets=["shell1:Fe->Fe"])
    (revision / "revision.json").write_text(json.dumps(meta))
    return manifest, source, revision, meta


def test_overlay_preserves_splits_and_original_frame(tmp_path):
    manifest, source, revision, _ = artifact(tmp_path)
    updated = apply_label_revision(manifest, source, revision, ["shell1:Fe->Fe"])
    assert updated.id_split.tolist() == ["test"]
    assert "Fe->Fe" in updated.sro_shell1_json[0]
    assert "Fe->Fe" not in manifest.sro_shell1_json[0]


def test_loader_revision_is_explicit_and_default_remains_unchanged(tmp_path):
    from alloy_xrd_bench.hea_validation_data import load_validation_rows

    manifest, source, revision, meta = artifact(tmp_path)
    manifest["structure_family"] = "bcc_like"
    manifest["grid_shard_path"] = "grids/a.h5"
    manifest["grid_row_index"] = 0
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    source = manifests / "task_samples.parquet"
    manifest.to_parquet(source, index=False)
    meta["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (revision / "revision.json").write_text(json.dumps(meta))
    pd.DataFrame([dict(structure_state_id="s", parent_group_id="p", fraction_Fe=.5)]).to_parquet(
        manifests / "composition_only_features.parquet", index=False)
    config = dict(elements=["Fe"], sro_targets=["shell1:Fe->Fe"],
                  family_mapping={"bcc_like": "bcc"}, split_columns=["id_split"])
    original = load_validation_rows(tmp_path, config)
    corrected = load_validation_rows(tmp_path, {**config, "sro_label_revision": str(revision)})
    assert not original[0].sro_mask[0]
    assert corrected[0].sro_mask[0]
    assert corrected[0].sro_values[0] == 1.
    assert original[0].splits == corrected[0].splits


@pytest.mark.parametrize("kind", ["hash", "identity", "missing", "duplicate", "targets", "status"])
def test_overlay_rejects_wrong_binding(tmp_path, kind):
    manifest, source, revision, meta = artifact(tmp_path)
    path = revision / "labels.parquet"
    labels = pd.read_parquet(path)
    if kind == "hash":
        meta["source_manifest_sha256"] = "wrong"
    elif kind == "identity":
        labels.loc[0, "parent_group_id"] = "wrong"
    elif kind == "missing":
        labels = labels.iloc[:0]
    elif kind == "duplicate":
        labels = pd.concat([labels, labels])
    elif kind == "targets":
        meta["sro_targets"] = []
    elif kind == "status":
        meta["status"] = "failed"
    labels.to_parquet(path, index=False)
    meta["labels_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (revision / "revision.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        apply_label_revision(manifest, source, revision, ["shell1:Fe->Fe"])
