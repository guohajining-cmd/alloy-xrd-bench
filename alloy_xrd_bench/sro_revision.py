"""Explicit, source-bound corrections to released SRO labels and masks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_label_correction(old, new, tolerance=1e-10):
    """Allow only zero-count pairs for an already-defined central species."""
    if not all(math.isfinite(float(value)) for value in [*old.values(), *new.values()]):
        raise ValueError("Non-finite label")
    if any(key not in new or abs(float(value) - float(new[key])) > tolerance
           for key, value in old.items()):
        raise ValueError("Previously defined label removed or changed")
    centers = {key.split("->")[0] for key in old}
    added = set(new) - set(old)
    if any(key.split("->")[0] not in centers or abs(float(new[key]) - 1) > tolerance
           for key in added):
        raise ValueError("Correction is not a zero-count pair in a defined shell")
    return len(added)


def apply_label_revision(manifest, source_manifest_path, revision_directory, targets):
    directory = Path(revision_directory)
    meta = json.loads((directory / "revision.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != "sro-label-revision-v1" or meta.get("status") != "verified":
        raise ValueError("Unverified or unsupported label revision")
    if meta.get("source_manifest_sha256") != file_sha256(source_manifest_path):
        raise ValueError("Source manifest hash mismatch")
    if meta.get("labels_sha256") != file_sha256(directory / "labels.parquet"):
        raise ValueError("Corrected labels hash mismatch")
    if meta.get("sro_targets") != list(targets):
        raise ValueError("SRO target vocabulary or order mismatch")
    labels = pd.read_parquet(directory / "labels.parquet")
    key = "structure_state_id"
    identity = [key, "parent_group_id", "xrd_sample_id", "structure_sha256"]
    columns = ["sro_shell1_json", "sro_shell2_json"]
    if not set(identity + columns).issubset(labels.columns):
        raise ValueError("Missing revision columns")
    if labels[key].isna().any() or manifest[key].isna().any():
        raise ValueError("Missing structure identity")
    if labels[key].duplicated().any() or manifest[key].duplicated().any():
        raise ValueError("Duplicate structure identity")
    if set(labels[key]) != set(manifest[key]):
        raise ValueError("Revision does not cover exactly the source manifest")
    aligned = labels.set_index(key).loc[manifest[key]].reset_index()
    for column in identity:
        if not manifest[column].reset_index(drop=True).equals(aligned[column]):
            raise ValueError(f"Revision identity mismatch: {column}")
    updated = manifest.copy()
    for column in columns:
        for old, new in zip(manifest[column], aligned[column]):
            check_label_correction(json.loads(old), json.loads(new))
        updated[column] = aligned[column].to_numpy()
    return updated
