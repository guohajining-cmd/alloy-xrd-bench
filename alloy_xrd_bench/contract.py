"""Dataset contract and validation helpers for Alloy-XRD-Bench."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FieldSpec:
    """A manifest field in the public benchmark contract."""

    name: str
    dtype: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ManifestSchema:
    """Ordered schema definition for one public manifest."""

    name: str
    id_field: str
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """A structured validation issue."""

    code: str
    message: str
    manifest: str | None = None
    record_index: int | None = None
    record_id: str | None = None


DEFAULT_XRD_CONFIG: dict[str, Any] = {
    "xrd_config_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
    "radiation": "Cu Kalpha",
    "two_theta_min": 20.0,
    "two_theta_max": 100.0,
    "step": 0.02,
    "broadening": {"kind": "gaussian", "fwhm_deg": 0.1},
    "intensity_normalization": {"method": "max", "max_intensity": 100.0},
}


DEFAULT_RELAX_CONFIG: dict[str, Any] = {
    "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
    "fmax_eV_per_A": 0.05,
    "max_steps": 1000,
    "abnormal_volume_change_threshold": 0.20,
}


MANIFEST_SCHEMAS: dict[str, ManifestSchema] = {
    "potentials_catalog": ManifestSchema(
        name="potentials_catalog",
        id_field="potential_id",
        fields=(
            FieldSpec("potential_id", "string"),
            FieldSpec("family", "string"),
            FieldSpec("engine", "string"),
            FieldSpec("elements", "list[string]"),
            FieldSpec("phases", "list[string]"),
            FieldSpec("source_url", "string"),
            FieldSpec("doi", "string"),
            FieldSpec("license", "string"),
            FieldSpec("file_sha256", "sha256"),
            FieldSpec("redistribution_policy", "string"),
            FieldSpec("citation_keys", "list[string]"),
            FieldSpec("potential_dir", "path"),
            FieldSpec("generated_data_dir", "path"),
        ),
    ),
    "initial_structures": ManifestSchema(
        name="initial_structures",
        id_field="initial_structure_id",
        fields=(
            FieldSpec("initial_structure_id", "string"),
            FieldSpec("alloy_system", "string"),
            FieldSpec("phase", "string"),
            FieldSpec("composition", "object"),
            FieldSpec("supercell", "string"),
            FieldSpec("generation_method", "string"),
            FieldSpec("generation_score", "number", required=False),
            FieldSpec("candidate_count", "integer", required=False),
            FieldSpec("seed", "integer"),
            FieldSpec("structure_family", "string", required=False),
            FieldSpec("phase_role", "string", required=False),
            FieldSpec("pair_id", "string", required=False),
            FieldSpec("nominal_alloy_system", "string", required=False),
            FieldSpec("partitioning_rule", "string", required=False),
            FieldSpec("structure_sha256", "sha256"),
            FieldSpec("artifact_file_sha256", "sha256", required=False),
            FieldSpec("artifact_path", "path"),
        ),
    ),
    "gamma_prime_pairs": ManifestSchema(
        name="gamma_prime_pairs",
        id_field="pair_id",
        fields=(
            FieldSpec("pair_id", "string"),
            FieldSpec("gamma_initial_structure_id", "string"),
            FieldSpec("gamma_prime_initial_structure_id", "string"),
            FieldSpec("gamma_relaxed_structure_id", "string", required=False),
            FieldSpec("gamma_prime_relaxed_structure_id", "string", required=False),
            FieldSpec("nominal_alloy_system", "string"),
            FieldSpec("partitioning_rule", "string"),
            FieldSpec("misfit_definition", "string"),
        ),
    ),
    "relaxations": ManifestSchema(
        name="relaxations",
        id_field="relaxation_run_id",
        fields=(
            FieldSpec("relaxation_run_id", "string"),
            FieldSpec("initial_structure_id", "string"),
            FieldSpec("potential_id", "string"),
            FieldSpec("engine", "string"),
            FieldSpec("relax_config_id", "string"),
            FieldSpec("converged", "boolean"),
            FieldSpec("fmax", "number"),
            FieldSpec("stress", "object"),
            FieldSpec("volume_change", "number"),
            FieldSpec("steps", "integer"),
            FieldSpec("log_path", "path"),
        ),
    ),
    "relaxed_structures": ManifestSchema(
        name="relaxed_structures",
        id_field="relaxed_structure_id",
        fields=(
            FieldSpec("relaxed_structure_id", "string"),
            FieldSpec("relaxation_run_id", "string"),
            FieldSpec("structure_sha256", "sha256"),
            FieldSpec("artifact_json_file_sha256", "sha256", required=False),
            FieldSpec("artifact_cif_file_sha256", "sha256", required=False),
            FieldSpec("artifact_path_json", "path"),
            FieldSpec("artifact_path_cif", "path"),
            FieldSpec("quality_tier", "string"),
        ),
    ),
    "xrd_samples": ManifestSchema(
        name="xrd_samples",
        id_field="xrd_sample_id",
        fields=(
            FieldSpec("xrd_sample_id", "string"),
            FieldSpec("relaxed_structure_id", "string"),
            FieldSpec("xrd_config_id", "string"),
            FieldSpec("two_theta_min", "number"),
            FieldSpec("two_theta_max", "number"),
            FieldSpec("step", "number"),
            FieldSpec("wavelength", "string"),
            FieldSpec("broadening", "string"),
            FieldSpec("grid_path", "path"),
            FieldSpec("peak_path", "path"),
            FieldSpec("split", "string"),
        ),
    ),
    "provenance_edges": ManifestSchema(
        name="provenance_edges",
        id_field="",
        fields=(
            FieldSpec("source_id", "string"),
            FieldSpec("source_type", "string"),
            FieldSpec("target_id", "string"),
            FieldSpec("target_type", "string"),
            FieldSpec("relation_type", "string"),
        ),
    ),
}


def canonical_structure_hash(structure_doc: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a pymatgen-style structure JSON dict."""

    if not isinstance(structure_doc, Mapping):
        raise TypeError("structure_doc must be a mapping")
    payload = json.dumps(
        structure_doc,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def potential_workspace_paths(potential_id: str) -> dict[str, str]:
    """Return the canonical per-potential workspace layout."""

    root = f"data/potentials/{potential_id}"
    generated = f"{root}/generated"
    return {
        "root": root,
        "potential_dir": f"{root}/potential",
        "generated_data_dir": generated,
        "manifests_dir": f"{generated}/manifests",
        "relaxation_logs_dir": f"{generated}/relaxation_logs",
        "relaxed_structures_dir": f"{generated}/structures_relaxed",
        "xrd_grid_dir": f"{generated}/xrd/grid",
        "xrd_peaks_dir": f"{generated}/xrd/peaks",
        "xrd_average_grid_dir": f"{generated}/xrd/averages/grid",
        "xrd_average_peaks_dir": f"{generated}/xrd/averages/peaks",
        "audits_dir": f"{root}/audits",
    }


def validate_manifest_records(
    manifest_name: str,
    records: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Validate required fields and ID uniqueness for one manifest table."""

    schema = MANIFEST_SCHEMAS.get(manifest_name)
    if schema is None:
        return [
            ValidationIssue(
                code="unknown_manifest",
                message=f"Unknown manifest: {manifest_name}",
                manifest=manifest_name,
            )
        ]

    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    required_fields = [field.name for field in schema.fields if field.required]

    for index, record in enumerate(records):
        record_id = _string_id(record.get(schema.id_field)) if schema.id_field else None
        for field_name in required_fields:
            value = record.get(field_name)
            if value is None or value == "":
                issues.append(
                    ValidationIssue(
                        code="missing_required_field",
                        message=f"Missing required field {field_name}",
                        manifest=manifest_name,
                        record_index=index,
                        record_id=record_id,
                    )
                )

        if schema.id_field:
            if not record_id:
                continue
            if record_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        code="duplicate_id",
                        message=(
                            f"Duplicate {schema.id_field} {record_id}; "
                            f"first seen at row {seen_ids[record_id]}"
                        ),
                        manifest=manifest_name,
                        record_index=index,
                        record_id=record_id,
                    )
                )
            else:
                seen_ids[record_id] = index

    return issues


def validate_potential_storage_layout(
    potentials: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Validate that each potential declares its canonical storage directories."""

    issues: list[ValidationIssue] = []
    for index, potential in enumerate(potentials):
        potential_id = _string_id(potential.get("potential_id"))
        paths = potential_workspace_paths(potential_id)
        potential_dir = _normalize_manifest_path(_string_id(potential.get("potential_dir")))
        generated_data_dir = _normalize_manifest_path(_string_id(potential.get("generated_data_dir")))

        if potential_dir != paths["potential_dir"]:
            issues.append(
                ValidationIssue(
                    code="invalid_potential_dir",
                    message=f"Potential {potential_id} must use {paths['potential_dir']}",
                    manifest="potentials_catalog",
                    record_index=index,
                    record_id=potential_id,
                )
            )
        if generated_data_dir != paths["generated_data_dir"]:
            issues.append(
                ValidationIssue(
                    code="invalid_generated_data_dir",
                    message=f"Potential {potential_id} must use {paths['generated_data_dir']}",
                    manifest="potentials_catalog",
                    record_index=index,
                    record_id=potential_id,
                )
            )

    return issues


def validate_potential_scoped_artifacts(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ValidationIssue]:
    """Validate generated artifacts live under the folder for their potential."""

    issues = validate_potential_storage_layout(manifests.get("potentials_catalog", []))
    relax_to_potential: dict[str, str] = {}
    relaxed_to_potential: dict[str, str] = {}

    for index, relaxation in enumerate(manifests.get("relaxations", [])):
        relaxation_id = _string_id(relaxation.get("relaxation_run_id"))
        potential_id = _string_id(relaxation.get("potential_id"))
        relax_to_potential[relaxation_id] = potential_id
        expected_dir = potential_workspace_paths(potential_id)["relaxation_logs_dir"]
        issues.extend(
            _validate_path_scope(
                path_value=relaxation.get("log_path"),
                expected_dir=expected_dir,
                manifest="relaxations",
                record_index=index,
                record_id=relaxation_id,
                field_name="log_path",
            )
        )

    for index, relaxed in enumerate(manifests.get("relaxed_structures", [])):
        relaxed_id = _string_id(relaxed.get("relaxed_structure_id"))
        relaxation_id = _string_id(relaxed.get("relaxation_run_id"))
        potential_id = relax_to_potential.get(relaxation_id, "")
        if not potential_id:
            continue
        relaxed_to_potential[relaxed_id] = potential_id
        expected_dir = potential_workspace_paths(potential_id)["relaxed_structures_dir"]
        for field_name in ("artifact_path_json", "artifact_path_cif"):
            issues.extend(
                _validate_path_scope(
                    path_value=relaxed.get(field_name),
                    expected_dir=expected_dir,
                    manifest="relaxed_structures",
                    record_index=index,
                    record_id=relaxed_id,
                    field_name=field_name,
                )
            )

    for index, xrd in enumerate(manifests.get("xrd_samples", [])):
        xrd_id = _string_id(xrd.get("xrd_sample_id"))
        relaxed_id = _string_id(xrd.get("relaxed_structure_id"))
        potential_id = relaxed_to_potential.get(relaxed_id, "")
        if not potential_id:
            continue
        paths = potential_workspace_paths(potential_id)
        path_fields = {
            "grid_path": paths["xrd_grid_dir"],
            "peak_path": paths["xrd_peaks_dir"],
        }
        for field_name, expected_dir in path_fields.items():
            issues.extend(
                _validate_path_scope(
                    path_value=xrd.get(field_name),
                    expected_dir=expected_dir,
                    manifest="xrd_samples",
                    record_index=index,
                    record_id=xrd_id,
                    field_name=field_name,
                )
            )

    return issues


def validate_provenance_graph(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ValidationIssue]:
    """Validate the required xrd -> relaxed -> relaxation -> inputs chain."""

    issues: list[ValidationIssue] = []
    for manifest_name, records in manifests.items():
        if manifest_name in MANIFEST_SCHEMAS:
            issues.extend(validate_manifest_records(manifest_name, records))

    potentials = _index_by(manifests.get("potentials_catalog", []), "potential_id")
    initials = _index_by(manifests.get("initial_structures", []), "initial_structure_id")
    relaxations = _index_by(manifests.get("relaxations", []), "relaxation_run_id")
    relaxed_structures = _index_by(
        manifests.get("relaxed_structures", []),
        "relaxed_structure_id",
    )
    edges = _edge_set(manifests.get("provenance_edges", []))

    for xrd in manifests.get("xrd_samples", []):
        xrd_id = _string_id(xrd.get("xrd_sample_id"))
        relaxed_id = _string_id(xrd.get("relaxed_structure_id"))
        relaxed = relaxed_structures.get(relaxed_id)
        if relaxed is None:
            issues.append(
                ValidationIssue(
                    code="unknown_relaxed_structure",
                    message=f"XRD sample {xrd_id} points to missing relaxed structure {relaxed_id}",
                    manifest="xrd_samples",
                    record_id=xrd_id,
                )
            )
            continue

        relaxation_id = _string_id(relaxed.get("relaxation_run_id"))
        relaxation = relaxations.get(relaxation_id)
        if relaxation is None:
            issues.append(
                ValidationIssue(
                    code="unknown_relaxation_run",
                    message=f"Relaxed structure {relaxed_id} points to missing relaxation {relaxation_id}",
                    manifest="relaxed_structures",
                    record_id=relaxed_id,
                )
            )
            continue

        initial_id = _string_id(relaxation.get("initial_structure_id"))
        potential_id = _string_id(relaxation.get("potential_id"))
        relax_config_id = _string_id(relaxation.get("relax_config_id"))
        xrd_config_id = _string_id(xrd.get("xrd_config_id"))
        initial = initials.get(initial_id)
        potential = potentials.get(potential_id)

        if initial is None:
            issues.append(
                ValidationIssue(
                    code="unknown_initial_structure",
                    message=f"Relaxation {relaxation_id} points to missing initial structure {initial_id}",
                    manifest="relaxations",
                    record_id=relaxation_id,
                )
            )
        if potential is None:
            issues.append(
                ValidationIssue(
                    code="unknown_potential",
                    message=f"Relaxation {relaxation_id} points to missing potential {potential_id}",
                    manifest="relaxations",
                    record_id=relaxation_id,
                )
            )

        if initial is None or potential is None:
            continue

        required_edges = (
            (initial_id, "initial_structure", relaxation_id, "relaxation_run", "input_to"),
            (potential_id, "potential_file", relaxation_id, "relaxation_run", "used_by"),
            (relax_config_id, "relax_config", relaxation_id, "relaxation_run", "configured_by"),
            (relaxation_id, "relaxation_run", relaxed_id, "relaxed_structure", "produced"),
            (relaxed_id, "relaxed_structure", xrd_id, "xrd_sample", "simulated_as"),
            (xrd_config_id, "xrd_config", xrd_id, "xrd_sample", "configured_by"),
        )
        for edge in required_edges:
            if edge not in edges:
                issues.append(
                    ValidationIssue(
                        code="missing_provenance_edge",
                        message=f"Missing provenance edge {edge}",
                        manifest="provenance_edges",
                        record_id=xrd_id,
                    )
                )

    return issues


def validate_split_isolation(
    xrd_samples: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Ensure private test samples do not reuse released relaxed structures."""

    private_ids = {
        _string_id(row.get("relaxed_structure_id"))
        for row in xrd_samples
        if row.get("split") == "private_test"
    }
    released_ids = {
        _string_id(row.get("relaxed_structure_id"))
        for row in xrd_samples
        if row.get("split") != "private_test"
    }
    leaks = sorted(private_ids & released_ids)

    return [
        ValidationIssue(
            code="private_relaxed_structure_leak",
            message=f"Private test shares released relaxed structure {relaxed_id}",
            manifest="xrd_samples",
            record_id=relaxed_id,
        )
        for relaxed_id in leaks
        if relaxed_id
    ]


def validate_release_potentials(
    potentials: Sequence[Mapping[str, Any]],
    release_paths: Iterable[str],
) -> list[ValidationIssue]:
    """Block packaging potential files unless redistribution is allowed."""

    path_parts = [_split_release_path(path) for path in release_paths]
    blocked_policies = {
        "metadata_only",
        "external_only",
        "not_redistributable",
        "no_redistribution",
    }
    issues: list[ValidationIssue] = []

    for potential in potentials:
        potential_id = _string_id(potential.get("potential_id"))
        policy = _string_id(potential.get("redistribution_policy"))
        if not potential_id or policy not in blocked_policies:
            continue
        if any(potential_id in parts for parts in path_parts):
            issues.append(
                ValidationIssue(
                    code="non_redistributable_potential_packaged",
                    message=f"Potential {potential_id} has policy {policy} but appears in release artifacts",
                    manifest="potentials_catalog",
                    record_id=potential_id,
                )
            )

    return issues


def validate_artifact_hashes(
    records: Sequence[Mapping[str, Any]],
    *,
    hash_field: str,
    path_field: str,
    root: str | Path,
    id_field: str | None = None,
) -> list[ValidationIssue]:
    """Validate that artifact files exist under root and match manifest hashes."""

    root_path = Path(root).resolve()
    issues: list[ValidationIssue] = []

    for index, record in enumerate(records):
        record_id = _string_id(record.get(id_field)) if id_field else None
        expected_hash = _string_id(record.get(hash_field)).lower()
        artifact_value = _string_id(record.get(path_field))
        artifact_path = (root_path / artifact_value).resolve()

        if not _is_relative_to(artifact_path, root_path):
            issues.append(
                ValidationIssue(
                    code="artifact_path_outside_root",
                    message=f"Artifact path {artifact_value} resolves outside validation root",
                    record_index=index,
                    record_id=record_id,
                )
            )
            continue

        if not artifact_path.exists():
            issues.append(
                ValidationIssue(
                    code="missing_artifact",
                    message=f"Artifact file is missing: {artifact_value}",
                    record_index=index,
                    record_id=record_id,
                )
            )
            continue

        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            issues.append(
                ValidationIssue(
                    code="artifact_hash_mismatch",
                    message=f"Artifact hash mismatch for {artifact_value}",
                    record_index=index,
                    record_id=record_id,
                )
            )

    return issues


def validate_structure_json_hashes(
    records: Sequence[Mapping[str, Any]],
    *,
    root: str | Path,
    path_field: str,
    id_field: str,
    canonical_hash_field: str = "structure_sha256",
    file_hash_field: str | None = None,
) -> list[ValidationIssue]:
    """Validate canonical pymatgen JSON hashes and optional file byte hashes."""

    root_path = Path(root).resolve()
    issues: list[ValidationIssue] = []

    for index, record in enumerate(records):
        record_id = _string_id(record.get(id_field))
        artifact_value = _string_id(record.get(path_field))
        artifact_path = (root_path / artifact_value).resolve()

        if not _is_relative_to(artifact_path, root_path):
            issues.append(
                ValidationIssue(
                    code="artifact_path_outside_root",
                    message=f"Artifact path {artifact_value} resolves outside validation root",
                    record_index=index,
                    record_id=record_id,
                )
            )
            continue
        if not artifact_path.exists():
            issues.append(
                ValidationIssue(
                    code="missing_artifact",
                    message=f"Artifact file is missing: {artifact_value}",
                    record_index=index,
                    record_id=record_id,
                )
            )
            continue

        try:
            structure_doc = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid_structure_json",
                    message=f"Invalid structure JSON in {artifact_value}: {exc}",
                    record_index=index,
                    record_id=record_id,
                )
            )
            continue

        expected_structure_hash = _string_id(record.get(canonical_hash_field)).lower()
        actual_structure_hash = canonical_structure_hash(structure_doc)
        if actual_structure_hash != expected_structure_hash:
            issues.append(
                ValidationIssue(
                    code="structure_hash_mismatch",
                    message=f"Canonical structure hash mismatch for {artifact_value}",
                    record_index=index,
                    record_id=record_id,
                )
            )

        if file_hash_field and record.get(file_hash_field):
            expected_file_hash = _string_id(record.get(file_hash_field)).lower()
            actual_file_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_file_hash != expected_file_hash:
                issues.append(
                    ValidationIssue(
                        code="artifact_hash_mismatch",
                        message=f"Artifact file hash mismatch for {artifact_value}",
                        record_index=index,
                        record_id=record_id,
                    )
                )

    return issues


def validate_relaxation_quality(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] = DEFAULT_RELAX_CONFIG,
) -> list[ValidationIssue]:
    """Validate default relaxation quality gates."""

    issues: list[ValidationIssue] = []
    fmax_limit = float(config["fmax_eV_per_A"])
    volume_limit = float(config["abnormal_volume_change_threshold"])
    max_steps = int(config["max_steps"])

    for index, record in enumerate(records):
        record_id = _string_id(record.get("relaxation_run_id"))
        fmax = _as_float(record.get("fmax"))
        volume_change = _as_float(record.get("volume_change"))
        steps = _as_int(record.get("steps"))

        if fmax is not None and fmax > fmax_limit:
            issues.append(
                ValidationIssue(
                    code="relaxation_fmax_above_default",
                    message=f"Relaxation {record_id} has fmax {fmax} above {fmax_limit}",
                    manifest="relaxations",
                    record_index=index,
                    record_id=record_id,
                )
            )
        if volume_change is not None and abs(volume_change) > volume_limit:
            issues.append(
                ValidationIssue(
                    code="abnormal_volume_change",
                    message=f"Relaxation {record_id} has volume_change {volume_change} above {volume_limit}",
                    manifest="relaxations",
                    record_index=index,
                    record_id=record_id,
                )
            )
        if steps is not None and steps > max_steps:
            issues.append(
                ValidationIssue(
                    code="relaxation_steps_above_default",
                    message=f"Relaxation {record_id} used {steps} steps above {max_steps}",
                    manifest="relaxations",
                    record_index=index,
                    record_id=record_id,
                )
            )

    return issues


def validate_xrd_geometry(
    records: Sequence[Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Validate XRD two-theta range and grid step."""

    issues: list[ValidationIssue] = []
    for index, record in enumerate(records):
        record_id = _string_id(record.get("xrd_sample_id"))
        two_theta_min = _as_float(record.get("two_theta_min"))
        two_theta_max = _as_float(record.get("two_theta_max"))
        step = _as_float(record.get("step"))

        if (
            two_theta_min is not None
            and two_theta_max is not None
            and two_theta_min >= two_theta_max
        ):
            issues.append(
                ValidationIssue(
                    code="invalid_two_theta_range",
                    message=f"XRD sample {record_id} has invalid two-theta range",
                    manifest="xrd_samples",
                    record_index=index,
                    record_id=record_id,
                )
            )
        if step is not None and step <= 0:
            issues.append(
                ValidationIssue(
                    code="invalid_two_theta_step",
                    message=f"XRD sample {record_id} has non-positive two-theta step",
                    manifest="xrd_samples",
                    record_index=index,
                    record_id=record_id,
                )
            )

    return issues


def schema_payloads() -> dict[str, dict[str, Any]]:
    """Return JSON-serializable schema payloads for writing schema files."""

    return {
        name: {
            "name": schema.name,
            "id_field": schema.id_field,
            "fields": [
                {
                    "name": field.name,
                    "dtype": field.dtype,
                    "required": field.required,
                    "description": field.description,
                }
                for field in schema.fields
            ],
        }
        for name, schema in MANIFEST_SCHEMAS.items()
    }


def _index_by(
    records: Sequence[Mapping[str, Any]],
    id_field: str,
) -> dict[str, Mapping[str, Any]]:
    return {
        _string_id(record.get(id_field)): record
        for record in records
        if _string_id(record.get(id_field))
    }


def _edge_set(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            _string_id(record.get("source_id")),
            _string_id(record.get("source_type")),
            _string_id(record.get("target_id")),
            _string_id(record.get("target_type")),
            _string_id(record.get("relation_type")),
        )
        for record in records
    }


def _split_release_path(path: str) -> set[str]:
    normalized = path.replace("\\", "/")
    return {part for part in normalized.split("/") if part}


def _validate_path_scope(
    *,
    path_value: Any,
    expected_dir: str,
    manifest: str,
    record_index: int,
    record_id: str,
    field_name: str,
) -> list[ValidationIssue]:
    path = _normalize_manifest_path(_string_id(path_value))
    expected = _normalize_manifest_path(expected_dir)
    if path == expected or path.startswith(f"{expected}/"):
        return []
    return [
        ValidationIssue(
            code="artifact_outside_potential_scope",
            message=f"{field_name} for {record_id} must live under {expected}",
            manifest=manifest,
            record_index=record_index,
            record_id=record_id,
        )
    ]


def _normalize_manifest_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_id(value: Any) -> str:
    return "" if value is None else str(value)
