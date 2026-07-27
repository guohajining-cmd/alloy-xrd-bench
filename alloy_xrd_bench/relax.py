"""Relaxation runners for Alloy-XRD-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from .contract import DEFAULT_RELAX_CONFIG, canonical_structure_hash, potential_workspace_paths
from .mlip import _load_potential_row, adapter_for_engine, smoke_mlip_initial_structure


def smoke_relax_initial_structures(
    *,
    root: str | Path,
    potential_id: str,
    quality_tier: str = "smoke_only",
) -> list[dict[str, Any]]:
    """Create smoke relaxed-structure artifacts from shared initial structures."""

    root_path = Path(root).resolve()
    initial_manifest_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not initial_manifest_path.exists():
        raise FileNotFoundError(f"Missing initial structures manifest: {initial_manifest_path}")

    relative_paths = potential_workspace_paths(potential_id)
    paths = {name: root_path / relative for name, relative in relative_paths.items()}
    for key in ("manifests_dir", "relaxation_logs_dir", "relaxed_structures_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    initial_rows = pd.read_parquet(initial_manifest_path).to_dict(orient="records")
    relaxation_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for initial in initial_rows:
        initial_id = str(initial["initial_structure_id"])
        suffix = initial_id[5:] if initial_id.startswith("init_") else initial_id
        relaxation_id = f"relax_{potential_id}_{suffix}"
        relaxed_id = f"rstr_{potential_id}_{suffix}"
        initial_path = root_path / str(initial["artifact_path"])
        structure_doc = json.loads(initial_path.read_text(encoding="utf-8"))
        structure = Structure.from_dict(structure_doc)
        relaxed_bytes = json.dumps(
            structure.as_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        cif_text = structure.to(fmt="cif")
        cif_bytes = cif_text.encode("utf-8")

        relaxed_json_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.json"
        relaxed_cif_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.cif"
        log_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.jsonl"
        (root_path / relaxed_json_relative).write_bytes(relaxed_bytes)
        (root_path / relaxed_cif_relative).write_bytes(cif_bytes)
        (root_path / log_relative).write_text(
            json.dumps(
                {
                    "runner": "identity_smoke",
                    "initial_structure_id": initial_id,
                    "relaxed_structure_id": relaxed_id,
                    "note": "Smoke runner copies the initial structure; not an MLIP relaxation.",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        relaxation_rows.append(
            {
                "relaxation_run_id": relaxation_id,
                "initial_structure_id": initial_id,
                "potential_id": potential_id,
                "engine": "identity_smoke",
                "relax_config_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                "converged": True,
                "fmax": 0.0,
                "stress": {"xx": 0.0, "yy": 0.0, "zz": 0.0},
                "volume_change": 0.0,
                "steps": 0,
                "log_path": log_relative,
            }
        )
        relaxed_rows.append(
            {
                "relaxed_structure_id": relaxed_id,
                "relaxation_run_id": relaxation_id,
                "structure_sha256": canonical_structure_hash(structure.as_dict()),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": relaxed_json_relative,
                "artifact_path_cif": relaxed_cif_relative,
                "quality_tier": quality_tier,
            }
        )
        provenance_rows.extend(
            [
                {
                    "source_id": initial_id,
                    "source_type": "initial_structure",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "input_to",
                },
                {
                    "source_id": potential_id,
                    "source_type": "potential_file",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "used_by",
                },
                {
                    "source_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                    "source_type": "relax_config",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "configured_by",
                },
                {
                    "source_id": relaxation_id,
                    "source_type": "relaxation_run",
                    "target_id": relaxed_id,
                    "target_type": "relaxed_structure",
                    "relation_type": "produced",
                },
            ]
        )

    _upsert_manifest(paths["manifests_dir"] / "relaxations.parquet", relaxation_rows, "relaxation_run_id")
    _upsert_manifest(paths["manifests_dir"] / "relaxed_structures.parquet", relaxed_rows, "relaxed_structure_id")
    _append_unique_edges(paths["manifests_dir"] / "provenance_edges.parquet", provenance_rows)
    _write_potential_gamma_prime_pairs(root_path, potential_id, paths["manifests_dir"])
    return relaxed_rows


def chgnet_relax_initial_structures(
    *,
    root: str | Path,
    potential_id: str,
    model_name: str = "0.3.0",
    use_device: str | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    relax_cell: bool = True,
) -> list[dict[str, Any]]:
    """Relax shared initial structures using CHGNet and write manifests."""

    from chgnet.model import CHGNet, StructOptimizer

    root_path = Path(root).resolve()
    initial_manifest_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not initial_manifest_path.exists():
        raise FileNotFoundError(f"Missing initial structures manifest: {initial_manifest_path}")

    relative_paths = potential_workspace_paths(potential_id)
    paths = {name: root_path / relative for name, relative in relative_paths.items()}
    for key in ("manifests_dir", "relaxation_logs_dir", "relaxed_structures_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    model = CHGNet.load(model_name=model_name, use_device=use_device, verbose=False)
    optimizer = StructOptimizer(model=model, use_device=use_device)
    initial_rows = pd.read_parquet(initial_manifest_path).to_dict(orient="records")
    relaxation_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for initial in initial_rows:
        initial_id = str(initial["initial_structure_id"])
        suffix = initial_id[5:] if initial_id.startswith("init_") else initial_id
        relaxation_id = f"relax_{potential_id}_{suffix}"
        relaxed_id = f"rstr_{potential_id}_{suffix}"
        initial_path = root_path / str(initial["artifact_path"])
        initial_structure = Structure.from_dict(json.loads(initial_path.read_text(encoding="utf-8")))
        result = optimizer.relax(
            initial_structure,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            verbose=False,
        )
        final_structure = result["final_structure"]
        trajectory = result["trajectory"]
        actual_fmax = _trajectory_fmax(trajectory)
        stress = _trajectory_stress(trajectory)
        steps_used = max(0, len(getattr(trajectory, "energies", [])) - 1)
        volume_change = (
            float(final_structure.volume) - float(initial_structure.volume)
        ) / float(initial_structure.volume)
        converged = actual_fmax <= fmax
        quality_tier = (
            "pass"
            if actual_fmax <= DEFAULT_RELAX_CONFIG["fmax_eV_per_A"]
            and abs(volume_change) <= DEFAULT_RELAX_CONFIG["abnormal_volume_change_threshold"]
            and steps_used <= DEFAULT_RELAX_CONFIG["max_steps"]
            else "fail"
        )

        relaxed_bytes = json.dumps(
            final_structure.as_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        cif_bytes = final_structure.to(fmt="cif").encode("utf-8")
        relaxed_json_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.json"
        relaxed_cif_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.cif"
        log_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.jsonl"
        (root_path / relaxed_json_relative).write_bytes(relaxed_bytes)
        (root_path / relaxed_cif_relative).write_bytes(cif_bytes)
        _write_relax_log(
            root_path / log_relative,
            {
                "runner": "chgnet",
                "model_name": model_name,
                "use_device": use_device,
                "initial_structure_id": initial_id,
                "relaxed_structure_id": relaxed_id,
                "requested_fmax": fmax,
                "requested_steps": steps,
                "relax_cell": relax_cell,
                "actual_fmax": actual_fmax,
                "converged": converged,
                "steps": steps_used,
                "volume_change": volume_change,
                "energies": [float(value) for value in getattr(trajectory, "energies", [])],
            },
        )

        relaxation_rows.append(
            {
                "relaxation_run_id": relaxation_id,
                "initial_structure_id": initial_id,
                "potential_id": potential_id,
                "engine": "chgnet",
                "relax_config_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                "converged": converged,
                "fmax": actual_fmax,
                "stress": stress,
                "volume_change": volume_change,
                "steps": steps_used,
                "log_path": log_relative,
            }
        )
        relaxed_rows.append(
            {
                "relaxed_structure_id": relaxed_id,
                "relaxation_run_id": relaxation_id,
                "structure_sha256": canonical_structure_hash(final_structure.as_dict()),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": relaxed_json_relative,
                "artifact_path_cif": relaxed_cif_relative,
                "quality_tier": quality_tier,
            }
        )
        provenance_rows.extend(
            [
                {
                    "source_id": initial_id,
                    "source_type": "initial_structure",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "input_to",
                },
                {
                    "source_id": potential_id,
                    "source_type": "potential_file",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "used_by",
                },
                {
                    "source_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                    "source_type": "relax_config",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "configured_by",
                },
                {
                    "source_id": relaxation_id,
                    "source_type": "relaxation_run",
                    "target_id": relaxed_id,
                    "target_type": "relaxed_structure",
                    "relation_type": "produced",
                },
            ]
        )

    _upsert_manifest(paths["manifests_dir"] / "relaxations.parquet", relaxation_rows, "relaxation_run_id")
    _upsert_manifest(paths["manifests_dir"] / "relaxed_structures.parquet", relaxed_rows, "relaxed_structure_id")
    _append_unique_edges(paths["manifests_dir"] / "provenance_edges.parquet", provenance_rows)
    _write_potential_gamma_prime_pairs(root_path, potential_id, paths["manifests_dir"])
    return relaxed_rows


def ase_relax_initial_structures(
    *,
    root: str | Path,
    potential_id: str,
    fmax: float = 0.05,
    steps: int = 1000,
    relax_cell: bool = False,
) -> list[dict[str, Any]]:
    """Relax shared initial structures using an ASE calculator from an MLIP adapter."""

    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    root_path = Path(root).resolve()
    initial_manifest_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not initial_manifest_path.exists():
        raise FileNotFoundError(f"Missing initial structures manifest: {initial_manifest_path}")

    potential = _load_potential_row(root_path, potential_id)
    if potential is None:
        raise ValueError(f"Potential {potential_id} is not declared in potentials_catalog.yml")
    adapter = adapter_for_engine(str(potential.get("engine", "")))
    load_result = adapter.load(root_path, potential)
    if not bool(load_result.get("ok")):
        raise RuntimeError(load_result.get("message", "Failed to load ASE MLIP adapter"))
    calculator = getattr(adapter, "_calculator", None)
    if calculator is None:
        raise RuntimeError(f"Adapter for {potential_id} did not expose an ASE calculator")

    relative_paths = potential_workspace_paths(potential_id)
    paths = {name: root_path / relative for name, relative in relative_paths.items()}
    for key in ("manifests_dir", "relaxation_logs_dir", "relaxed_structures_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    initial_rows = pd.read_parquet(initial_manifest_path).to_dict(orient="records")
    relaxation_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    adaptor = AseAtomsAdaptor()

    for initial in initial_rows:
        initial_id = str(initial["initial_structure_id"])
        unsupported = _unsupported_initial_elements(potential, initial)
        if unsupported:
            continue
        if _unsupported_initial_phase(potential, initial):
            continue
        suffix = initial_id[5:] if initial_id.startswith("init_") else initial_id
        relaxation_id = f"relax_{potential_id}_{suffix}"
        relaxed_id = f"rstr_{potential_id}_{suffix}"
        initial_path = root_path / str(initial["artifact_path"])
        initial_structure = Structure.from_dict(json.loads(initial_path.read_text(encoding="utf-8")))
        atoms = adaptor.get_atoms(initial_structure)
        atoms.calc = calculator
        initial_volume = float(atoms.get_volume())

        optimize_target: Any = atoms
        if relax_cell:
            try:
                from ase.filters import FrechetCellFilter

                optimize_target = FrechetCellFilter(atoms)
            except Exception:
                from ase.filters import UnitCellFilter

                optimize_target = UnitCellFilter(atoms)

        optimizer = BFGS(optimize_target, logfile=None)
        converged = bool(optimizer.run(fmax=fmax, steps=steps))
        steps_used = int(optimizer.get_number_of_steps())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        actual_fmax = _max_force_from_array(forces)
        try:
            stress = _stress_dict(atoms.get_stress())
        except Exception:
            stress = {"xx": 0.0, "yy": 0.0, "zz": 0.0}
        final_structure = adaptor.get_structure(atoms)
        volume_change = (float(final_structure.volume) - initial_volume) / initial_volume
        quality_tier = (
            "pass"
            if actual_fmax <= DEFAULT_RELAX_CONFIG["fmax_eV_per_A"]
            and abs(volume_change) <= DEFAULT_RELAX_CONFIG["abnormal_volume_change_threshold"]
            and steps_used <= DEFAULT_RELAX_CONFIG["max_steps"]
            else "fail"
        )

        relaxed_bytes = json.dumps(
            final_structure.as_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        cif_bytes = final_structure.to(fmt="cif").encode("utf-8")
        relaxed_json_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.json"
        relaxed_cif_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.cif"
        log_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.jsonl"
        (root_path / relaxed_json_relative).write_bytes(relaxed_bytes)
        (root_path / relaxed_cif_relative).write_bytes(cif_bytes)
        _write_relax_log(
            root_path / log_relative,
            {
                "runner": "ase_mlip",
                "adapter": load_result.get("adapter", ""),
                "engine": potential.get("engine", ""),
                "initial_structure_id": initial_id,
                "relaxed_structure_id": relaxed_id,
                "requested_fmax": fmax,
                "requested_steps": steps,
                "relax_cell": relax_cell,
                "actual_fmax": actual_fmax,
                "converged": converged,
                "steps": steps_used,
                "volume_change": volume_change,
            },
        )

        relaxation_rows.append(
            {
                "relaxation_run_id": relaxation_id,
                "initial_structure_id": initial_id,
                "potential_id": potential_id,
                "engine": str(potential.get("engine", "")),
                "relax_config_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                "converged": converged,
                "fmax": actual_fmax,
                "stress": stress,
                "volume_change": volume_change,
                "steps": steps_used,
                "log_path": log_relative,
            }
        )
        relaxed_rows.append(
            {
                "relaxed_structure_id": relaxed_id,
                "relaxation_run_id": relaxation_id,
                "structure_sha256": canonical_structure_hash(final_structure.as_dict()),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": relaxed_json_relative,
                "artifact_path_cif": relaxed_cif_relative,
                "quality_tier": quality_tier,
            }
        )
        provenance_rows.extend(
            [
                {
                    "source_id": initial_id,
                    "source_type": "initial_structure",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "input_to",
                },
                {
                    "source_id": potential_id,
                    "source_type": "potential_file",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "used_by",
                },
                {
                    "source_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                    "source_type": "relax_config",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "configured_by",
                },
                {
                    "source_id": relaxation_id,
                    "source_type": "relaxation_run",
                    "target_id": relaxed_id,
                    "target_type": "relaxed_structure",
                    "relation_type": "produced",
                },
            ]
        )

    _upsert_manifest(paths["manifests_dir"] / "relaxations.parquet", relaxation_rows, "relaxation_run_id")
    _upsert_manifest(paths["manifests_dir"] / "relaxed_structures.parquet", relaxed_rows, "relaxed_structure_id")
    _append_unique_edges(paths["manifests_dir"] / "provenance_edges.parquet", provenance_rows)
    _write_potential_gamma_prime_pairs(root_path, potential_id, paths["manifests_dir"])
    return relaxed_rows


def mtp_mlp_relax_initial_structures(
    *,
    root: str | Path,
    potential_id: str,
    mlp_command: Sequence[str] | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    relax_cell: bool = False,
) -> list[dict[str, Any]]:
    """Relax shared initial structures using an external MLIP-3 ``mlp`` executable."""

    root_path = Path(root).resolve()
    initial_manifest_path = root_path / "data" / "shared" / "manifests" / "initial_structures.parquet"
    if not initial_manifest_path.exists():
        raise FileNotFoundError(f"Missing initial structures manifest: {initial_manifest_path}")

    potential = _load_potential_row(root_path, potential_id)
    if potential is None:
        raise ValueError(f"Potential {potential_id} is not declared in potentials_catalog.yml")

    type_to_element, element_to_type = _mtp_type_maps(potential)
    model_file = _mtp_model_file(root_path, potential)
    model_relative = _relative_path(root_path, model_file)
    command = _resolve_mlp_command(root_path, mlp_command)

    relative_paths = potential_workspace_paths(potential_id)
    paths = {name: root_path / relative for name, relative in relative_paths.items()}
    for key in ("manifests_dir", "relaxation_logs_dir", "relaxed_structures_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    initial_rows = pd.read_parquet(initial_manifest_path).to_dict(orient="records")
    relaxation_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for initial in initial_rows:
        unsupported = _unsupported_initial_elements(potential, initial)
        if unsupported:
            continue
        if _unsupported_initial_phase(potential, initial):
            continue

        initial_id = str(initial["initial_structure_id"])
        suffix = initial_id[5:] if initial_id.startswith("init_") else initial_id
        relaxation_id = f"relax_{potential_id}_{suffix}"
        relaxed_id = f"rstr_{potential_id}_{suffix}"
        initial_path = root_path / str(initial["artifact_path"])
        initial_structure = Structure.from_dict(json.loads(initial_path.read_text(encoding="utf-8")))
        initial_volume = float(initial_structure.volume)

        input_cfg_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.input.cfg"
        output_cfg_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.relaxed.cfg"
        settings_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.relax.ini"
        raw_log_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.mlp.log"
        log_relative = f"{relative_paths['relaxation_logs_dir']}/{relaxation_id}.jsonl"

        (root_path / input_cfg_relative).write_text(
            _structure_to_mtp_cfg(
                initial_structure,
                element_to_type=element_to_type,
                energy=0.0,
                stress=[0.0] * 6,
                efs_by="alloy-xrd-bench-mtp-mlp-input",
            ),
            encoding="utf-8",
        )
        (root_path / settings_relative).write_text(
            _mtp_relax_settings(
                fmax=fmax,
                steps=steps,
                relax_cell=relax_cell,
                raw_log_relative=raw_log_relative,
            ),
            encoding="utf-8",
        )

        _run_mlp_command(
            root_path=root_path,
            command=command,
            args=[
                "relax",
                model_relative,
                input_cfg_relative,
                output_cfg_relative,
                f"--relaxation_settings={settings_relative}",
            ],
        )

        parsed = _parse_mtp_cfg(root_path / output_cfg_relative, type_to_element=type_to_element)
        final_structure = parsed["structure"]
        actual_fmax = float(parsed["max_force"])
        stress = _stress_dict_from_sequence(parsed.get("stress") or [])
        steps_used = _parse_mlp_relax_steps(root_path / raw_log_relative)
        volume_change = (float(final_structure.volume) - initial_volume) / initial_volume
        converged = actual_fmax <= fmax
        quality_tier = (
            "pass"
            if actual_fmax <= DEFAULT_RELAX_CONFIG["fmax_eV_per_A"]
            and abs(volume_change) <= DEFAULT_RELAX_CONFIG["abnormal_volume_change_threshold"]
            and steps_used <= DEFAULT_RELAX_CONFIG["max_steps"]
            else "fail"
        )

        relaxed_bytes = json.dumps(
            final_structure.as_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        cif_bytes = final_structure.to(fmt="cif").encode("utf-8")
        relaxed_json_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.json"
        relaxed_cif_relative = f"{relative_paths['relaxed_structures_dir']}/{relaxed_id}.cif"
        (root_path / relaxed_json_relative).write_bytes(relaxed_bytes)
        (root_path / relaxed_cif_relative).write_bytes(cif_bytes)
        _write_relax_log(
            root_path / log_relative,
            {
                "runner": "mtp_mlp",
                "engine": potential.get("engine", ""),
                "mlp_command": command.get("display", command.get("argv", [])),
                "model_file": model_relative,
                "type_map": {str(key): value for key, value in type_to_element.items()},
                "initial_structure_id": initial_id,
                "relaxed_structure_id": relaxed_id,
                "input_cfg": input_cfg_relative,
                "relaxed_cfg": output_cfg_relative,
                "raw_log_path": raw_log_relative,
                "requested_fmax": fmax,
                "requested_steps": steps,
                "relax_cell": relax_cell,
                "actual_fmax": actual_fmax,
                "converged": converged,
                "steps": steps_used,
                "volume_change": volume_change,
                "energy_eV": parsed.get("energy"),
            },
        )

        relaxation_rows.append(
            {
                "relaxation_run_id": relaxation_id,
                "initial_structure_id": initial_id,
                "potential_id": potential_id,
                "engine": str(potential.get("engine", "")),
                "relax_config_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                "converged": converged,
                "fmax": actual_fmax,
                "stress": stress,
                "volume_change": volume_change,
                "steps": steps_used,
                "log_path": log_relative,
            }
        )
        relaxed_rows.append(
            {
                "relaxed_structure_id": relaxed_id,
                "relaxation_run_id": relaxation_id,
                "structure_sha256": canonical_structure_hash(final_structure.as_dict()),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": relaxed_json_relative,
                "artifact_path_cif": relaxed_cif_relative,
                "quality_tier": quality_tier,
            }
        )
        provenance_rows.extend(
            [
                {
                    "source_id": initial_id,
                    "source_type": "initial_structure",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "input_to",
                },
                {
                    "source_id": potential_id,
                    "source_type": "potential_file",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "used_by",
                },
                {
                    "source_id": DEFAULT_RELAX_CONFIG["relax_config_id"],
                    "source_type": "relax_config",
                    "target_id": relaxation_id,
                    "target_type": "relaxation_run",
                    "relation_type": "configured_by",
                },
                {
                    "source_id": relaxation_id,
                    "source_type": "relaxation_run",
                    "target_id": relaxed_id,
                    "target_type": "relaxed_structure",
                    "relation_type": "produced",
                },
            ]
        )

    _upsert_manifest(paths["manifests_dir"] / "relaxations.parquet", relaxation_rows, "relaxation_run_id")
    _upsert_manifest(paths["manifests_dir"] / "relaxed_structures.parquet", relaxed_rows, "relaxed_structure_id")
    _append_unique_edges(paths["manifests_dir"] / "provenance_edges.parquet", provenance_rows)
    _write_potential_gamma_prime_pairs(root_path, potential_id, paths["manifests_dir"])
    return relaxed_rows


def _unsupported_initial_elements(potential: Mapping[str, Any], initial: Mapping[str, Any]) -> list[str]:
    supported = {str(element) for element in potential.get("elements", [])}
    if not supported:
        return []
    composition = initial.get("composition", {})
    if isinstance(composition, Mapping):
        elements = {
            str(element)
            for element, amount in composition.items()
            if amount is not None and float(amount) > 0
        }
        return sorted(elements - supported)
    return []


def _unsupported_initial_phase(potential: Mapping[str, Any], initial: Mapping[str, Any]) -> str:
    supported = {str(phase).lower() for phase in potential.get("phases", []) if str(phase)}
    if not supported:
        return ""
    phase = str(initial.get("phase", "")).lower()
    return phase if phase and phase not in supported else ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Alloy-XRD-Bench relaxation workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Create smoke relaxed structures from initial structures")
    smoke.add_argument("--root", default=".", help="Benchmark workspace root")
    smoke.add_argument("--potential-id", required=True)
    smoke.add_argument("--quality-tier", default="smoke_only")

    chgnet = subparsers.add_parser("chgnet", help="Relax shared initial structures with CHGNet")
    chgnet.add_argument("--root", default=".", help="Benchmark workspace root")
    chgnet.add_argument("--potential-id", required=True)
    chgnet.add_argument("--model-name", default="0.3.0")
    chgnet.add_argument("--use-device")
    chgnet.add_argument("--fmax", type=float, default=DEFAULT_RELAX_CONFIG["fmax_eV_per_A"])
    chgnet.add_argument("--steps", type=int, default=DEFAULT_RELAX_CONFIG["max_steps"])
    chgnet.add_argument("--relax-cell", dest="relax_cell", action="store_true", default=True)
    chgnet.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")

    ase_mlip = subparsers.add_parser("ase-mlip", help="Relax shared initial structures with an ASE-backed MLIP")
    ase_mlip.add_argument("--root", default=".", help="Benchmark workspace root")
    ase_mlip.add_argument("--potential-id", required=True)
    ase_mlip.add_argument("--fmax", type=float, default=DEFAULT_RELAX_CONFIG["fmax_eV_per_A"])
    ase_mlip.add_argument("--steps", type=int, default=DEFAULT_RELAX_CONFIG["max_steps"])
    ase_mlip.add_argument("--relax-cell", dest="relax_cell", action="store_true", default=False)
    ase_mlip.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")

    mtp_mlp = subparsers.add_parser("mtp-mlp", help="Relax shared initial structures with an MLIP-3 mlp executable")
    mtp_mlp.add_argument("--root", default=".", help="Benchmark workspace root")
    mtp_mlp.add_argument("--potential-id", required=True)
    mtp_mlp.add_argument("--mlp-command-json", help="JSON array command prefix, for example [\"mlp\"]")
    mtp_mlp.add_argument("--fmax", type=float, default=DEFAULT_RELAX_CONFIG["fmax_eV_per_A"])
    mtp_mlp.add_argument("--steps", type=int, default=DEFAULT_RELAX_CONFIG["max_steps"])
    mtp_mlp.add_argument("--relax-cell", dest="relax_cell", action="store_true", default=False)
    mtp_mlp.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")

    smoke_mlip = subparsers.add_parser(
        "smoke-mlip",
        help="Run a non-manifest MLIP smoke audit for one initial structure",
    )
    smoke_mlip.add_argument("--root", default=".", help="Benchmark workspace root")
    smoke_mlip.add_argument("--potential-id", required=True)
    smoke_mlip.add_argument("--initial-id", required=True)
    smoke_mlip.add_argument("--json", action="store_true", help="Emit JSON")

    args = parser.parse_args(argv)
    if args.command == "smoke":
        rows = smoke_relax_initial_structures(
            root=args.root,
            potential_id=args.potential_id,
            quality_tier=args.quality_tier,
        )
        print(f"generated_relaxed_structures: {len(rows)}")
        return 0
    if args.command == "chgnet":
        rows = chgnet_relax_initial_structures(
            root=args.root,
            potential_id=args.potential_id,
            model_name=args.model_name,
            use_device=args.use_device,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=args.relax_cell,
        )
        print(f"generated_relaxed_structures: {len(rows)}")
        return 0
    if args.command == "ase-mlip":
        rows = ase_relax_initial_structures(
            root=args.root,
            potential_id=args.potential_id,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=args.relax_cell,
        )
        print(f"generated_relaxed_structures: {len(rows)}")
        return 0
    if args.command == "mtp-mlp":
        mlp_command = json.loads(args.mlp_command_json) if args.mlp_command_json else None
        rows = mtp_mlp_relax_initial_structures(
            root=args.root,
            potential_id=args.potential_id,
            mlp_command=mlp_command,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=args.relax_cell,
        )
        print(f"generated_relaxed_structures: {len(rows)}")
        return 0
    if args.command == "smoke-mlip":
        report = smoke_mlip_initial_structure(
            root=args.root,
            potential_id=args.potential_id,
            initial_id=args.initial_id,
            write_audit=True,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"status: {report['status']}")
            print(f"potential_id: {report['potential_id']}")
            print(f"initial_structure_id: {report['initial_structure_id']}")
        return 0 if report["status"] in {"pass", "skipped_unsupported_elements", "skipped_unsupported_phase"} else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _upsert_manifest(path: Path, rows: list[dict[str, Any]], id_field: str) -> None:
    if path.exists():
        existing = pd.read_parquet(path).to_dict(orient="records")
    else:
        existing = []
    incoming_ids = {row[id_field] for row in rows}
    merged = [row for row in existing if row.get(id_field) not in incoming_ids] + rows
    pd.DataFrame(merged).to_parquet(path)


def _append_unique_edges(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        existing = pd.read_parquet(path).to_dict(orient="records")
    else:
        existing = []
    all_rows = existing + rows
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in all_rows:
        key = (
            str(row["source_id"]),
            str(row["source_type"]),
            str(row["target_id"]),
            str(row["target_type"]),
            str(row["relation_type"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    pd.DataFrame(unique).to_parquet(path)


def _write_potential_gamma_prime_pairs(root_path: Path, potential_id: str, manifest_dir: Path) -> None:
    shared_path = root_path / "data" / "shared" / "manifests" / "gamma_prime_pairs.parquet"
    relax_path = manifest_dir / "relaxations.parquet"
    relaxed_path = manifest_dir / "relaxed_structures.parquet"
    output_path = manifest_dir / "gamma_prime_pairs.parquet"
    if not shared_path.exists() or not relax_path.exists() or not relaxed_path.exists():
        return

    shared_rows = pd.read_parquet(shared_path).to_dict(orient="records")
    relax_rows = pd.read_parquet(relax_path).to_dict(orient="records")
    relaxed_rows = pd.read_parquet(relaxed_path).to_dict(orient="records")
    initial_to_relax = {
        str(row["initial_structure_id"]): str(row["relaxation_run_id"])
        for row in relax_rows
        if str(row.get("potential_id", "")) == potential_id
    }
    relax_to_relaxed = {
        str(row["relaxation_run_id"]): str(row["relaxed_structure_id"])
        for row in relaxed_rows
    }

    rows: list[dict[str, Any]] = []
    for row in shared_rows:
        gamma_initial = str(row["gamma_initial_structure_id"])
        gamma_prime_initial = str(row["gamma_prime_initial_structure_id"])
        gamma_relaxed = relax_to_relaxed.get(initial_to_relax.get(gamma_initial, ""), "")
        gamma_prime_relaxed = relax_to_relaxed.get(initial_to_relax.get(gamma_prime_initial, ""), "")
        if not gamma_relaxed or not gamma_prime_relaxed:
            continue
        output = dict(row)
        output["gamma_relaxed_structure_id"] = gamma_relaxed
        output["gamma_prime_relaxed_structure_id"] = gamma_prime_relaxed
        rows.append(output)

    if rows:
        pd.DataFrame(rows).sort_values("pair_id").to_parquet(output_path)
    elif output_path.exists():
        output_path.unlink()


def _mtp_type_maps(potential: Mapping[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    raw_type_map = potential.get("type_map")
    if not isinstance(raw_type_map, Mapping) or not raw_type_map:
        raise ValueError(
            f"Potential {potential.get('potential_id', '')} must declare type_map for mtp_mlp relaxation"
        )
    type_to_element = {int(key): str(value) for key, value in raw_type_map.items()}
    element_to_type = {element: type_index for type_index, element in type_to_element.items()}
    missing = sorted(str(element) for element in potential.get("elements", []) if str(element) not in element_to_type)
    if missing:
        raise ValueError(f"type_map is missing declared elements: {missing}")
    return dict(sorted(type_to_element.items())), element_to_type


def _mtp_model_file(root_path: Path, potential: Mapping[str, Any]) -> Path:
    potential_id = str(potential.get("potential_id", ""))
    files_dir = root_path / potential_workspace_paths(potential_id)["potential_dir"] / "files"
    candidates = sorted(
        path
        for path in files_dir.glob("*")
        if path.is_file() and (path.suffix.lower() == ".mtp" or "mtp" in path.name.lower())
    )
    if not candidates:
        raise FileNotFoundError(f"No .mtp model file found under {files_dir}")
    model_file = candidates[0]
    expected = str(potential.get("file_sha256", "")).lower()
    if expected:
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"MTP model hash mismatch for {model_file}: {digest} != {expected}")
    return model_file


def _structure_to_mtp_cfg(
    structure: Structure,
    *,
    element_to_type: Mapping[str, int],
    energy: float,
    stress: Sequence[float],
    efs_by: str,
) -> str:
    lines = [
        "BEGIN_CFG",
        " Size",
        f"{len(structure):6d}",
        " Supercell",
    ]
    for vector in structure.lattice.matrix:
        lines.append("  " + " ".join(f"{float(value):16.10f}" for value in vector))
    lines.append(" AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz")
    for index, site in enumerate(structure, start=1):
        element = str(site.specie.symbol)
        type_index = element_to_type[element]
        x, y, z = (float(value) for value in site.coords)
        lines.append(
            f"{index:14d} {type_index:4d} {x:14.8f} {y:13.8f} {z:13.8f}"
            "     0.000000    0.000000    0.000000"
        )
    stress_values = list(stress)[:6] + [0.0] * max(0, 6 - len(stress))
    lines.extend(
        [
            " Energy",
            f"       {float(energy):.12f}",
            " PlusStress:  xx          yy          zz          yz          xz          xy",
            "  " + " ".join(f"{float(value):10.5f}" for value in stress_values[:6]),
            f" Feature   EFS_by\t{efs_by}",
            "END_CFG",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_mtp_cfg(path: Path, *, type_to_element: Mapping[int, str]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cell: list[list[float]] = []
    species: list[str] = []
    coords: list[list[float]] = []
    forces: list[list[float]] = []
    energy: float | None = None
    stress: list[float] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "Supercell":
            cell = [[float(value) for value in lines[index + offset].split()[:3]] for offset in (1, 2, 3)]
            index += 4
            continue
        if stripped.startswith("AtomData:"):
            index += 1
            while index < len(lines):
                atom_line = lines[index].strip()
                if not atom_line or atom_line.startswith(("Energy", "PlusStress", "Feature", "END_CFG")):
                    break
                parts = atom_line.split()
                if len(parts) >= 8:
                    type_index = int(parts[1])
                    species.append(type_to_element[type_index])
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
                index += 1
            continue
        if stripped == "Energy" and index + 1 < len(lines):
            energy = float(lines[index + 1].strip())
            index += 2
            continue
        if stripped.startswith("PlusStress") and index + 1 < len(lines):
            stress = [float(value) for value in lines[index + 1].split()[:6]]
            index += 2
            continue
        index += 1

    if not cell or not species:
        raise ValueError(f"Could not parse relaxed MTP cfg: {path}")
    structure = Structure(cell, species, coords, coords_are_cartesian=True)
    force_array = np.asarray(forces, dtype=float)
    return {
        "structure": structure,
        "energy": energy,
        "stress": stress,
        "forces": forces,
        "max_force": _max_force_from_array(force_array),
    }


def _mtp_relax_settings(*, fmax: float, steps: int, relax_cell: bool, raw_log_relative: str) -> str:
    stress_tolerance = 0.001 if relax_cell else 0.0
    return "\n".join(
        [
            "### RELAXATION OPTIONS ####",
            "relax:pressure                        0.0",
            f"relax:iteration_limit                 {int(steps)}",
            "relax:mindist                         1.0",
            f"relax:force_tolerance                 {float(fmax)}",
            f"relax:stress_tolerance                {stress_tolerance}",
            "relax:max_step                        0.2",
            "relax:min_step                        1.0e-8",
            "relax:bfgs_wolfe_c1                   1.0e-3",
            "relax:bfgs_wolfe_c2                   0.7",
            f"relax:log                             {raw_log_relative}",
            "relax:freeze_small_grads              FALSE",
            "relax:use_gd_method                   FALSE",
            "relax:correct_cell                    FALSE",
            "relax:init_mindist                    1.0",
            "relax:forces_elasticity_scale         10",
            "relax:stress_elasticity_scale         50",
            "",
        ]
    )


def _resolve_mlp_command(root_path: Path, command: Sequence[str] | None) -> dict[str, Any]:
    if command:
        return {"kind": "direct", "argv": [str(part) for part in command], "display": list(command)}
    command_json = os.environ.get("ALLOY_XRD_MLP_COMMAND_JSON")
    if command_json:
        parsed = json.loads(command_json)
        return {"kind": "direct", "argv": [str(part) for part in parsed], "display": parsed}
    command_text = os.environ.get("ALLOY_XRD_MLP_COMMAND")
    if command_text:
        parsed = shlex.split(command_text, posix=os.name != "nt")
        return {"kind": "direct", "argv": parsed, "display": parsed}
    for name in ("mlp", "mlp.exe"):
        executable = shutil.which(name)
        if executable:
            return {"kind": "direct", "argv": [executable], "display": [executable]}
    wsl_build = root_path / "local_potential_cache" / "source_documents" / "mlip-3-wsl-build-round48" / "bin" / "mlp"
    if wsl_build.exists() and shutil.which("wsl"):
        return {
            "kind": "wsl",
            "argv": [str(wsl_build)],
            "display": ["wsl", "bash", "-lc", str(wsl_build)],
            "wsl_mlp": _windows_path_to_wsl(wsl_build),
        }
    raise FileNotFoundError(
        "No mlp executable found. Set ALLOY_XRD_MLP_COMMAND_JSON or pass --mlp-command-json."
    )


def _run_mlp_command(*, root_path: Path, command: Mapping[str, Any], args: Sequence[str]) -> None:
    if command.get("kind") == "wsl":
        wsl_cwd = _windows_path_to_wsl(root_path)
        wsl_mlp = str(command["wsl_mlp"])
        shell_command = " ".join(shlex.quote(part) for part in [wsl_mlp, *args])
        completed = subprocess.run(
            ["wsl", "bash", "-lc", f"cd {shlex.quote(wsl_cwd)} && {shell_command}"],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    else:
        completed = subprocess.run(
            [*command["argv"], *args],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "mlp command failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(parts)


def _relative_path(root_path: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root_path)).replace("\\", "/")


def _stress_dict_from_sequence(values: Sequence[float]) -> dict[str, float]:
    labels = ("xx", "yy", "zz", "yz", "xz", "xy")
    return {label: float(values[index]) for index, label in enumerate(labels[: len(values)])}


def _parse_mlp_relax_steps(path: Path) -> int:
    if not path.exists():
        return 0
    max_iteration = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace("\t", " ").split()
        for index, part in enumerate(parts[:-1]):
            if part == "itr":
                try:
                    max_iteration = max(max_iteration, int(float(parts[index + 1])))
                except ValueError:
                    pass
    return max_iteration


def _trajectory_fmax(trajectory: Any) -> float:
    forces = getattr(trajectory, "forces", [])
    if not forces:
        return float("inf")
    final_forces = np.asarray(forces[-1], dtype=float)
    if final_forces.size == 0:
        return 0.0
    return float(np.linalg.norm(final_forces, axis=1).max())


def _trajectory_stress(trajectory: Any) -> dict[str, float]:
    stresses = getattr(trajectory, "stresses", [])
    if not stresses:
        return {}
    final = np.asarray(stresses[-1], dtype=float).reshape(-1)
    labels = ("xx", "yy", "zz", "yz", "xz", "xy")
    return {label: float(final[index]) for index, label in enumerate(labels[: len(final)])}


def _max_force_from_array(forces: np.ndarray) -> float:
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces, axis=1).max())


def _write_relax_log(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
