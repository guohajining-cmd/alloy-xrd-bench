"""Generate shared initial structures for Alloy-XRD-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml
from pymatgen.core import Lattice, Structure

from .contract import canonical_structure_hash


BASIS_BY_PHASE: dict[str, tuple[tuple[float, float, float], ...]] = {
    "fcc": (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
    "l12": (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
    "bcc": (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ),
}


def generate_initial_structures(
    *,
    root: str | Path,
    recipes_path: str | Path,
) -> list[dict[str, Any]]:
    """Generate initial structure artifacts and manifest rows from recipes."""

    root_path = Path(root).resolve()
    recipes = _load_recipes(Path(recipes_path))
    structures_dir = root_path / "data" / "shared" / "structures_initial"
    manifests_dir = root_path / "data" / "shared" / "manifests"
    structures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for recipe in recipes:
        structure, generation_score, candidate_count = _structure_from_recipe(recipe)
        structure_doc = structure.as_dict()
        structure_hash = canonical_structure_hash(structure_doc)
        initial_structure_id = _initial_structure_id(recipe)
        artifact_relative = f"data/shared/structures_initial/{initial_structure_id}.json"
        artifact_path = root_path / artifact_relative
        artifact_bytes = json.dumps(
            structure_doc,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        artifact_path.write_bytes(artifact_bytes)

        rows.append(
            {
                "initial_structure_id": initial_structure_id,
                "alloy_system": str(recipe["alloy_system"]),
                "phase": str(recipe["phase"]).lower(),
                "composition": dict(recipe["composition"]),
                "supercell": _supercell_label(recipe["supercell"]),
                "generation_method": str(
                    recipe.get("generation_method", "deterministic_random_occupancy")
                ),
                "generation_score": generation_score,
                "candidate_count": candidate_count,
                "seed": int(recipe["seed"]),
                "structure_family": str(recipe.get("structure_family", "")),
                "phase_role": str(recipe.get("phase_role", "")),
                "pair_id": str(recipe.get("pair_id", "")),
                "nominal_alloy_system": str(recipe.get("nominal_alloy_system", recipe["alloy_system"])),
                "partitioning_rule": str(recipe.get("partitioning_rule", "")),
                "structure_sha256": structure_hash,
                "artifact_file_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "artifact_path": artifact_relative,
            }
        )

    rows.sort(key=lambda row: row["initial_structure_id"])
    pd.DataFrame(rows).to_parquet(manifests_dir / "initial_structures.parquet")
    _write_gamma_prime_pairs(rows, manifests_dir)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Alloy-XRD-Bench structure manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-initial", help="Generate shared initial structures")
    generate.add_argument("--root", default=".", help="Benchmark workspace root")
    generate.add_argument("--recipes", required=True, help="Recipe YAML path")

    args = parser.parse_args(argv)
    if args.command == "generate-initial":
        rows = generate_initial_structures(root=args.root, recipes_path=args.recipes)
        print(f"generated_initial_structures: {len(rows)}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _load_recipes(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    recipes = payload.get("recipes", [])
    if not isinstance(recipes, list):
        raise TypeError("recipes YAML must contain a list under 'recipes'")
    return [dict(recipe) for recipe in recipes]


def _structure_from_recipe(recipe: Mapping[str, Any]) -> tuple[Structure, float, int]:
    phase = str(recipe["phase"]).lower()
    if phase not in BASIS_BY_PHASE:
        raise ValueError(f"Unsupported phase {phase}; expected one of {sorted(BASIS_BY_PHASE)}")
    supercell = tuple(int(value) for value in recipe["supercell"])
    if len(supercell) != 3 or any(value <= 0 for value in supercell):
        raise ValueError("supercell must contain three positive integers")

    basis = BASIS_BY_PHASE[phase]
    site_count = len(basis) * supercell[0] * supercell[1] * supercell[2]
    coords = _supercell_coords(basis, supercell)
    lattice_a = float(recipe["lattice_a"])
    lattice = Lattice.from_parameters(
        lattice_a * supercell[0],
        lattice_a * supercell[1],
        lattice_a * supercell[2],
        90,
        90,
        90,
    )
    method = str(recipe.get("generation_method", "deterministic_random_occupancy"))
    candidate_count = int(recipe.get("candidate_count", 32 if method == "pair_balance_occupancy" else 1))
    if method == "l12_sublattice_occupancy":
        if phase != "l12":
            raise ValueError("l12_sublattice_occupancy requires phase: l12")
        structure = _l12_sublattice_structure(
            lattice=lattice,
            coords=coords,
            supercell=supercell,
            recipe=recipe,
        )
        return structure, 0.0, 1
    if method == "pair_balance_occupancy":
        structure, score = _best_pair_balance_structure(
            lattice=lattice,
            coords=coords,
            composition=recipe["composition"],
            site_count=site_count,
            seed=int(recipe["seed"]),
            candidate_count=candidate_count,
        )
        return structure, score, candidate_count

    species = _species_sequence(
        composition=recipe["composition"],
        site_count=site_count,
        seed=int(recipe["seed"]),
    )
    return Structure(lattice, species, coords), 0.0, candidate_count


def _l12_sublattice_structure(
    *,
    lattice: Lattice,
    coords: Sequence[tuple[float, float, float]],
    supercell: tuple[int, int, int],
    recipe: Mapping[str, Any],
) -> Structure:
    sublattice_composition = recipe.get("sublattice_composition", {})
    if not isinstance(sublattice_composition, Mapping):
        raise ValueError("l12_sublattice_occupancy requires sublattice_composition")
    corner_composition = sublattice_composition.get("corner")
    face_composition = sublattice_composition.get("face")
    if not isinstance(corner_composition, Mapping) or not isinstance(face_composition, Mapping):
        raise ValueError("sublattice_composition must contain corner and face mappings")

    cell_count = supercell[0] * supercell[1] * supercell[2]
    corner_species = _species_sequence(
        composition=corner_composition,
        site_count=cell_count,
        seed=int(recipe["seed"]),
    )
    face_species = _species_sequence(
        composition=face_composition,
        site_count=3 * cell_count,
        seed=int(recipe["seed"]) + 104729,
    )

    species: list[str] = []
    corner_index = 0
    face_index = 0
    for basis_index in range(len(coords)):
        if basis_index % 4 == 0:
            species.append(corner_species[corner_index])
            corner_index += 1
        else:
            species.append(face_species[face_index])
            face_index += 1
    return Structure(lattice, species, coords)


def _supercell_coords(
    basis: Sequence[tuple[float, float, float]],
    supercell: tuple[int, int, int],
) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    nx, ny, nz = supercell
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for x, y, z in basis:
                    coords.append(((i + x) / nx, (j + y) / ny, (k + z) / nz))
    return coords


def _species_sequence(
    *,
    composition: Mapping[str, float],
    site_count: int,
    seed: int,
) -> list[str]:
    weights = {element: float(amount) for element, amount in composition.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("composition weights must sum to a positive value")

    raw_counts = {element: value / total * site_count for element, value in weights.items()}
    counts = {element: int(raw_counts[element]) for element in raw_counts}
    remainder = site_count - sum(counts.values())
    ranked = sorted(
        raw_counts,
        key=lambda element: (raw_counts[element] - counts[element], element),
        reverse=True,
    )
    for element in ranked[:remainder]:
        counts[element] += 1

    species = [
        element
        for element in sorted(counts)
        for _ in range(counts[element])
    ]
    random.Random(seed).shuffle(species)
    return species


def _best_pair_balance_structure(
    *,
    lattice: Lattice,
    coords: Sequence[tuple[float, float, float]],
    composition: Mapping[str, float],
    site_count: int,
    seed: int,
    candidate_count: int,
) -> tuple[Structure, float]:
    best_structure: Structure | None = None
    best_score = float("inf")
    for offset in range(max(1, candidate_count)):
        species = _species_sequence(
            composition=composition,
            site_count=site_count,
            seed=seed + offset * 7919,
        )
        structure = Structure(lattice, species, coords)
        score = _pair_balance_score(structure, composition)
        if score < best_score:
            best_score = score
            best_structure = structure
    if best_structure is None:
        raise ValueError("No candidate structures generated")
    return best_structure, best_score


def _pair_balance_score(
    structure: Structure,
    composition: Mapping[str, float],
    shell_count: int = 2,
) -> float:
    distance_matrix = structure.distance_matrix
    distances = sorted(
        {
            round(float(distance_matrix[i, j]), 6)
            for i in range(len(structure))
            for j in range(i + 1, len(structure))
            if distance_matrix[i, j] > 1e-8
        }
    )
    shells = distances[:shell_count]
    if not shells:
        return 0.0

    elements = sorted(str(element) for element in composition)
    total_weight = sum(float(composition[element]) for element in composition)
    fractions = {str(element): float(composition[element]) / total_weight for element in composition}
    expected = {}
    for i, left in enumerate(elements):
        for right in elements[i:]:
            expected[(left, right)] = (
                fractions[left] * fractions[right]
                if left == right
                else 2.0 * fractions[left] * fractions[right]
            )

    species = [str(site.specie) for site in structure]
    score = 0.0
    for shell_distance in shells:
        counts = {pair: 0 for pair in expected}
        total_pairs = 0
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                if abs(float(distance_matrix[i, j]) - shell_distance) > 1e-5:
                    continue
                pair = tuple(sorted((species[i], species[j])))
                counts[pair] = counts.get(pair, 0) + 1
                total_pairs += 1
        if total_pairs == 0:
            continue
        for pair, target in expected.items():
            observed = counts.get(pair, 0) / total_pairs
            score += abs(observed - target)
    return float(score)


def _initial_structure_id(recipe: Mapping[str, Any]) -> str:
    return (
        f"init_{recipe['alloy_system']}_{str(recipe['phase']).lower()}_"
        f"{_composition_label(recipe['composition'])}_seed{int(recipe['seed']):06d}"
    )


def _composition_label(composition: Mapping[str, Any]) -> str:
    return "".join(f"{element}{composition[element]}" for element in sorted(composition))


def _supercell_label(supercell: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in supercell)


def _write_gamma_prime_pairs(rows: Sequence[Mapping[str, Any]], manifests_dir: Path) -> None:
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            continue
        item = pairs.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "gamma_initial_structure_id": "",
                "gamma_prime_initial_structure_id": "",
                "gamma_relaxed_structure_id": "",
                "gamma_prime_relaxed_structure_id": "",
                "nominal_alloy_system": str(row.get("nominal_alloy_system", "")),
                "partitioning_rule": str(row.get("partitioning_rule", "")),
                "misfit_definition": "a_gamma_prime_minus_a_gamma_over_a_gamma",
            },
        )
        role = str(row.get("phase_role", ""))
        if role == "gamma":
            item["gamma_initial_structure_id"] = str(row["initial_structure_id"])
        elif role == "gamma_prime":
            item["gamma_prime_initial_structure_id"] = str(row["initial_structure_id"])

    complete_rows = [
        row
        for row in pairs.values()
        if row["gamma_initial_structure_id"] and row["gamma_prime_initial_structure_id"]
    ]
    path = manifests_dir / "gamma_prime_pairs.parquet"
    if complete_rows:
        pd.DataFrame(sorted(complete_rows, key=lambda row: row["pair_id"])).to_parquet(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
