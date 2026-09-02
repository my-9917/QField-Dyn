import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.guesser.default_guesser import DefaultGuesser
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
from rdkit import Chem


def _atomic_numbers(atoms):
    periodic_table = Chem.GetPeriodicTable()
    element_guesser = DefaultGuesser(None)
    numbers = []
    for atom in atoms:
        symbol = atom.element.strip() or element_guesser.guess_atom_element(
            atom.name
        )
        numbers.append(periodic_table.GetAtomicNumber(symbol.capitalize()))
    return np.asarray(numbers, dtype=np.int64)


def rigid_fit(moving, reference):
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (moving - moving_center).T @ (reference - reference_center)
    )
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt, moving_center, reference_center


def _read_xtc(path):
    with XTCFile(str(path)) as trajectory:
        frames = [trajectory.read() for _ in range(len(trajectory))]
    coordinates = np.stack([frame.x.copy() for frame in frames]) * 10.0
    boxes = np.stack([frame.box.copy() for frame in frames]) * 10.0
    times = np.asarray([frame.time for frame in frames])
    return coordinates, boxes, times


def _correct_ligand_observations(
    coordinates,
    protein_indices,
    protein_heavy_indices,
    ligand_indices,
    reference_coordinates,
):
    reference_protein = reference_coordinates[protein_heavy_indices]
    protein_heavy_local = np.flatnonzero(
        np.isin(protein_indices, protein_heavy_indices)
    )
    corrected = []
    for frame in coordinates:
        raw_protein = frame[protein_indices]
        rotation, moving_center, reference_center = rigid_fit(
            raw_protein[protein_heavy_local], reference_protein
        )
        ligand = (frame[ligand_indices] - moving_center) @ rotation + reference_center
        corrected.append(ligand)
    return np.asarray(corrected, dtype=np.float32)


def load_public_system(data_directory, tier, complex_id, pocket_cutoff=10.0):
    system_directory = Path(data_directory) / tier / complex_id
    meta = json.loads(
        (system_directory / "meta.json").read_text(encoding="utf-8")
    )
    topology_path = system_directory / f"{complex_id}.pdb"
    observation_path = system_directory / f"{complex_id}_obs.xtc"
    topology = mda.Universe(str(topology_path))
    coordinates, boxes, times = _read_xtc(observation_path)
    if coordinates.shape != (meta["n_obs"], meta["n_atoms"], 3):
        raise ValueError(f"{complex_id}: observation shape")
    if not np.isclose(times[1] - times[0], meta["dt_ps"]):
        raise ValueError(f"{complex_id}: timestep")

    ligand = topology.select_atoms(f"resname {meta['ligand_resname']}")
    protein = topology.select_atoms("protein")
    environment = topology.select_atoms(
        f"not protein and not resname {meta['ligand_resname']}"
    )
    atomic_numbers = _atomic_numbers(topology.atoms)
    protein_heavy_indices = protein.indices[
        atomic_numbers[protein.indices] != 1
    ]
    ligand_atomic_numbers = atomic_numbers[ligand.indices]
    ligand_heavy_local = ligand_atomic_numbers != 1
    reference = coordinates[-1].copy()
    observed_ligand = _correct_ligand_observations(
        coordinates,
        protein.indices,
        protein_heavy_indices,
        ligand.indices,
        reference,
    )

    global_to_ligand = {
        int(global_index): local_index
        for local_index, global_index in enumerate(ligand.indices)
    }
    ligand_bonds = np.asarray(
        sorted(
            {
                tuple(
                    sorted(
                        (
                            global_to_ligand[int(atom_i)],
                            global_to_ligand[int(atom_j)],
                        )
                    )
                )
                for atom_i, atom_j in topology.bonds.indices
                if int(atom_i) in global_to_ligand
                and int(atom_j) in global_to_ligand
            }
        ),
        dtype=np.int64,
    )
    ligand_reference = observed_ligand[-1]
    protein_reference = reference[protein_heavy_indices]
    squared_distance = np.sum(
        (
            protein_reference[:, None]
            - ligand_reference[ligand_heavy_local][None]
        )
        ** 2,
        axis=-1,
    )
    pocket_mask = np.min(squared_distance, axis=1) <= pocket_cutoff**2
    pocket_indices = protein_heavy_indices[pocket_mask]
    return {
        "meta": meta,
        "reference_coordinates": reference,
        "reference_box_angstrom": boxes[0],
        "ligand_indices": ligand.indices.copy(),
        "protein_atomic_numbers": atomic_numbers[protein_heavy_indices],
        "protein_coordinates": reference[protein_heavy_indices],
        "fixed_environment_coordinates": reference[environment.indices],
        "pocket_atomic_numbers": atomic_numbers[pocket_indices],
        "pocket_coordinates": reference[pocket_indices],
        "ligand_atomic_numbers": ligand_atomic_numbers,
        "ligand_bonds": ligand_bonds,
        "ligand_structure_coordinates": ligand_reference,
        "observed_ligand_coordinates": observed_ligand,
    }


def write_public_prediction(system, ligand_prediction, output):
    meta = system["meta"]
    if ligand_prediction.shape != (
        meta["n_pred"],
        len(system["ligand_indices"]),
        3,
    ):
        raise ValueError(f"{meta['id']}: prediction shape")
    if not np.isfinite(ligand_prediction).all():
        raise FloatingPointError(f"{meta['id']}: non-finite prediction")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = system["reference_coordinates"].copy()
    with XTCFile(str(output), "w") as trajectory:
        for prediction_index, ligand_coordinates in enumerate(ligand_prediction):
            frame[system["ligand_indices"]] = ligand_coordinates
            trajectory.write(
                frame / 10.0,
                system["reference_box_angstrom"] / 10.0,
                meta["n_obs"] + prediction_index,
                (meta["n_obs"] + prediction_index) * meta["dt_ps"],
                1000.0,
            )
