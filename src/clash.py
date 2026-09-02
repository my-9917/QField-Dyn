import numpy as np
from rdkit import Chem


_PERIODIC_TABLE = Chem.GetPeriodicTable()
_VDW_RADII = np.asarray(
    [_PERIODIC_TABLE.GetRvdw(atomic_number) for atomic_number in range(119)],
    dtype=np.float64,
)


def build_graph_clash_physics(
    protein_atomic_numbers, ligand_atomic_numbers, ligand_bonds
):
    adjacency = np.zeros(
        (len(ligand_atomic_numbers), len(ligand_atomic_numbers)), dtype=np.int64
    )
    adjacency[ligand_bonds[:, 0], ligand_bonds[:, 1]] = 1
    adjacency[ligand_bonds[:, 1], ligand_bonds[:, 0]] = 1
    within_three = (
        adjacency + adjacency @ adjacency + adjacency @ adjacency @ adjacency
    ) > 0
    ligand_heavy = ligand_atomic_numbers != 1
    atom_i, atom_j = np.triu_indices(len(ligand_atomic_numbers), k=1)
    internal_mask = (
        ligand_heavy[atom_i]
        & ligand_heavy[atom_j]
        & ~within_three[atom_i, atom_j]
    )
    internal_pairs = np.column_stack(
        (atom_i[internal_mask], atom_j[internal_mask])
    )
    ligand_radii = _VDW_RADII[ligand_atomic_numbers]
    internal_threshold = 0.75 * (
        ligand_radii[internal_pairs[:, 0]] + ligand_radii[internal_pairs[:, 1]]
    )
    protein_heavy = protein_atomic_numbers != 1
    protein_radii = _VDW_RADII[protein_atomic_numbers[protein_heavy]]
    ligand_heavy_indices = np.flatnonzero(ligand_heavy)
    protein_ligand_threshold = 0.75 * (
        protein_radii[:, None] + ligand_radii[ligand_heavy_indices][None]
    )
    return {
        "protein_heavy": protein_heavy,
        "internal_pairs": internal_pairs,
        "internal_threshold": internal_threshold,
        "ligand_heavy_indices": ligand_heavy_indices,
        "protein_ligand_threshold": protein_ligand_threshold,
    }


def clash_counts(frames, fixed_protein, physics):
    internal_pairs = physics["internal_pairs"]
    internal_distance = np.linalg.norm(
        frames[:, internal_pairs[:, 0]] - frames[:, internal_pairs[:, 1]],
        axis=-1,
    )
    internal = np.sum(
        internal_distance < physics["internal_threshold"], axis=1
    )
    ligand_heavy = frames[:, physics["ligand_heavy_indices"]]
    threshold = physics["protein_ligand_threshold"]
    max_threshold = float(np.max(threshold))
    protein = []
    for ligand_frame in ligand_heavy:
        lower = np.min(ligand_frame, axis=0) - max_threshold
        upper = np.max(ligand_frame, axis=0) + max_threshold
        candidate_mask = np.all(
            (fixed_protein >= lower) & (fixed_protein <= upper), axis=1
        )
        distance = np.linalg.norm(
            fixed_protein[candidate_mask, None] - ligand_frame[None], axis=-1
        )
        protein.append(np.sum(distance < threshold[candidate_mask]))
    return internal, np.asarray(protein, dtype=np.int64)
