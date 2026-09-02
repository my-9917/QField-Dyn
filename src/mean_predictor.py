import csv
from pathlib import Path

import numpy as np

from .clash import build_graph_clash_physics, clash_counts
from .dual_expert import DualExpertPredictor
from .equivariant_residual import apply_patch_residual
from .torsion_geometry import graph_data, kinematic_tree
from .torsion_projection import project_to_torsion_manifold


class MeanTrajectoryPredictor:
    def __init__(
        self,
        qe_checkpoint,
        qe_statistics,
        trajectory_checkpoint,
        gate_checkpoint,
        geometry_coefficients,
        max_rms_displacement,
        device="cuda",
    ):
        self.dual_expert = DualExpertPredictor(
            qe_checkpoint,
            qe_statistics,
            trajectory_checkpoint,
            gate_checkpoint,
            device,
        )
        with Path(geometry_coefficients).open(encoding="utf-8", newline="") as handle:
            internal_coefficients = np.asarray(
                [
                    [float(value) for value in list(row.values())[1:]]
                    for row in csv.DictReader(handle)
                ]
            )
        self.coefficients = np.zeros((4, 4), dtype=np.float64)
        self.coefficients[:, 2:] = internal_coefficients
        self.max_rms_displacement = float(max_rms_displacement)

    def predict(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
    ):
        protein_atomic_numbers = np.asarray(
            protein_atomic_numbers, dtype=np.int64
        )
        protein_coordinates = np.asarray(protein_coordinates, dtype=np.float32)
        ligand_atomic_numbers = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        ligand_bonds = np.asarray(ligand_bonds, dtype=np.int64)
        observed = np.asarray(observed_ligand_coordinates, dtype=np.float32)
        mean_prediction, weights, q_prediction = self.dual_expert.predict(
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed,
            return_q_prediction=True,
        )
        previous, current = observed[18:20]
        heavy = ligand_atomic_numbers != 1
        target = apply_patch_residual(
            previous,
            current,
            q_prediction,
            heavy,
            self.coefficients,
        )
        rotatable, fragment, fragments = graph_data(
            len(ligand_atomic_numbers), ligand_bonds
        )
        _, order, subtree_atoms = kinematic_tree(
            current, rotatable, fragment, fragments
        )
        raw = np.empty_like(mean_prediction, dtype=np.float64)
        for frame_index, frame in enumerate(mean_prediction):
            raw[frame_index], _ = project_to_torsion_manifold(
                frame,
                target[frame_index],
                heavy,
                order,
                subtree_atoms,
            )
        raw = raw.astype(np.float32)

        physics = build_graph_clash_physics(
            protein_atomic_numbers, ligand_atomic_numbers, ligand_bonds
        )
        fixed_protein = protein_coordinates[physics["protein_heavy"]]
        baseline_internal, baseline_protein = clash_counts(
            mean_prediction, fixed_protein, physics
        )
        candidate_internal, candidate_protein = clash_counts(
            raw, fixed_protein, physics
        )
        displacement = np.sqrt(
            np.mean(np.sum((raw - mean_prediction) ** 2, axis=-1), axis=-1)
        )
        accept = (
            (candidate_internal <= baseline_internal)
            & (candidate_protein <= baseline_protein)
            & (displacement <= self.max_rms_displacement)
        )
        prediction = np.where(
            accept[:, None, None], raw, mean_prediction
        ).astype(np.float32)
        return prediction, weights
