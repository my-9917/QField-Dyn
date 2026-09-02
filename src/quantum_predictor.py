from pathlib import Path

import h5py
import numpy as np
import torch

from .quantum_features import QuantumFeatureEncoder
from .trajectory_model import QuantumTrajectoryModel


TASKS = {
    "T1": {"current": 9, "steps": 10},
    "T2": {"current": 79, "steps": 20},
    "T3": {"current": 19, "steps": 80},
}


class QuantumTrajectoryPredictor:
    def __init__(
        self,
        qe_checkpoint,
        qe_statistics,
        trajectory_checkpoint,
        device="cuda",
    ):
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(device)

        qe_state = torch.load(
            Path(qe_checkpoint), map_location=self.device, weights_only=False
        )
        self.qe_model = QuantumFeatureEncoder(hidden_dim=128, layer_count=3).to(
            self.device
        )
        self.qe_model.load_state_dict(qe_state["model_state_dict"])
        self.qe_model.eval()

        with h5py.File(qe_statistics, "r") as statistics:
            self.atom_mean = float(statistics.attrs["atom_mean"])
            self.atom_std = float(statistics.attrs["atom_std"])
            self.molecule_mean = np.asarray(
                statistics.attrs["molecule_mean"], dtype=np.float32
            )
            self.molecule_std = np.asarray(
                statistics.attrs["molecule_std"], dtype=np.float32
            )

        dynamics_state = torch.load(
            Path(trajectory_checkpoint), map_location=self.device, weights_only=False
        )
        self.dynamics = QuantumTrajectoryModel(
            hidden_dim=128,
            layer_count=3,
            quantum_feature_dim=132,
        ).to(self.device)
        self.dynamics.load_state_dict(dynamics_state["model_state_dict"])
        self.dynamics.eval()

    def quantum_features(
        self,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
    ):
        atomic_numbers = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        bonds = np.asarray(ligand_bonds, dtype=np.int64)
        structure = np.asarray(ligand_structure_coordinates, dtype=np.float32)
        heavy = atomic_numbers != 1
        heavy_indices = np.flatnonzero(heavy)
        old_to_heavy = np.full(len(atomic_numbers), -1, dtype=np.int64)
        old_to_heavy[heavy_indices] = np.arange(len(heavy_indices))
        heavy_bonds = old_to_heavy[bonds[np.all(heavy[bonds], axis=1)]]
        edge_index = np.concatenate(
            (heavy_bonds, heavy_bonds[:, ::-1]), axis=0
        ).T

        with torch.no_grad():
            atom_prediction, molecule_prediction, layers = (
                self.qe_model.forward_layers(
                    torch.as_tensor(
                        atomic_numbers[heavy], dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        structure[heavy], dtype=torch.float32, device=self.device
                    ),
                    torch.as_tensor(
                        edge_index, dtype=torch.long, device=self.device
                    ),
                    torch.zeros(
                        int(heavy.sum()), dtype=torch.long, device=self.device
                    ),
                )
            )

        atom_charge = np.zeros(len(atomic_numbers), dtype=np.float32)
        atom_charge[heavy] = (
            atom_prediction.cpu().numpy() * self.atom_std + self.atom_mean
        )
        atom_layers = np.zeros(
            (len(atomic_numbers), layers.shape[1], layers.shape[2]),
            dtype=np.float32,
        )
        atom_layers[heavy] = layers.cpu().numpy()
        molecule_properties = (
            molecule_prediction[0].cpu().numpy() * self.molecule_std
            + self.molecule_mean
        )
        dynamics_features = np.concatenate(
            (
                atom_layers[:, -1],
                atom_charge[:, None],
                np.repeat(
                    molecule_properties[None], len(atomic_numbers), axis=0
                ),
                heavy[:, None],
            ),
            axis=1,
        ).astype(np.float32)
        return {
            "atom_charge": atom_charge,
            "atom_latent_layers": atom_layers,
            "molecule_properties": molecule_properties,
            "heavy_mask": heavy,
            "dynamics_features": dynamics_features,
        }

    def predict(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
        task,
    ):
        quantum = self.quantum_features(
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
        )["dynamics_features"]
        return self.predict_from_quantum(
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed_ligand_coordinates,
            task,
            quantum,
        )

    def predict_from_quantum(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
        task,
        quantum_dynamics_features,
    ):
        task_spec = TASKS[task]
        protein_atomic_numbers = np.asarray(
            protein_atomic_numbers, dtype=np.int64
        )
        protein_coordinates = np.asarray(
            protein_coordinates, dtype=np.float32
        )
        ligand_atomic_numbers = np.asarray(
            ligand_atomic_numbers, dtype=np.int64
        )
        ligand_bonds = np.asarray(ligand_bonds, dtype=np.int64)
        ligand_structure = np.asarray(
            ligand_structure_coordinates, dtype=np.float32
        )
        observed = np.asarray(observed_ligand_coordinates, dtype=np.float32)

        protein_heavy = protein_atomic_numbers != 1
        ligand_heavy = ligand_atomic_numbers != 1
        squared = np.sum(
            (
                protein_coordinates[protein_heavy, None]
                - ligand_structure[ligand_heavy][None]
            )
            ** 2,
            axis=-1,
        )
        pocket_mask = np.min(squared, axis=1) <= 100.0
        pocket_z = protein_atomic_numbers[protein_heavy][pocket_mask]
        pocket_x = protein_coordinates[protein_heavy][pocket_mask]
        quantum = np.asarray(quantum_dynamics_features, dtype=np.float32)

        ligand_z = torch.as_tensor(
            ligand_atomic_numbers, dtype=torch.long, device=self.device
        )
        protein_z = torch.as_tensor(
            pocket_z, dtype=torch.long, device=self.device
        )
        protein_x = torch.as_tensor(
            pocket_x, dtype=torch.float32, device=self.device
        )
        directed_bonds = np.concatenate(
            (ligand_bonds, ligand_bonds[:, ::-1]), axis=0
        ).T
        bond_index = torch.as_tensor(
            directed_bonds, dtype=torch.long, device=self.device
        )
        previous = torch.as_tensor(
            observed[task_spec["current"] - 1],
            dtype=torch.float32,
            device=self.device,
        )
        current = torch.as_tensor(
            observed[task_spec["current"]],
            dtype=torch.float32,
            device=self.device,
        )
        velocity = current - previous
        quantum = torch.as_tensor(
            quantum, dtype=torch.float32, device=self.device
        )
        predictions = []
        with torch.no_grad():
            for _ in range(task_spec["steps"]):
                cross = torch.nonzero(
                    torch.cdist(current, protein_x) <= 6.0, as_tuple=False
                ).T
                acceleration = self.dynamics(
                    ligand_z,
                    protein_z,
                    current,
                    protein_x,
                    velocity,
                    bond_index,
                    cross,
                    quantum,
                )
                next_coordinates = current + velocity + acceleration
                predictions.append(next_coordinates)
                velocity, current = next_coordinates - current, next_coordinates
        return torch.stack(predictions).cpu().numpy().astype(np.float32)
