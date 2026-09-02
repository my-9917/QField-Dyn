from pathlib import Path

import numpy as np
import torch

from .history_trajectory_model import HistoryQuantumTrajectoryModel
from .quantum_predictor import QuantumTrajectoryPredictor


class HistoryQuantumTrajectoryPredictor:
    def __init__(self, project_directory, checkpoint, device="cuda"):
        project_directory = Path(project_directory)
        artifacts = project_directory / "artifacts"
        self.device = torch.device(device)
        self.quantum_encoder = QuantumTrajectoryPredictor(
            artifacts / "quantum_encoder.pt",
            artifacts / "quantum_statistics.hdf5",
            artifacts / "trajectory_model.pt",
            device,
        )
        state = torch.load(
            Path(checkpoint), map_location=self.device, weights_only=False
        )
        self.dynamics = HistoryQuantumTrajectoryModel().to(self.device)
        self.dynamics.load_state_dict(state["model_state_dict"])
        self.dynamics.eval()

    def predict(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
        steps=80,
    ):
        protein_z = torch.as_tensor(
            protein_atomic_numbers, dtype=torch.long, device=self.device
        )
        protein_x = torch.as_tensor(
            protein_coordinates, dtype=torch.float32, device=self.device
        )
        ligand_z = torch.as_tensor(
            ligand_atomic_numbers, dtype=torch.long, device=self.device
        )
        bonds = np.asarray(ligand_bonds, dtype=np.int64)
        bond_index = torch.as_tensor(
            np.concatenate((bonds, bonds[:, ::-1]), axis=0).T,
            dtype=torch.long,
            device=self.device,
        )
        observed = np.asarray(observed_ligand_coordinates, dtype=np.float32)
        observed_tensor = torch.as_tensor(
            observed, dtype=torch.float32, device=self.device
        )
        velocity_history = np.diff(observed, axis=0)
        speed = np.linalg.norm(velocity_history, axis=-1)
        centered = observed - observed.mean(axis=0, keepdims=True)
        fluctuation = np.sqrt(
            np.mean(np.sum(centered**2, axis=-1), axis=0)
        )
        lag_correlation = np.mean(
            np.sum(
                velocity_history[:-1] * velocity_history[1:], axis=-1
            ),
            axis=0,
        ) / np.mean(
            np.sum(velocity_history[:-1] ** 2, axis=-1), axis=0
        )
        history_features = torch.as_tensor(
            np.stack(
                (
                    speed.mean(axis=0),
                    speed.std(axis=0),
                    fluctuation,
                    lag_correlation,
                ),
                axis=-1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        memory = torch.as_tensor(
            velocity_history.mean(axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        quantum = self.quantum_encoder.quantum_features(
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
        )["dynamics_features"]
        quantum = torch.as_tensor(
            quantum, dtype=torch.float32, device=self.device
        )

        current = observed_tensor[-1]
        velocity = observed_tensor[-1] - observed_tensor[-2]
        predictions = []
        with torch.no_grad():
            for _ in range(steps):
                cross_index = torch.nonzero(
                    torch.cdist(current, protein_x) <= 6.0,
                    as_tuple=False,
                ).T
                acceleration, memory_decay = self.dynamics(
                    ligand_z,
                    protein_z,
                    current,
                    protein_x,
                    velocity,
                    memory,
                    history_features,
                    bond_index,
                    cross_index,
                    quantum,
                )
                next_coordinates = current + velocity + acceleration
                next_velocity = next_coordinates - current
                predictions.append(next_coordinates)
                memory = (
                    memory_decay * memory
                    + (1.0 - memory_decay) * next_velocity
                )
                velocity = next_velocity
                current = next_coordinates
        return torch.stack(predictions).cpu().numpy().astype(np.float32)
