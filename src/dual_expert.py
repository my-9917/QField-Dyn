from pathlib import Path

import numpy as np
import torch

from .expert_gate import ExpertGate
from .quantum_predictor import QuantumTrajectoryPredictor


class DualExpertPredictor:
    def __init__(
        self,
        qe_checkpoint,
        qe_statistics,
        trajectory_checkpoint,
        gate_checkpoint,
        device="cuda",
    ):
        self.quantum_predictor = QuantumTrajectoryPredictor(
            qe_checkpoint,
            qe_statistics,
            trajectory_checkpoint,
            device,
        )
        self.device = self.quantum_predictor.device
        checkpoint = torch.load(
            Path(gate_checkpoint), map_location=self.device, weights_only=False
        )
        self.gate = ExpertGate(context_dim=1033, hidden_dim=128).to(
            self.device
        )
        self.gate.load_state_dict(checkpoint["model_state_dict"])
        self.gate.eval()
        self.context_mean = torch.as_tensor(
            checkpoint["context_mean"], dtype=torch.float32, device=self.device
        )
        self.context_std = torch.as_tensor(
            checkpoint["context_std"], dtype=torch.float32, device=self.device
        )
        self.segment_index = torch.as_tensor(
            [0] * 10 + [1] * 10 + [2] * 20 + [3] * 40,
            dtype=torch.long,
            device=self.device,
        )
        horizon = np.arange(1, 81, dtype=np.float32)
        self.damping = (1 - 0.25**horizon) / (1 - 0.25)

    def predict(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
        return_q_prediction=False,
    ):
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
        heavy = ligand_atomic_numbers != 1

        quantum = self.quantum_predictor.quantum_features(
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure,
        )
        q_prediction = self.quantum_predictor.predict_from_quantum(
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure,
            observed,
            "T3",
            quantum["dynamics_features"],
        )

        previous = observed[18]
        current = observed[19]
        heavy_previous = previous[heavy]
        heavy_current = current[heavy]
        centered = heavy_current - heavy_current.mean(axis=0)
        protein_heavy_coordinates = protein_coordinates[
            protein_atomic_numbers != 1
        ]
        layers = quantum["atom_latent_layers"][heavy]
        charge = quantum["atom_charge"][heavy]
        context = np.concatenate(
            (
                layers.mean(axis=0).reshape(-1),
                layers.std(axis=0).reshape(-1),
                np.asarray(
                    [
                        np.log1p(heavy.sum()),
                        charge.mean(),
                        charge.std(),
                        *quantum["molecule_properties"],
                        np.linalg.norm(
                            heavy_current.mean(axis=0)
                            - heavy_previous.mean(axis=0)
                        ),
                        np.sqrt(np.mean((heavy_current - heavy_previous) ** 2)),
                        np.sqrt(np.mean(np.sum(centered**2, axis=1))),
                        np.min(
                            np.linalg.norm(
                                heavy_current[:, None]
                                - protein_heavy_coordinates[None],
                                axis=-1,
                            )
                        ),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)

        with torch.no_grad():
            segment_weights = self.gate(
                (
                    torch.as_tensor(context, device=self.device)
                    - self.context_mean
                )
                / self.context_std
            )[0]
            frame_weights = segment_weights[self.segment_index]
            centroid_velocity = (
                heavy_current.mean(axis=0) - heavy_previous.mean(axis=0)
            )
            o_prediction = (
                current[None]
                - 0.20
                * self.damping[:, None, None]
                * centroid_velocity[None, None]
            )
            prediction = (
                frame_weights[:, 0, None, None]
                * torch.as_tensor(q_prediction, device=self.device)
                + frame_weights[:, 1, None, None]
                * torch.as_tensor(o_prediction, device=self.device)
            )
        result = (
            prediction.cpu().numpy().astype(np.float32),
            segment_weights.cpu().numpy().astype(np.float32),
        )
        if return_q_prediction:
            return (*result, q_prediction.astype(np.float32))
        return result
