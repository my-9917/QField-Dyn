import csv

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HistoryQuantumTrajectoryDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        internal_split_path,
        partition,
        graph_cache_path,
        coordinate_cache_path,
        quantum_cache_path,
        observed_frames=20,
        prediction_frames=80,
    ):
        with open(manifest_path, encoding="utf-8", newline="") as handle:
            train_ids = {
                row["complex_id"]
                for row in csv.DictReader(handle)
                if row["split"] == "train"
            }
        with open(internal_split_path, encoding="utf-8", newline="") as handle:
            self.ids = [
                row["complex_id"]
                for row in csv.DictReader(handle)
                if row["partition"] == partition
                and row["complex_id"] in train_ids
            ]
        self.graphs = h5py.File(graph_cache_path, "r")
        self.coordinates = h5py.File(coordinate_cache_path, "r")
        self.quantum = h5py.File(quantum_cache_path, "r")
        self.observed_frames = observed_frames
        self.prediction_frames = prediction_frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        complex_id = self.ids[index]
        graph = self.graphs[complex_id]
        frames = self.coordinates[complex_id]["ligand_coordinates"][
            : self.observed_frames + self.prediction_frames
        ]
        quantum = self.quantum[complex_id]
        atom_charge = quantum["atom_charge"][:]
        atom_latent = quantum["atom_latent"][:]
        molecule_properties = quantum["molecule_properties"][:]
        heavy_mask = quantum["heavy_mask"][:]
        quantum_features = np.concatenate(
            (
                atom_latent,
                atom_charge[:, None],
                np.repeat(
                    molecule_properties[None], len(atom_charge), axis=0
                ),
                heavy_mask[:, None],
            ),
            axis=1,
        ).astype(np.float32)
        return {
            "complex_id": complex_id,
            "ligand_atomic_numbers": graph["ligand_atomic_numbers"][:],
            "protein_atomic_numbers": graph["protein_atomic_numbers"][:],
            "observed_coordinates": frames[
                : self.observed_frames
            ].astype(np.float32),
            "target_coordinates": frames[
                self.observed_frames :
            ].astype(np.float32),
            "protein_coordinates": graph["protein_coordinates"][:],
            "ligand_bonds": graph["ligand_bonds"][:],
            "quantum_features": quantum_features,
        }


def collate_history_quantum_trajectory(samples):
    sample = samples[0]
    observed = sample["observed_coordinates"]
    velocity = np.diff(observed, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    centered = observed - observed.mean(axis=0, keepdims=True)
    fluctuation = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=0))
    lag_correlation = np.mean(
        np.sum(velocity[:-1] * velocity[1:], axis=-1), axis=0
    ) / np.mean(np.sum(velocity[:-1] ** 2, axis=-1), axis=0)
    history_features = np.stack(
        (
            speed.mean(axis=0),
            speed.std(axis=0),
            fluctuation,
            lag_correlation,
        ),
        axis=-1,
    ).astype(np.float32)

    bonds = sample["ligand_bonds"]
    directed_bonds = np.concatenate((bonds, bonds[:, ::-1]), axis=0).T
    neighbors = [[] for _ in range(len(observed[0]))]
    for atom_i, atom_j in bonds:
        neighbors[atom_i].append(atom_j)
        neighbors[atom_j].append(atom_i)
    angles = np.asarray(
        [
            (atom_i, center, atom_j)
            for center, atoms in enumerate(neighbors)
            for neighbor_index, atom_i in enumerate(atoms)
            for atom_j in atoms[neighbor_index + 1 :]
        ],
        dtype=np.int64,
    ).T
    return {
        "complex_id": sample["complex_id"],
        "ligand_atomic_numbers": torch.as_tensor(
            sample["ligand_atomic_numbers"], dtype=torch.long
        ),
        "protein_atomic_numbers": torch.as_tensor(
            sample["protein_atomic_numbers"], dtype=torch.long
        ),
        "observed_coordinates": torch.as_tensor(
            observed, dtype=torch.float32
        ),
        "target_coordinates": torch.as_tensor(
            sample["target_coordinates"], dtype=torch.float32
        ),
        "protein_coordinates": torch.as_tensor(
            sample["protein_coordinates"], dtype=torch.float32
        ),
        "bond_index": torch.as_tensor(directed_bonds, dtype=torch.long),
        "angle_index": torch.as_tensor(angles, dtype=torch.long),
        "history_features": torch.as_tensor(
            history_features, dtype=torch.float32
        ),
        "memory_state": torch.as_tensor(
            velocity.mean(axis=0), dtype=torch.float32
        ),
        "quantum_features": torch.as_tensor(
            sample["quantum_features"], dtype=torch.float32
        ),
    }
