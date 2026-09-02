import numpy as np
import torch

from .trajectory_data import TrajectoryRolloutDataset, collate_trajectory_rollout


class QuantumTrajectoryDataset(TrajectoryRolloutDataset):
    def __init__(self, *args, quantum_cache_path, **kwargs):
        super().__init__(*args, **kwargs)
        import h5py

        self.quantum_cache = h5py.File(quantum_cache_path, "r")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        quantum = self.quantum_cache[sample["complex_id"]]
        atom_charge = quantum["atom_charge"][:]
        atom_latent = quantum["atom_latent"][:]
        molecule_properties = quantum["molecule_properties"][:]
        heavy_mask = quantum["heavy_mask"][:]
        sample["quantum_features"] = np.concatenate(
            (
                atom_latent,
                atom_charge[:, None],
                np.repeat(molecule_properties[None, :], len(atom_charge), axis=0),
                heavy_mask[:, None],
            ),
            axis=1,
        ).astype(np.float32)
        return sample


def collate_quantum_trajectory(samples):
    batch = collate_trajectory_rollout(samples)
    batch["quantum_features"] = torch.as_tensor(
        np.concatenate([sample["quantum_features"] for sample in samples]),
        dtype=torch.float32,
    )
    return batch
