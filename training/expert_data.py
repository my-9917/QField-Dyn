import csv

import h5py
import numpy as np
from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    def __init__(
        self,
        q_cache_path,
        coordinate_cache_path,
        graph_cache_path,
        split_path,
        subset,
        context_cache_path=None,
    ):
        with open(split_path, encoding="utf-8", newline="") as handle:
            self.ids = [
                row["complex_id"]
                for row in csv.DictReader(handle)
                if row["partition"] == subset
            ]
        self.q_cache = h5py.File(q_cache_path, "r")
        self.context_cache = (
            h5py.File(context_cache_path, "r") if context_cache_path else self.q_cache
        )
        self.coordinates = h5py.File(coordinate_cache_path, "r")
        self.graphs = h5py.File(graph_cache_path, "r")
        horizon = np.arange(1, 81, dtype=np.float32)
        self.damping = (1 - 0.25**horizon) / (1 - 0.25)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        complex_id = self.ids[index]
        q_group = self.q_cache[complex_id]
        frames = self.coordinates[complex_id]["ligand_coordinates"][18:100]
        atomic_numbers = self.graphs[complex_id]["ligand_atomic_numbers"][:]
        heavy = atomic_numbers != 1
        previous, current = frames[:2]
        centroid_velocity = (
            current[heavy].mean(axis=0) - previous[heavy].mean(axis=0)
        )
        observation_prediction = (
            current[None]
            - 0.20 * self.damping[:, None, None] * centroid_velocity[None, None]
        )
        return {
            "complex_id": complex_id,
            "context": self.context_cache[complex_id]["context"][:].astype(np.float32),
            "q_prediction": q_group["q_prediction"][:].astype(np.float32),
            "o_prediction": observation_prediction.astype(np.float32),
            "current": current.astype(np.float32),
            "target": frames[2:].astype(np.float32),
            "heavy": heavy,
            "bonds": self.graphs[complex_id]["ligand_bonds"][:].astype(np.int64),
        }


def collate_expert(samples):
    return samples
