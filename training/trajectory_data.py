import csv

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryRolloutDataset(Dataset):
    def __init__(
        self,
        md_path,
        graph_cache_path,
        manifest_path,
        split,
        seed,
        rollout_steps,
        fixed_frame_index=None,
        ligand_coordinate_cache_path=None,
    ):
        with open(manifest_path, encoding="utf-8", newline="") as handle:
            self.rows = [
                row for row in csv.DictReader(handle) if row["split"] == split
            ]
        self.md = h5py.File(md_path, "r")
        self.graph_cache = h5py.File(graph_cache_path, "r")
        self.ligand_coordinate_cache = (
            h5py.File(ligand_coordinate_cache_path, "r")
            if ligand_coordinate_cache_path is not None
            else None
        )
        self.seed = seed
        self.rollout_steps = rollout_steps
        self.fixed_frame_index = fixed_frame_index
        self.frame_indices = None

    def set_epoch(self, epoch):
        generator = np.random.default_rng(self.seed + epoch)
        self.frame_indices = generator.integers(
            1, 100 - self.rollout_steps, size=len(self.rows)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        complex_id = row["complex_id"]
        md_group = self.md[row["md_group"]]
        graph = self.graph_cache[complex_id]
        ligand_start = int(md_group["molecules_begin_atom_index"][-1])
        frame_index = (
            self.fixed_frame_index
            if self.fixed_frame_index is not None
            else int(self.frame_indices[index])
        )
        if self.ligand_coordinate_cache is None:
            frames = md_group["trajectory_coordinates"][
                frame_index - 1 : frame_index + self.rollout_steps + 1,
                ligand_start:,
                :,
            ].astype(np.float32)
        else:
            frames = self.ligand_coordinate_cache[complex_id]["ligand_coordinates"][
                frame_index - 1 : frame_index + self.rollout_steps + 1
            ].astype(np.float32)
        return {
            "complex_id": complex_id,
            "ligand_atomic_numbers": graph["ligand_atomic_numbers"][:],
            "protein_atomic_numbers": graph["protein_atomic_numbers"][:],
            "previous_coordinates": frames[0],
            "current_coordinates": frames[1],
            "target_coordinates": frames[2:],
            "protein_coordinates": graph["protein_coordinates"][:],
            "ligand_bonds": graph["ligand_bonds"][:],
        }


def collate_trajectory_rollout(samples):
    ligand_atomic_numbers = []
    protein_atomic_numbers = []
    ligand_coordinates = []
    protein_coordinates = []
    ligand_velocity = []
    target_coordinates = []
    bond_indices = []
    angle_indices = []
    ligand_ptr = [0]
    protein_ptr = [0]

    for sample in samples:
        current = sample["current_coordinates"]
        bonds = sample["ligand_bonds"]
        directed_bonds = np.concatenate((bonds, bonds[:, ::-1]), axis=0)
        bond_indices.append((directed_bonds + ligand_ptr[-1]).T)
        neighbors = [set() for _ in range(len(current))]
        for atom_i, atom_j in bonds:
            neighbors[atom_i].add(atom_j)
            neighbors[atom_j].add(atom_i)
        angles = []
        for center, center_neighbors in enumerate(neighbors):
            ordered_neighbors = sorted(center_neighbors)
            for neighbor_index, atom_i in enumerate(ordered_neighbors):
                for atom_k in ordered_neighbors[neighbor_index + 1 :]:
                    angles.append((atom_i, center, atom_k))
        angle_indices.append(
            (np.asarray(angles, dtype=np.int64) + ligand_ptr[-1]).T
        )
        ligand_atomic_numbers.append(sample["ligand_atomic_numbers"])
        protein_atomic_numbers.append(sample["protein_atomic_numbers"])
        ligand_coordinates.append(current)
        protein_coordinates.append(sample["protein_coordinates"])
        ligand_velocity.append(current - sample["previous_coordinates"])
        target_coordinates.append(sample["target_coordinates"])
        ligand_ptr.append(ligand_ptr[-1] + len(current))
        protein_ptr.append(
            protein_ptr[-1] + len(sample["protein_coordinates"])
        )

    return {
        "ligand_atomic_numbers": torch.as_tensor(
            np.concatenate(ligand_atomic_numbers), dtype=torch.long
        ),
        "protein_atomic_numbers": torch.as_tensor(
            np.concatenate(protein_atomic_numbers), dtype=torch.long
        ),
        "ligand_coordinates": torch.as_tensor(
            np.concatenate(ligand_coordinates), dtype=torch.float32
        ),
        "protein_coordinates": torch.as_tensor(
            np.concatenate(protein_coordinates), dtype=torch.float32
        ),
        "ligand_velocity": torch.as_tensor(
            np.concatenate(ligand_velocity), dtype=torch.float32
        ),
        "target_coordinates": torch.as_tensor(
            np.concatenate(target_coordinates, axis=1), dtype=torch.float32
        ),
        "bond_index": torch.as_tensor(
            np.concatenate(bond_indices, axis=1), dtype=torch.long
        ),
        "angle_index": torch.as_tensor(
            np.concatenate(angle_indices, axis=1), dtype=torch.long
        ),
        "ligand_ptr": ligand_ptr,
        "protein_ptr": protein_ptr,
    }
