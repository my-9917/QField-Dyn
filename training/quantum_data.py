import h5py
import torch
from torch_geometric.data import Data


def load_quantum_graphs(cache_path, ids):
    graphs = []
    with h5py.File(cache_path, "r") as cache:
        atom_mean = float(cache.attrs["atom_mean"])
        atom_std = float(cache.attrs["atom_std"])
        molecule_mean = torch.as_tensor(
            cache.attrs["molecule_mean"], dtype=torch.float32
        )
        molecule_std = torch.as_tensor(
            cache.attrs["molecule_std"], dtype=torch.float32
        )
        for complex_id in ids:
            group = cache[complex_id]
            bonds = torch.as_tensor(group["bonds"][:], dtype=torch.long)
            edge_index = torch.cat((bonds, bonds.flip(1)), dim=0).T
            graphs.append(
                Data(
                    z=torch.as_tensor(group["atomic_numbers"][:], dtype=torch.long),
                    pos=torch.as_tensor(group["coordinates"][:], dtype=torch.float32),
                    edge_index=edge_index,
                    atom_target=(
                        torch.as_tensor(group["atom_target"][:], dtype=torch.float32)
                        - atom_mean
                    )
                    / atom_std,
                    molecule_target=(
                        torch.as_tensor(
                            group["molecule_target"][:], dtype=torch.float32
                        )
                        - molecule_mean
                    )[None, :]
                    / molecule_std[None, :],
                    complex_id=complex_id,
                )
            )
    stats = {
        "atom_mean": atom_mean,
        "atom_std": atom_std,
        "molecule_mean": molecule_mean,
        "molecule_std": molecule_std,
    }
    return graphs, stats
