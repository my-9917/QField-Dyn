import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.trajectory_model import QuantumTrajectoryModel
from training.quantum_trajectory_data import QuantumTrajectoryDataset


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--layer-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-systems", type=int)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
dataset = QuantumTrajectoryDataset(
    config["data"]["md"],
    config["data"]["train_graph_cache"],
    config["data"]["manifest"],
    "train",
    int(config["seed"]),
    80,
    fixed_frame_index=19,
    quantum_cache_path=config["data"]["train_quantum_features"],
    ligand_coordinate_cache_path=config["data"].get("train_ligand_coordinates"),
)
if args.max_systems is not None:
    dataset.rows = dataset.rows[: args.max_systems]
row_by_id = {row["complex_id"]: row for row in dataset.rows}

device = torch.device("cuda")
model = QuantumTrajectoryModel(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
    quantum_feature_dim=int(config["model"]["quantum_feature_dim"]),
).to(device)
model.load_state_dict(
    torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"]
)
model.eval()
cutoff = float(config["graph"]["cross_cutoff_angstrom"])

args.output.parent.mkdir(parents=True, exist_ok=True)
start_time = time.perf_counter()
with h5py.File(args.layer_cache, "r") as layer_cache, h5py.File(args.output, "w") as output:
    output.attrs["split"] = "train"
    output.attrs["task"] = "T3"
    output.attrs["expert"] = "quantum trajectory expert"
    output.attrs["system_count"] = len(dataset)
    output.attrs["inference_batch_size"] = 1
    output.attrs["qm_lookup"] = "none"
    with torch.no_grad():
        for dataset_index in range(len(dataset)):
            sample = dataset[dataset_index]
            complex_id = sample["complex_id"]
            ligand_z = torch.as_tensor(
                sample["ligand_atomic_numbers"], dtype=torch.long, device=device
            )
            protein_z = torch.as_tensor(
                sample["protein_atomic_numbers"], dtype=torch.long, device=device
            )
            protein_x = torch.as_tensor(
                sample["protein_coordinates"], dtype=torch.float32, device=device
            )
            bonds = torch.as_tensor(sample["ligand_bonds"], dtype=torch.long, device=device)
            bond_index = torch.cat((bonds, bonds.flip(1)), dim=0).T
            previous = torch.as_tensor(
                sample["previous_coordinates"], dtype=torch.float32, device=device
            )
            current = torch.as_tensor(
                sample["current_coordinates"], dtype=torch.float32, device=device
            )
            velocity = current - previous
            quantum_features = torch.as_tensor(
                sample["quantum_features"], dtype=torch.float32, device=device
            )
            predictions = []
            for horizon in range(1, 81):
                cross = torch.nonzero(
                    torch.cdist(current, protein_x) <= cutoff, as_tuple=False
                ).T
                next_coordinates = current + velocity + model(
                    ligand_z,
                    protein_z,
                    current,
                    protein_x,
                    velocity,
                    bond_index,
                    cross,
                    quantum_features,
                )
                if not torch.isfinite(next_coordinates).all():
                    raise FloatingPointError(
                        f"{complex_id}: non-finite trajectory at horizon {horizon}"
                    )
                predictions.append(next_coordinates)
                velocity, current = next_coordinates - current, next_coordinates
            prediction = torch.stack(predictions).cpu().numpy()

            quantum = layer_cache[complex_id]
            heavy = sample["ligand_atomic_numbers"] != 1
            layers = quantum["atom_latent_layers"][:][heavy]
            charge = quantum["atom_charge"][:][heavy]
            heavy_current = sample["current_coordinates"][heavy]
            heavy_previous = sample["previous_coordinates"][heavy]
            centered = heavy_current - heavy_current.mean(axis=0)
            md_group = dataset.md[row_by_id[complex_id]["md_group"]]
            ligand_start = int(md_group["molecules_begin_atom_index"][-1])
            protein_heavy = md_group["atoms_number"][:ligand_start] != 1
            fixed_protein = md_group["trajectory_coordinates"][0, :ligand_start][protein_heavy]
            context = np.concatenate(
                (
                    layers.mean(axis=0).reshape(-1),
                    layers.std(axis=0).reshape(-1),
                    np.asarray(
                        [
                            np.log1p(heavy.sum()),
                            charge.mean(),
                            charge.std(),
                            *quantum["molecule_properties"][:],
                            np.linalg.norm(
                                heavy_current.mean(axis=0) - heavy_previous.mean(axis=0)
                            ),
                            np.sqrt(np.mean((heavy_current - heavy_previous) ** 2)),
                            np.sqrt(np.mean(np.sum(centered**2, axis=1))),
                            np.min(
                                np.linalg.norm(
                                    heavy_current[:, None] - fixed_protein[None], axis=-1
                                )
                            ),
                        ],
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32)
            group = output.create_group(complex_id)
            group.create_dataset("q_prediction", data=prediction, compression="lzf")
            group.create_dataset("context", data=context)
            completed = dataset_index + 1
            if completed % 320 == 0 or completed == len(dataset):
                elapsed = time.perf_counter() - start_time
                print(
                    f"generated {completed}/{len(dataset)} elapsed={elapsed:.1f}s rate={completed / elapsed:.2f}/s",
                    flush=True,
                )

elapsed = time.perf_counter() - start_time
print(
    {
        "system_count": len(dataset),
        "elapsed_seconds": elapsed,
        "systems_per_second": len(dataset) / elapsed,
        "projected_full_minutes": 13066 / (len(dataset) / elapsed) / 60,
        "output_bytes": args.output.stat().st_size,
    }
)
