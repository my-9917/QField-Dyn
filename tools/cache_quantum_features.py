import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.quantum_features import QuantumFeatureEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--md", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--split", choices=("train", "validation"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
with args.manifest.open(encoding="utf-8", newline="") as handle:
    manifest = [row for row in csv.DictReader(handle) if row["split"] == args.split]
with h5py.File(config["data"]["cache"], "r") as quantum_cache:
    atom_mean = float(quantum_cache.attrs["atom_mean"])
    atom_std = float(quantum_cache.attrs["atom_std"])
    molecule_mean = np.asarray(quantum_cache.attrs["molecule_mean"], dtype=np.float32)
    molecule_std = np.asarray(quantum_cache.attrs["molecule_std"], dtype=np.float32)

device = torch.device("cuda")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = QuantumFeatureEncoder(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.md, "r") as md, h5py.File(
    args.graph_cache, "r"
) as graphs, h5py.File(args.output, "w") as output:
    output.attrs["method"] = "structure-derived multilayer quantum features"
    output.attrs["split"] = args.split
    output.attrs["source_checkpoint"] = str(args.checkpoint)
    output.attrs["qm_lookup"] = "none"
    output.attrs["frame_index"] = 0
    output.attrs["layer_count"] = int(config["model"]["layer_count"]) + 1
    with torch.no_grad():
        for system_index, row in enumerate(manifest, start=1):
            complex_id = row["complex_id"]
            graph = graphs[complex_id]
            md_group = md[row["md_group"]]
            ligand_start = int(md_group["molecules_begin_atom_index"][-1])
            coordinates = md_group["trajectory_coordinates"][0, ligand_start:].astype(
                np.float32
            )
            atomic_numbers = graph["ligand_atomic_numbers"][:].astype(np.int64)
            heavy = atomic_numbers != 1
            heavy_indices = np.flatnonzero(heavy)
            old_to_heavy = np.full(len(atomic_numbers), -1, dtype=np.int64)
            old_to_heavy[heavy_indices] = np.arange(len(heavy_indices))
            bonds = graph["ligand_bonds"][:].astype(np.int64)
            heavy_bonds = old_to_heavy[bonds[np.all(heavy[bonds], axis=1)]]
            edge_index = np.concatenate((heavy_bonds, heavy_bonds[:, ::-1]), axis=0).T
            z = torch.as_tensor(atomic_numbers[heavy], dtype=torch.long, device=device)
            atom_prediction, molecule_prediction, layers = model.forward_layers(
                z,
                torch.as_tensor(coordinates[heavy], dtype=torch.float32, device=device),
                torch.as_tensor(edge_index, dtype=torch.long, device=device),
                torch.zeros(len(z), dtype=torch.long, device=device),
            )
            atom_charge = np.zeros(len(atomic_numbers), dtype=np.float32)
            atom_charge[heavy] = atom_prediction.cpu().numpy() * atom_std + atom_mean
            atom_layers = np.zeros(
                (len(atomic_numbers), layers.shape[1], layers.shape[2]), dtype=np.float32
            )
            atom_layers[heavy] = layers.cpu().numpy()
            molecule_properties = (
                molecule_prediction[0].cpu().numpy() * molecule_std + molecule_mean
            )
            group = output.create_group(complex_id)
            group.create_dataset("atom_charge", data=atom_charge)
            group.create_dataset("atom_latent", data=atom_layers[:, -1], compression="gzip")
            group.create_dataset("atom_latent_layers", data=atom_layers, compression="gzip")
            group.create_dataset("molecule_properties", data=molecule_properties)
            group.create_dataset("heavy_mask", data=heavy.astype(np.int8))
            if system_index % 1000 == 0 or system_index == len(manifest):
                print(f"cached {system_index}/{len(manifest)}", flush=True)
