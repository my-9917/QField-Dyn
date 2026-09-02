import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from src.equivariant_residual import apply_patch_residual
from src.expert_gate import ExpertGate
from src.torsion_geometry import graph_data, kinematic_tree
from src.torsion_projection import project_to_torsion_manifold


parser = argparse.ArgumentParser()
parser.add_argument("--q-cache", type=Path, required=True)
parser.add_argument("--coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--internal-split", type=Path, required=True)
parser.add_argument("--partition-column", default="partition")
parser.add_argument("--expert-checkpoint", type=Path, required=True)
parser.add_argument("--geometry-coefficients", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

with args.internal_split.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row[args.partition_column] == "internal_validation"
    ]
with args.geometry_coefficients.open(encoding="utf-8", newline="") as handle:
    internal_coefficients = np.asarray(
        [
            [float(value) for value in list(row.values())[1:]]
            for row in csv.DictReader(handle)
        ]
    )
coefficients = np.zeros((4, 4), dtype=np.float64)
coefficients[:, 2:] = internal_coefficients

checkpoint = torch.load(args.expert_checkpoint, map_location="cpu", weights_only=False)
gate = ExpertGate(
    context_dim=int(checkpoint["config"]["model"]["context_dim"]),
    hidden_dim=int(checkpoint["config"]["model"]["hidden_dim"]),
)
gate.load_state_dict(checkpoint["model_state_dict"])
gate.eval()
context_mean = torch.as_tensor(checkpoint["context_mean"])
context_std = torch.as_tensor(checkpoint["context_std"])
segment_index = np.asarray([0] * 10 + [1] * 10 + [2] * 20 + [3] * 40)
horizon = np.arange(1, 81, dtype=np.float32)
damping = (1 - 0.25**horizon) / (1 - 0.25)
max_rms_displacement = 0.0

with h5py.File(args.q_cache, "r") as q_cache, h5py.File(
    args.coordinate_cache, "r"
) as coordinates, h5py.File(args.graph_cache, "r") as graphs, torch.no_grad():
    for index, complex_id in enumerate(ids, start=1):
        cache = q_cache[complex_id]
        q_prediction = cache["q_prediction"][:]
        weights = gate(
            (torch.as_tensor(cache["context"][:]) - context_mean) / context_std
        )[0].numpy()
        previous, current = coordinates[complex_id]["ligand_coordinates"][18:20]
        graph = graphs[complex_id]
        atomic_numbers = graph["ligand_atomic_numbers"][:]
        heavy = atomic_numbers != 1
        centroid_velocity = current[heavy].mean(0) - previous[heavy].mean(0)
        observation_prediction = (
            current[None]
            - 0.20 * damping[:, None, None] * centroid_velocity[None, None]
        )
        frame_weights = weights[segment_index]
        baseline = (
            frame_weights[:, 0, None, None] * q_prediction
            + frame_weights[:, 1, None, None] * observation_prediction
        )
        target = apply_patch_residual(
            previous, current, q_prediction, heavy, coefficients
        )
        bonds = graph["ligand_bonds"][:].astype(np.int64)
        rotatable, fragment, fragments = graph_data(len(current), bonds)
        _, order, subtree_atoms = kinematic_tree(
            current, rotatable, fragment, fragments
        )
        for frame, target_frame in zip(baseline, target):
            corrected, _ = project_to_torsion_manifold(
                frame, target_frame, heavy, order, subtree_atoms
            )
            displacement = float(
                np.sqrt(np.mean(np.sum((corrected - frame) ** 2, axis=-1)))
            )
            max_rms_displacement = max(max_rms_displacement, displacement)
        if index % 100 == 0 or index == len(ids):
            print(f"measured {index}/{len(ids)}", flush=True)

row = {
    "internal_validation_system_count": len(ids),
    "max_rms_displacement": max_rms_displacement,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
print(row)
