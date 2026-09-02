import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.history_trajectory_model import HistoryQuantumTrajectoryModel
from training.history_joint_trajectory_loss import rollout_components
from training.history_trajectory_data import (
    HistoryQuantumTrajectoryDataset,
    collate_history_quantum_trajectory,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--samples", type=int, required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
output = Path(config["outputs"]["probe"])
if output.exists():
    raise FileExistsError(output)
task_lengths = {"T1": (10, 10), "T2": (80, 20), "T3": (20, 80)}
observed_frames, prediction_frames = task_lengths[config["task"]]
torch.manual_seed(int(config["seed"]))
torch.cuda.manual_seed_all(int(config["seed"]))
device = torch.device("cuda")

dataset = HistoryQuantumTrajectoryDataset(
    config["data"]["manifest"],
    config["data"]["internal_split"],
    "fit",
    config["data"]["graph_cache"],
    config["data"]["coordinate_cache"],
    config["data"]["quantum_cache"],
    observed_frames=observed_frames,
    prediction_frames=prediction_frames,
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_history_quantum_trajectory,
)
model = HistoryQuantumTrajectoryModel().to(device)
checkpoint = torch.load(
    config["initialization"]["checkpoint"],
    map_location=device,
    weights_only=False,
)
incompatible = model.load_state_dict(
    checkpoint["model_state_dict"], strict=False
)
print(
    {
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }
)
model.eval()
totals = None
names = None
with torch.no_grad():
    for sample_index, batch in enumerate(loader, start=1):
        batch = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in batch.items()
        }
        components = rollout_components(
            model,
            batch,
            float(config["graph"]["cross_cutoff_angstrom"]),
            float(config["loss"]["contact_temperature"]),
        )
        if names is None:
            names = tuple(components)
            totals = np.zeros(len(names), dtype=np.float64)
        totals += np.asarray([float(components[name]) for name in names])
        if sample_index == args.samples:
            break

means = dict(zip(names, totals / args.samples))
old = config["loss"]["old_weights"]
weights = {
    "coordinate": float(old["coordinate"]),
    "velocity_vector": means["coordinate"] / means["velocity_vector"],
    "bond": float(old["bond"]),
    "angle": float(old["angle"]),
    "atom_rmsf": float(old["rmsf"])
    * means["legacy_rmsf"]
    / means["atom_rmsf"],
    "speed_w1": float(old["speed_w1"]),
    "contact_w1": float(old["contact_w1"]),
    "contact_transition": float(old["contact_transition"]),
    "multi_lag_velocity_correlation": float(old["velocity_correlation"])
    * means["legacy_velocity_correlation"]
    / means["multi_lag_velocity_correlation"],
    "radius_of_gyration": float(old["radius_of_gyration"]),
}
rows = [
    {
        "sample_count": args.samples,
        "component": name,
        "mean": means[name],
        "weight": weights.get(name, ""),
    }
    for name in names
]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"component_means": means, "weights": weights}, indent=2))
