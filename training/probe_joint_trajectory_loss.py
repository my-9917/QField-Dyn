import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.trajectory_model import QuantumTrajectoryModel

from .joint_trajectory_loss import rollout_components
from .quantum_trajectory_data import (
    QuantumTrajectoryDataset,
    collate_quantum_trajectory,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
output = Path(config["outputs"]["probe"])
if output.exists():
    raise FileExistsError(output)

seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dataset = QuantumTrajectoryDataset(
    config["data"]["md"],
    config["data"]["train_graph_cache"],
    config["data"]["manifest"],
    "train",
    seed,
    int(config["training"]["rollout_steps"]),
    quantum_cache_path=config["data"]["train_quantum_features"],
    ligand_coordinate_cache_path=config["data"]["train_ligand_coordinates"],
)
dataset.set_epoch(1)
loader = DataLoader(
    dataset,
    batch_size=int(config["training"]["probe_batch_size"]),
    shuffle=False,
    num_workers=0,
    collate_fn=collate_quantum_trajectory,
)
batch = next(iter(loader))
device = torch.device("cuda")
batch = {
    name: value.to(device) if torch.is_tensor(value) else value
    for name, value in batch.items()
}
model = QuantumTrajectoryModel(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
    quantum_feature_dim=int(config["model"]["quantum_feature_dim"]),
).to(device)
checkpoint = torch.load(
    config["initialization"]["checkpoint"],
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.train()
components = rollout_components(
    model,
    batch,
    float(config["graph"]["cross_cutoff_angstrom"]),
    float(config["loss"]["contact_temperature"]),
)
rows = []
for name, value in components.items():
    model.zero_grad(set_to_none=True)
    value.backward(retain_graph=True)
    gradient_norm = torch.sqrt(
        sum(
            torch.sum(parameter.grad**2)
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    rows.append(
        {
            "component": name,
            "value": float(value.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
    )
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(rows)
