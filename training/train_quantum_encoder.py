import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.quantum_encoder import QuantumEncoder

from .quantum_data import load_quantum_graphs


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

output_directory = Path(config["outputs"]["directory"])
log_path = Path(config["outputs"]["log"])
metrics_path = Path(config["outputs"]["metrics"])
checkpoint_path = output_directory / "quantum_encoder.pt"
for output_path in (log_path, metrics_path, checkpoint_path):
    if output_path.exists():
        raise FileExistsError(output_path)

with Path(config["data"]["internal_split"]).open(
    encoding="utf-8", newline=""
) as handle:
    split_rows = list(csv.DictReader(handle))
train_ids = [
    row["complex_id"] for row in split_rows if row["internal_split"] == "train"
]
validation_ids = [
    row["complex_id"]
    for row in split_rows
    if row["internal_split"] == "validation"
]
train_graphs, stats = load_quantum_graphs(config["data"]["cache"], train_ids)
validation_graphs, validation_stats = load_quantum_graphs(
    config["data"]["cache"], validation_ids
)
if stats["atom_mean"] != validation_stats["atom_mean"]:
    raise ValueError("normalization mismatch")

batch_size = int(config["training"]["batch_size"])
train_loader = DataLoader(
    train_graphs,
    batch_size=batch_size,
    shuffle=True,
    num_workers=int(config["training"]["num_workers"]),
)
validation_loader = DataLoader(
    validation_graphs,
    batch_size=batch_size,
    shuffle=False,
    num_workers=int(config["training"]["num_workers"]),
)
device = torch.device("cuda")
model = QuantumEncoder(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
)

output_directory.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
history = []
best_score = float("inf")
epochs_without_improvement = 0
for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
    model.train()
    train_loss_sum = 0.0
    train_batches = 0
    for batch in train_loader:
        batch = batch.to(device)
        atom_prediction, molecule_prediction, _ = model(
            batch.z, batch.pos, batch.edge_index, batch.batch
        )
        atom_loss = torch.mean((atom_prediction - batch.atom_target) ** 2)
        molecule_loss = torch.mean(
            (molecule_prediction - batch.molecule_target) ** 2
        )
        loss = atom_loss + molecule_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch}: non-finite train loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError(f"epoch {epoch}: non-finite gradients")
        optimizer.step()
        train_loss_sum += float(loss.detach())
        train_batches += 1

    model.eval()
    atom_absolute_error = 0.0
    atom_count = 0
    molecule_absolute_error = torch.zeros(2, device=device)
    molecule_count = 0
    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(device)
            atom_prediction, molecule_prediction, _ = model(
                batch.z, batch.pos, batch.edge_index, batch.batch
            )
            atom_absolute_error += float(
                torch.sum(torch.abs(atom_prediction - batch.atom_target))
            )
            atom_count += len(batch.atom_target)
            molecule_absolute_error += torch.sum(
                torch.abs(molecule_prediction - batch.molecule_target), dim=0
            )
            molecule_count += len(batch.molecule_target)
    atom_mae_normalized = atom_absolute_error / atom_count
    molecule_mae_normalized = molecule_absolute_error / molecule_count
    score = float(
        (atom_mae_normalized + molecule_mae_normalized.mean().item()) / 2
    )
    if not np.isfinite(score):
        raise FloatingPointError(f"epoch {epoch}: non-finite validation score")
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_batches,
            "validation_atom_mae_normalized": atom_mae_normalized,
            "validation_ea_mae_normalized": float(molecule_mae_normalized[0]),
            "validation_hardness_mae_normalized": float(
                molecule_mae_normalized[1]
            ),
            "selection_score": score,
        }
    )
    print(history[-1], flush=True)
    if score < best_score:
        best_score = score
        epochs_without_improvement = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "selection_score": score,
                "config": config,
            },
            checkpoint_path,
        )
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= int(
        config["training"]["early_stopping_patience"]
    ):
        break

with log_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
atom_predictions = []
atom_targets = []
molecule_predictions = []
molecule_targets = []
with torch.no_grad():
    for batch in validation_loader:
        batch = batch.to(device)
        atom_prediction, molecule_prediction, _ = model(
            batch.z, batch.pos, batch.edge_index, batch.batch
        )
        atom_predictions.append(atom_prediction.cpu())
        atom_targets.append(batch.atom_target.cpu())
        molecule_predictions.append(molecule_prediction.cpu())
        molecule_targets.append(batch.molecule_target.cpu())
atom_predictions = torch.cat(atom_predictions).numpy() * stats["atom_std"] + stats[
    "atom_mean"
]
atom_targets = torch.cat(atom_targets).numpy() * stats["atom_std"] + stats[
    "atom_mean"
]
molecule_predictions = torch.cat(molecule_predictions)
molecule_targets = torch.cat(molecule_targets)
molecule_predictions = (
    molecule_predictions * stats["molecule_std"] + stats["molecule_mean"]
).numpy()
molecule_targets = (
    molecule_targets * stats["molecule_std"] + stats["molecule_mean"]
).numpy()

target_data = [
    ("gfn2_charge_(water)", atom_predictions, atom_targets, float(stats["atom_mean"])),
    (
        "Electron_Affinity",
        molecule_predictions[:, 0],
        molecule_targets[:, 0],
        float(stats["molecule_mean"][0]),
    ),
    (
        "Hardness",
        molecule_predictions[:, 1],
        molecule_targets[:, 1],
        float(stats["molecule_mean"][1]),
    ),
]
metric_rows = []
for target, prediction, truth, baseline_value in target_data:
    mae = float(np.mean(np.abs(prediction - truth)))
    baseline_mae = float(np.mean(np.abs(truth - baseline_value)))
    metric_rows.append(
        {
            "model": "QField-Dyn quantum encoder",
            "best_epoch": checkpoint["epoch"],
            "target": target,
            "validation_count": len(truth),
            "model_mae": mae,
            "baseline_mae": baseline_mae,
            "relative_mae_reduction": 1 - mae / baseline_mae,
            "pearson_r": float(np.corrcoef(prediction, truth)[0, 1]),
        }
    )
charge_pass = metric_rows[0]["model_mae"] < metric_rows[0]["baseline_mae"]
molecule_pass = any(
    row["model_mae"] < row["baseline_mae"] for row in metric_rows[1:]
)
for row in metric_rows:
    row["beats_required_baselines"] = charge_pass and molecule_pass
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with metrics_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
    writer.writeheader()
    writer.writerows(metric_rows)
print(metric_rows, flush=True)
