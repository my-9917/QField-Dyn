import argparse
import csv
import time
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
checkpoint_path = Path(config["outputs"]["checkpoint"])
log_path = Path(config["outputs"]["log"])
for output_path in (checkpoint_path, log_path):
    if output_path.exists():
        raise FileExistsError(output_path)

seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
with open(config["data"]["internal_split"], encoding="utf-8", newline="") as handle:
    partition = {
        row["complex_id"]: row["partition"] for row in csv.DictReader(handle)
    }

dataset_arguments = (
    config["data"]["md"],
    config["data"]["train_graph_cache"],
    config["data"]["manifest"],
    "train",
    seed,
    int(config["training"]["rollout_steps"]),
)
dataset_keywords = {
    "quantum_cache_path": config["data"]["train_quantum_features"],
    "ligand_coordinate_cache_path": config["data"]["train_ligand_coordinates"],
}
train_dataset = QuantumTrajectoryDataset(*dataset_arguments, **dataset_keywords)
train_dataset.rows = [
    row for row in train_dataset.rows if partition[row["complex_id"]] == "fit"
]
validation_dataset = QuantumTrajectoryDataset(
    *dataset_arguments,
    fixed_frame_index=int(config["validation"]["current_frame_index"]),
    **dataset_keywords,
)
validation_dataset.rows = [
    row
    for row in validation_dataset.rows
    if partition[row["complex_id"]] == "internal_validation"
]
validation_loader = DataLoader(
    validation_dataset,
    batch_size=int(config["training"]["validation_batch_size"]),
    shuffle=False,
    num_workers=int(config["training"]["num_workers"]),
    collate_fn=collate_quantum_trajectory,
)

device = torch.device("cuda")
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
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
)
component_names = tuple(config["loss"]["weights"])
loss_weights = {
    name: float(weight) for name, weight in config["loss"]["weights"].items()
}


def calculate_components(batch):
    batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    return rollout_components(
        model,
        batch,
        float(config["graph"]["cross_cutoff_angstrom"]),
        float(config["loss"]["contact_temperature"]),
    )


def evaluate():
    totals = np.zeros(len(component_names) + 1, dtype=np.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            components = calculate_components(batch)
            values = np.asarray(
                [float(components[name]) for name in component_names]
            )
            loss = sum(
                loss_weights[name] * components[name] for name in component_names
            )
            batch_count = len(batch["ligand_ptr"]) - 1
            totals += batch_count * np.concatenate(([float(loss)], values))
            count += batch_count
    return totals / count


checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = (
    ["epoch", "train_loss"]
    + [f"train_{name}" for name in component_names]
    + ["validation_loss"]
    + [f"validation_{name}" for name in component_names]
    + ["elapsed_seconds"]
)
best_validation_loss = float("inf")
epochs_without_improvement = 0
start_time = time.perf_counter()
with log_path.open("w", encoding="utf-8", newline="") as log_file:
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["micro_batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_quantum_trajectory,
        )
        model.train()
        train_total = np.zeros(len(component_names) + 1, dtype=np.float64)
        train_count = 0
        micro_batches = []
        accumulation_steps = int(
            config["training"]["gradient_accumulation_steps"]
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            micro_batches.append(batch)
            if len(micro_batches) < accumulation_steps and batch_index < len(
                train_loader
            ):
                continue
            counts = np.asarray(
                [len(item["ligand_ptr"]) - 1 for item in micro_batches],
                dtype=np.float64,
            )
            total_count = counts.sum()
            optimizer.zero_grad(set_to_none=True)
            component_total = np.zeros(len(component_names), dtype=np.float64)
            for item, item_count in zip(micro_batches, counts):
                components = calculate_components(item)
                fraction = item_count / total_count
                loss = sum(
                    loss_weights[name] * components[name]
                    for name in component_names
                )
                (fraction * loss).backward()
                component_total += fraction * np.asarray(
                    [float(components[name].detach()) for name in component_names]
                )
            if not all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError(
                    f"epoch {epoch} batch {batch_index}: non-finite gradients"
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            weighted = float(
                sum(
                    loss_weights[name] * value
                    for name, value in zip(component_names, component_total)
                )
            )
            train_total += total_count * np.concatenate(
                ([weighted], component_total)
            )
            train_count += total_count
            micro_batches = []
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch} trained {batch_index}/{len(train_loader)}",
                    flush=True,
                )

        train_metrics = train_total / train_count
        validation_metrics = evaluate()
        elapsed_seconds = time.perf_counter() - start_time
        row = {"epoch": epoch, "train_loss": train_metrics[0]}
        row.update(
            {
                f"train_{name}": value
                for name, value in zip(component_names, train_metrics[1:])
            }
        )
        row["validation_loss"] = validation_metrics[0]
        row.update(
            {
                f"validation_{name}": value
                for name, value in zip(component_names, validation_metrics[1:])
            }
        )
        row["elapsed_seconds"] = elapsed_seconds
        writer.writerow(row)
        log_file.flush()
        print(row, flush=True)

        if validation_metrics[0] < best_validation_loss:
            best_validation_loss = float(validation_metrics[0])
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "loss_weights": loss_weights,
                    "task": config["task"],
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(
            config["training"]["early_stopping_patience"]
        ):
            break
        if elapsed_seconds >= 3600 * float(config["training"]["max_wall_hours"]):
            break
