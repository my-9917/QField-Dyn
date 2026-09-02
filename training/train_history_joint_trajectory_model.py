import argparse
import csv
import time
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
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
task_lengths = {"T1": (10, 10), "T2": (80, 20), "T3": (20, 80)}
observed_frames, prediction_frames = task_lengths[config["task"]]
checkpoint_directory = Path(config["outputs"]["checkpoint_directory"])
log_path = Path(config["outputs"]["log"])
if checkpoint_directory.exists() or log_path.exists():
    raise FileExistsError("history training outputs already exist")

seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda")

dataset_arguments = (
    config["data"]["manifest"],
    config["data"]["internal_split"],
)
dataset_keywords = {
    "graph_cache_path": config["data"]["graph_cache"],
    "coordinate_cache_path": config["data"]["coordinate_cache"],
    "quantum_cache_path": config["data"]["quantum_cache"],
    "observed_frames": observed_frames,
    "prediction_frames": prediction_frames,
}
train_dataset = HistoryQuantumTrajectoryDataset(
    *dataset_arguments, "fit", **dataset_keywords
)
validation_dataset = HistoryQuantumTrajectoryDataset(
    *dataset_arguments, "internal_validation", **dataset_keywords
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(config["training"]["num_workers"]),
    collate_fn=collate_history_quantum_trajectory,
)

model = HistoryQuantumTrajectoryModel(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
    quantum_feature_dim=int(config["model"]["quantum_feature_dim"]),
).to(device)
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
    },
    flush=True,
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
)
component_names = tuple(config["loss"]["weights"])
loss_weights = {
    name: float(weight) for name, weight in config["loss"]["weights"].items()
}


def move_to_device(batch):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def calculate_components(batch):
    return rollout_components(
        model,
        move_to_device(batch),
        float(config["graph"]["cross_cutoff_angstrom"]),
        float(config["loss"]["contact_temperature"]),
    )


def evaluate():
    totals = np.zeros(len(component_names) + 1, dtype=np.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader, start=1):
            components = calculate_components(batch)
            values = np.asarray(
                [float(components[name]) for name in component_names]
            )
            loss = sum(
                loss_weights[name] * components[name]
                for name in component_names
            )
            totals += np.concatenate(([float(loss)], values))
            count += 1
            if batch_index % 200 == 0 or batch_index == len(validation_loader):
                print(
                    f"validated {batch_index}/{len(validation_loader)}",
                    flush=True,
                )
    return totals / count


checkpoint_directory.mkdir(parents=True)
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
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_history_quantum_trajectory,
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
            optimizer.zero_grad(set_to_none=True)
            component_total = np.zeros(len(component_names), dtype=np.float64)
            fraction = 1.0 / len(micro_batches)
            for item in micro_batches:
                components = calculate_components(item)
                loss = sum(
                    loss_weights[name] * components[name]
                    for name in component_names
                )
                (fraction * loss).backward()
                component_total += fraction * np.asarray(
                    [
                        float(components[name].detach())
                        for name in component_names
                    ]
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
            batch_count = len(micro_batches)
            train_total += batch_count * np.concatenate(
                ([weighted], component_total)
            )
            train_count += batch_count
            micro_batches = []
            if (
                batch_index == accumulation_steps
                or batch_index % 200 == 0
                or batch_index == len(train_loader)
            ):
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
                for name, value in zip(
                    component_names, validation_metrics[1:]
                )
            }
        )
        row["elapsed_seconds"] = elapsed_seconds
        writer.writerow(row)
        log_file.flush()
        print(row, flush=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": float(validation_metrics[0]),
                "loss_weights": loss_weights,
                "task": config["task"],
                "seed": seed,
            },
            checkpoint_directory
            / f"history_trajectory_{config['task'].lower()}_epoch{epoch}.pt",
        )
        if validation_metrics[0] < best_validation_loss:
            best_validation_loss = float(validation_metrics[0])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(
            config["training"]["early_stopping_patience"]
        ):
            break
        if elapsed_seconds >= 3600 * float(
            config["training"]["max_wall_hours"]
        ):
            break
