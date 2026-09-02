import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.expert_gate import ExpertGate

from .expert_data import ExpertDataset, collate_expert


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
fit_dataset = ExpertDataset(
    config["data"]["q_cache"],
    config["data"]["ligand_coordinates"],
    config["data"]["graph_cache"],
    config["data"]["internal_split"],
    "fit",
    config["data"].get("context_cache"),
)
validation_dataset = ExpertDataset(
    config["data"]["q_cache"],
    config["data"]["ligand_coordinates"],
    config["data"]["graph_cache"],
    config["data"]["internal_split"],
    "internal_validation",
    config["data"].get("context_cache"),
)
fit_context = np.stack(
    [fit_dataset.context_cache[complex_id]["context"][:] for complex_id in fit_dataset.ids]
).astype(np.float32)
context_mean = fit_context.mean(axis=0)
context_std = fit_context.std(axis=0)
if np.any(context_std == 0):
    raise ValueError("fit context contains constant dimensions")

batch_size = int(config["training"]["batch_size"])
fit_loader = DataLoader(
    fit_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed),
    num_workers=int(config["training"]["num_workers"]),
    collate_fn=collate_expert,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=int(config["training"]["num_workers"]),
    collate_fn=collate_expert,
)
device = torch.device("cuda")
model = ExpertGate(
    context_dim=int(config["model"]["context_dim"]),
    hidden_dim=int(config["model"]["hidden_dim"]),
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
)
context_mean_tensor = torch.as_tensor(context_mean, device=device)
context_std_tensor = torch.as_tensor(context_std, device=device)
segment_index = torch.as_tensor(
    [0] * 10 + [1] * 10 + [2] * 20 + [3] * 40,
    dtype=torch.long,
    device=device,
)
loss_weights = (
    float(config["loss"]["coordinate_weight"]),
    float(config["loss"]["bond_weight"]),
    float(config["loss"]["velocity_weight"]),
)


def batch_loss(samples):
    context = torch.as_tensor(
        np.stack([sample["context"] for sample in samples]), device=device
    )
    segment_weights = model(
        (context - context_mean_tensor) / context_std_tensor
    )
    losses = []
    components = []
    for sample_index, sample in enumerate(samples):
        q_prediction = torch.as_tensor(sample["q_prediction"], device=device)
        o_prediction = torch.as_tensor(sample["o_prediction"], device=device)
        target = torch.as_tensor(sample["target"], device=device)
        current = torch.as_tensor(sample["current"], device=device)
        heavy = torch.as_tensor(sample["heavy"], device=device)
        bonds = torch.as_tensor(sample["bonds"], device=device)
        weights = segment_weights[sample_index][segment_index]
        prediction = (
            weights[:, 0, None, None] * q_prediction
            + weights[:, 1, None, None] * o_prediction
        )
        coordinate = torch.mean((prediction[:, heavy] - target[:, heavy]) ** 2)
        pred_bond = torch.linalg.vector_norm(
            prediction[:, bonds[:, 0]] - prediction[:, bonds[:, 1]], dim=-1
        )
        true_bond = torch.linalg.vector_norm(
            target[:, bonds[:, 0]] - target[:, bonds[:, 1]], dim=-1
        )
        bond = torch.mean((pred_bond - true_bond) ** 2)
        pred_velocity = torch.diff(
            torch.cat((current[None], prediction), dim=0), dim=0
        )
        true_velocity = torch.diff(
            torch.cat((current[None], target), dim=0), dim=0
        )
        velocity = torch.mean((pred_velocity - true_velocity) ** 2)
        components.append(torch.stack((coordinate, bond, velocity)))
        losses.append(sum(weight * value for weight, value in zip(loss_weights, components[-1])))
    component_mean = torch.stack(components).mean(dim=0)
    return torch.stack(losses).mean(), component_mean, segment_weights


def evaluate(loader):
    total = np.zeros(4, dtype=np.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for samples in loader:
            loss, components, _ = batch_loss(samples)
            if not torch.isfinite(torch.cat((loss[None], components))).all():
                raise FloatingPointError("non-finite validation loss")
            values = np.concatenate(([loss.item()], components.cpu().numpy()))
            total += len(samples) * values
            count += len(samples)
    return total / count


checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
best_validation_loss = float("inf")
epochs_without_improvement = 0
start_time = time.perf_counter()
with log_path.open("w", encoding="utf-8", newline="") as log_file:
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_coordinate_loss",
            "train_bond_loss",
            "train_velocity_loss",
            "validation_loss",
            "validation_coordinate_loss",
            "validation_bond_loss",
            "validation_velocity_loss",
            "mean_q_weight",
            "elapsed_seconds",
        ],
    )
    writer.writeheader()
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        train_total = np.zeros(4, dtype=np.float64)
        train_q_weight = 0.0
        train_count = 0
        for samples in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, components, weights = batch_loss(samples)
            if not torch.isfinite(torch.cat((loss[None], components))).all():
                raise FloatingPointError(f"epoch {epoch}: non-finite train loss")
            loss.backward()
            if not all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError(f"epoch {epoch}: non-finite gradients")
            optimizer.step()
            values = np.concatenate(([loss.item()], components.detach().cpu().numpy()))
            train_total += len(samples) * values
            train_q_weight += len(samples) * float(
                weights[..., 0].mean().detach()
            )
            train_count += len(samples)
        train_metrics = train_total / train_count
        validation_metrics = evaluate(validation_loader)
        elapsed_seconds = time.perf_counter() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_metrics[0],
            "train_coordinate_loss": train_metrics[1],
            "train_bond_loss": train_metrics[2],
            "train_velocity_loss": train_metrics[3],
            "validation_loss": validation_metrics[0],
            "validation_coordinate_loss": validation_metrics[1],
            "validation_bond_loss": validation_metrics[2],
            "validation_velocity_loss": validation_metrics[3],
            "mean_q_weight": train_q_weight / train_count,
            "elapsed_seconds": elapsed_seconds,
        }
        writer.writerow(row)
        log_file.flush()
        print(row, flush=True)
        if validation_metrics[0] < best_validation_loss:
            best_validation_loss = float(validation_metrics[0])
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "model_state_dict": model.state_dict(),
                    "context_mean": context_mean,
                    "context_std": context_std,
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
