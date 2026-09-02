import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .base_loss import rollout_loss
from .base_model import BaseTrajectoryModel
from .trajectory_data import TrajectoryRolloutDataset, collate_trajectory_rollout


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-root", type=Path, default=Path("."))
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
checkpoint_path = args.output_root / config.get("outputs", {}).get(
    "checkpoint", "reproduced_artifacts/base_trajectory_model.pt"
)
log_path = args.output_root / config.get("outputs", {}).get(
    "log", "reproduced_results/base_training.csv"
)
if checkpoint_path.exists() or log_path.exists():
    raise FileExistsError("base dynamics training output already exists")

seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rollout_steps = int(config["training"]["rollout_steps"])
train_dataset = TrajectoryRolloutDataset(
    config["data"]["md"],
    config["data"]["train_graph_cache"],
    config["data"]["manifest"],
    "train",
    seed,
    rollout_steps,
    ligand_coordinate_cache_path=config["data"].get("train_ligand_coordinates"),
)
validation_dataset = TrajectoryRolloutDataset(
    config["data"]["md"],
    config["data"]["validation_graph_cache"],
    config["data"]["manifest"],
    "validation",
    seed,
    rollout_steps,
    fixed_frame_index=int(config["validation"]["current_frame_index"]),
    ligand_coordinate_cache_path=config["data"].get(
        "validation_ligand_coordinates"
    ),
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=int(config["training"]["validation_batch_size"]),
    shuffle=False,
    num_workers=int(config["training"]["num_workers"]),
    collate_fn=collate_trajectory_rollout,
)

device = torch.device("cuda")
model = BaseTrajectoryModel(
    hidden_dim=int(config["model"]["hidden_dim"]),
    layer_count=int(config["model"]["layer_count"]),
).to(device)
if "initialization" in config:
    source_checkpoint = torch.load(
        config["initialization"]["checkpoint"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(source_checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
)
loss_weights = np.asarray(
    [
        float(config["loss"]["coordinate_weight"]),
        float(config["loss"]["bond_weight"]),
        float(config["loss"]["angle_weight"]),
    ],
    dtype=np.float64,
)


def calculate_losses(batch):
    batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    _, coordinate, bond, angle = rollout_loss(
        model,
        batch,
        float(config["graph"]["cross_cutoff_angstrom"]),
        *loss_weights,
    )
    counts = np.asarray(
        [
            len(batch["ligand_coordinates"]),
            batch["bond_index"].shape[1],
            batch["angle_index"].shape[1],
        ],
        dtype=np.float64,
    )
    return (coordinate, bond, angle), counts


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
            "train_angle_loss",
            "validation_loss",
            "validation_coordinate_loss",
            "validation_bond_loss",
            "validation_angle_loss",
            "elapsed_seconds",
        ],
    )
    writer.writeheader()

    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["micro_batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_trajectory_rollout,
        )
        accumulation_steps = int(
            config["training"]["gradient_accumulation_steps"]
        )

        model.train()
        train_total = np.zeros(4, dtype=np.float64)
        train_weight = 0
        micro_batches = []
        for batch_index, batch in enumerate(train_loader, start=1):
            micro_batches.append(batch)
            if len(micro_batches) < accumulation_steps and batch_index < len(
                train_loader
            ):
                continue

            component_counts = np.asarray(
                [
                    [
                        len(item["ligand_coordinates"]),
                        item["bond_index"].shape[1],
                        item["angle_index"].shape[1],
                    ]
                    for item in micro_batches
                ],
                dtype=np.float64,
            )
            count_totals = component_counts.sum(axis=0)
            optimizer.zero_grad(set_to_none=True)
            component_values = np.zeros(3, dtype=np.float64)
            for item, counts in zip(micro_batches, component_counts):
                losses, measured_counts = calculate_losses(item)
                if not np.array_equal(counts, measured_counts):
                    raise RuntimeError("micro-batch component count mismatch")
                if not torch.isfinite(torch.stack(losses)).all():
                    raise FloatingPointError(
                        f"epoch {epoch} batch {batch_index}: non-finite loss"
                    )
                fractions = counts / count_totals
                scaled_loss = sum(
                    weight * fraction * loss
                    for weight, fraction, loss in zip(
                        loss_weights, fractions, losses
                    )
                )
                scaled_loss.backward()
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise FloatingPointError(
                        f"epoch {epoch} batch {batch_index}: non-finite gradients"
                    )
                component_values += fractions * np.asarray(
                    [loss.detach().item() for loss in losses]
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            effective_loss = float(np.sum(loss_weights * component_values))
            train_total += count_totals[0] * np.concatenate(
                ([effective_loss], component_values)
            )
            train_weight += count_totals[0]
            micro_batches = []

        model.eval()
        validation_total = np.zeros(4, dtype=np.float64)
        validation_weight = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader, start=1):
                losses, counts = calculate_losses(batch)
                if not torch.isfinite(torch.stack(losses)).all():
                    raise FloatingPointError(
                        f"validation batch {batch_index}: non-finite loss"
                    )
                component_values = np.asarray(
                    [loss.item() for loss in losses], dtype=np.float64
                )
                total_value = float(np.sum(loss_weights * component_values))
                validation_total += counts[0] * np.concatenate(
                    ([total_value], component_values)
                )
                validation_weight += counts[0]

        train_metrics = train_total / train_weight
        validation_metrics = validation_total / validation_weight
        elapsed_seconds = time.perf_counter() - start_time
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics[0],
                "train_coordinate_loss": train_metrics[1],
                "train_bond_loss": train_metrics[2],
                "train_angle_loss": train_metrics[3],
                "validation_loss": validation_metrics[0],
                "validation_coordinate_loss": validation_metrics[1],
                "validation_bond_loss": validation_metrics[2],
                "validation_angle_loss": validation_metrics[3],
                "elapsed_seconds": elapsed_seconds,
            }
        )
        log_file.flush()
        print(
            f"epoch {epoch}: train={train_metrics[0]:.6f} "
            f"validation={validation_metrics[0]:.6f} "
            f"elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )

        if validation_metrics[0] < best_validation_loss:
            best_validation_loss = float(validation_metrics[0])
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "source_checkpoint": config.get("initialization", {}).get(
                        "checkpoint"
                    ),
                    "model_state_dict": model.state_dict(),
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
        if elapsed_seconds >= 3600 * float(
            config["training"]["max_wall_hours"]
        ):
            break
