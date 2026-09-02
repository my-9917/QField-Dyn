import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--internal-split", type=Path, required=True)
parser.add_argument("--partition-column", default="partition")
parser.add_argument("--ligand-coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

with args.internal_split.open(encoding="utf-8", newline="") as handle:
    partitions = {
        row["complex_id"]: row[args.partition_column]
        for row in csv.DictReader(handle)
    }
values = {"fit": [], "internal_validation": []}
velocity_correlations = []
with h5py.File(args.ligand_coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs:
    for complex_id, partition in partitions.items():
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        trajectory = coordinates[complex_id]["ligand_coordinates"][:100, heavy]
        centroids = trajectory.mean(axis=1)
        observed = centroids[:20]
        future = centroids[20:]
        if partition == "fit":
            velocity = np.diff(future, axis=0)
            velocity_correlations.append(
                float(
                    np.mean(np.sum(velocity[:-1] * velocity[1:], axis=-1))
                    / np.mean(np.sum(velocity[:-1] ** 2, axis=-1))
                )
            )
        values[partition].append(
            (
                np.sqrt(
                    np.mean(
                        np.sum((observed - observed.mean(0)) ** 2, axis=1)
                    )
                ),
                np.sqrt(
                    np.mean(np.sum((future - future.mean(0)) ** 2, axis=1))
                ),
            )
        )

fit = np.asarray(values["fit"])
validation = np.asarray(values["internal_validation"])
linear_coefficients = np.linalg.lstsq(
    np.column_stack((np.ones(len(fit)), fit[:, 0])), fit[:, 1], rcond=None
)[0]
log_coefficients = np.linalg.lstsq(
    np.column_stack((np.ones(len(fit)), np.log(fit[:, 0]))),
    np.log(fit[:, 1]),
    rcond=None,
)[0]


def metrics(target, prediction):
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mean_prediction": float(np.mean(prediction)),
    }


linear_prediction = linear_coefficients[0] + linear_coefficients[1] * validation[:, 0]
log_prediction = np.exp(
    log_coefficients[0] + log_coefficients[1] * np.log(validation[:, 0])
)
linear_metrics = metrics(validation[:, 1], linear_prediction)
log_metrics = metrics(validation[:, 1], log_prediction)
row = {
    "fit_system_count": len(fit),
    "validation_system_count": len(validation),
    "linear_intercept": float(linear_coefficients[0]),
    "linear_slope": float(linear_coefficients[1]),
    "linear_validation_mae": linear_metrics["mae"],
    "linear_validation_rmse": linear_metrics["rmse"],
    "linear_validation_mean_prediction": linear_metrics["mean_prediction"],
    "log_intercept": float(log_coefficients[0]),
    "log_slope": float(log_coefficients[1]),
    "log_validation_mae": log_metrics["mae"],
    "log_validation_rmse": log_metrics["rmse"],
    "log_validation_mean_prediction": log_metrics["mean_prediction"],
    "validation_mean_target": float(validation[:, 1].mean()),
    "position_correlation": float(
        1.0 + 2.0 * np.mean(velocity_correlations)
    ),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
print(row)
