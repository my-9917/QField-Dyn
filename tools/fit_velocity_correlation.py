import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--split", type=Path, required=True)
parser.add_argument("--coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

with args.split.open(encoding="utf-8", newline="") as handle:
    fit_ids = {
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["partition"] == "fit"
    }
with args.manifest.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["split"] == "train" and row["complex_id"] in fit_ids
    ]

observed_values = []
future_values = []
with h5py.File(args.coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs:
    for index, complex_id in enumerate(ids, start=1):
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        trajectory = coordinates[complex_id]["ligand_coordinates"][:, heavy]
        observed_velocity = np.diff(trajectory[:80], axis=0)
        future_velocity = np.diff(trajectory[80:100], axis=0)
        observed_values.append(
            np.mean(
                np.sum(
                    observed_velocity[:-1] * observed_velocity[1:], axis=-1
                )
            )
            / np.mean(np.sum(observed_velocity[:-1] ** 2, axis=-1))
        )
        future_values.append(
            np.mean(
                np.sum(future_velocity[:-1] * future_velocity[1:], axis=-1)
            )
            / np.mean(np.sum(future_velocity[:-1] ** 2, axis=-1))
        )
        if index % 1000 == 0 or index == len(ids):
            print(f"measured {index}/{len(ids)}", flush=True)

observed_values = np.asarray(observed_values)
future_values = np.asarray(future_values)
design = np.column_stack((np.ones(len(observed_values)), observed_values))
intercept, slope = np.linalg.lstsq(design, future_values, rcond=None)[0]
prediction = intercept + slope * observed_values
pearson = np.corrcoef(observed_values, future_values)[0, 1]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "sample_count",
            "intercept",
            "slope",
            "pearson",
            "fit_mae",
            "constant_mae",
        ),
    )
    writer.writeheader()
    writer.writerow(
        {
            "sample_count": len(ids),
            "intercept": intercept,
            "slope": slope,
            "pearson": pearson,
            "fit_mae": np.mean(np.abs(prediction - future_values)),
            "constant_mae": np.mean(
                np.abs(future_values - future_values.mean())
            ),
        }
    )