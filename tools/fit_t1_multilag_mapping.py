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


def correlation(velocity, lag):
    return np.mean(
        np.sum(velocity[:-lag] * velocity[lag:], axis=-1)
    ) / np.mean(np.sum(velocity[:-lag] ** 2, axis=-1))


lags = (1, 2, 4, 8)
observed = {lag: [] for lag in lags}
future = {lag: [] for lag in lags}
with h5py.File(args.coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs:
    for complex_id in ids:
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        trajectory = coordinates[complex_id]["ligand_coordinates"][:, heavy]
        observed_velocity = np.diff(trajectory[:10], axis=0)
        future_velocity = np.diff(trajectory[9:20], axis=0)
        for lag in lags:
            observed[lag].append(correlation(observed_velocity, lag))
            future[lag].append(correlation(future_velocity, lag))

rows = []
for lag in lags:
    observed_values = np.asarray(observed[lag])
    future_values = np.asarray(future[lag])
    design = np.column_stack((np.ones(len(observed_values)), observed_values))
    intercept, slope = np.linalg.lstsq(design, future_values, rcond=None)[0]
    prediction = intercept + slope * observed_values
    rows.append(
        {
            "lag": lag,
            "sample_count": len(ids),
            "intercept": intercept,
            "slope": slope,
            "pearson": np.corrcoef(observed_values, future_values)[0, 1],
            "fit_mae": np.mean(np.abs(prediction - future_values)),
            "constant_mae": np.mean(
                np.abs(future_values - future_values.mean())
            ),
        }
    )

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
