import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--partition", type=Path, required=True)
parser.add_argument("--coordinates", type=Path, required=True)
parser.add_argument("--graphs", type=Path, required=True)
parser.add_argument("--per-system", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()
if args.per_system.exists() or args.summary.exists():
    raise FileExistsError("output already exists")

with args.manifest.open(encoding="utf-8", newline="") as handle:
    train_ids = {
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["split"] == "train"
    }
with args.partition.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["partition"] == "fit" and row["complex_id"] in train_ids
    ]


def nearest_distances(frames, pocket):
    return np.stack(
        [
            np.min(
                np.linalg.norm(frame[:, None] - pocket[None], axis=-1),
                axis=1,
            )
            for frame in frames
        ]
    )


def correlation(velocity, lag):
    return float(
        np.mean(np.sum(velocity[:-lag] * velocity[lag:], axis=-1))
        / np.mean(np.sum(velocity[:-lag] ** 2, axis=-1))
    )


rows = []
skipped = []
with h5py.File(args.coordinates, "r") as coordinates, h5py.File(
    args.graphs, "r"
) as graphs:
    for index, complex_id in enumerate(ids):
        graph = graphs[complex_id]
        ligand_heavy = graph["ligand_atomic_numbers"][:] != 1
        protein_heavy = graph["protein_atomic_numbers"][:] != 1
        if not np.any(protein_heavy):
            skipped.append(complex_id)
            continue
        trajectory = coordinates[complex_id]["ligand_coordinates"][:20][
            :, ligand_heavy
        ]
        pocket = graph["protein_coordinates"][:][protein_heavy]
        observed_distance = nearest_distances(trajectory[:10], pocket)
        future_distance = nearest_distances(trajectory[10:20], pocket)
        observed_velocity = np.diff(trajectory[:10], axis=0)
        future_velocity = np.diff(trajectory[9:20], axis=0)
        rows.append(
            {
                "complex_id": complex_id,
                "holdout": int(index % 5 == 0),
                "observed_contact_mean": float(np.mean(observed_distance)),
                "future_contact_mean": float(np.mean(future_distance)),
                "observed_future_contact_w1": float(
                    np.mean(
                        np.abs(
                            np.sort(observed_distance, axis=None)
                            - np.sort(future_distance, axis=None)
                        )
                    )
                ),
                "observed_lag2": correlation(observed_velocity, 2),
                "future_lag2": correlation(future_velocity, 2),
                "observed_lag4": correlation(observed_velocity, 4),
                "future_lag4": correlation(future_velocity, 4),
            }
        )
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(ids)}", flush=True)

args.per_system.parent.mkdir(parents=True, exist_ok=True)
with args.per_system.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

holdout = np.asarray([row["holdout"] for row in rows], dtype=bool)
fit = ~holdout
observed_contact = np.asarray([row["observed_contact_mean"] for row in rows])
future_contact = np.asarray([row["future_contact_mean"] for row in rows])
design = np.column_stack((np.ones(np.sum(fit)), observed_contact[fit]))
intercept, slope = np.linalg.lstsq(design, future_contact[fit], rcond=None)[0]
contact_prediction = intercept + slope * observed_contact[holdout]
summary_rows = [
    {
        "metric": "contact_mean",
        "fit_count": int(np.sum(fit)),
        "holdout_count": int(np.sum(holdout)),
        "pearson": float(
            np.corrcoef(observed_contact[holdout], future_contact[holdout])[0, 1]
        ),
        "model_mae": float(
            np.mean(np.abs(contact_prediction - future_contact[holdout]))
        ),
        "constant_mae": float(
            np.mean(
                np.abs(
                    np.median(future_contact[fit]) - future_contact[holdout]
                )
            )
        ),
        "persistence_mae": float(
            np.mean(np.abs(observed_contact[holdout] - future_contact[holdout]))
        ),
        "target_value": "",
        "observed_future_w1": float(
            np.mean([row["observed_future_contact_w1"] for row in rows if row["holdout"]])
        ),
    }
]
for lag in (2, 4):
    observed = np.asarray([row[f"observed_lag{lag}"] for row in rows])
    future = np.asarray([row[f"future_lag{lag}"] for row in rows])
    target = float(np.median(future[fit]))
    summary_rows.append(
        {
            "metric": f"lag{lag}",
            "fit_count": int(np.sum(fit)),
            "holdout_count": int(np.sum(holdout)),
            "pearson": float(np.corrcoef(observed[holdout], future[holdout])[0, 1]),
            "model_mae": "",
            "constant_mae": float(np.mean(np.abs(target - future[holdout]))),
            "persistence_mae": float(
                np.mean(np.abs(observed[holdout] - future[holdout]))
            ),
            "target_value": target,
            "observed_future_w1": "",
        }
    )

with args.summary.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0])
    writer.writeheader()
    writer.writerows(summary_rows)
print(summary_rows)
print({"skipped_empty_pocket": skipped})
