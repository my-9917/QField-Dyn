import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset-split", required=True)
parser.add_argument("--internal-split", type=Path)
parser.add_argument("--partition", default="internal_validation")
parser.add_argument("--coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--mapping", type=Path, required=True)
parser.add_argument("--alphas", type=float, nargs="+", required=True)
parser.add_argument("--candidates", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True)
args = parser.parse_args()
if len(args.alphas) != len(args.candidates):
    raise ValueError("alphas and candidates differ")
if args.output.exists() or args.selection.exists():
    raise FileExistsError("output already exists")

with args.mapping.open(encoding="utf-8", newline="") as handle:
    mapping = next(csv.DictReader(handle))
intercept = float(mapping["intercept"])
slope = float(mapping["slope"])
with args.manifest.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["split"] == args.dataset_split
    ]
if args.internal_split is not None:
    with args.internal_split.open(encoding="utf-8", newline="") as handle:
        selected_ids = {
            row["complex_id"]
            for row in csv.DictReader(handle)
            if row["partition"] == args.partition
        }
    ids = [complex_id for complex_id in ids if complex_id in selected_ids]

candidate_files = [h5py.File(path, "r") for path in args.candidates]
selection_rows = []
args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs, h5py.File(args.output, "w") as output:
    for index, complex_id in enumerate(ids, start=1):
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        observed = coordinates[complex_id]["ligand_coordinates"][:80, heavy]
        observed_velocity = np.diff(observed, axis=0)
        observed_correlation = np.mean(
            np.sum(observed_velocity[:-1] * observed_velocity[1:], axis=-1)
        ) / np.mean(np.sum(observed_velocity[:-1] ** 2, axis=-1))
        target = intercept + slope * observed_correlation
        candidate_correlations = []
        for candidate_file in candidate_files:
            prediction = candidate_file[complex_id][:, heavy]
            velocity = np.diff(prediction, axis=0)
            candidate_correlations.append(
                np.mean(np.sum(velocity[:-1] * velocity[1:], axis=-1))
                / np.mean(np.sum(velocity[:-1] ** 2, axis=-1))
            )
        choice = int(
            np.argmin(np.abs(np.asarray(candidate_correlations) - target))
        )
        output.create_dataset(
            complex_id,
            data=candidate_files[choice][complex_id][:],
            compression="lzf",
        )
        selection_rows.append(
            {
                "complex_id": complex_id,
                "observed_correlation": observed_correlation,
                "target_correlation": target,
                "selected_alpha": args.alphas[choice],
                "predicted_correlation": candidate_correlations[choice],
            }
        )
        if index % 100 == 0 or index == len(ids):
            print(f"selected {index}/{len(ids)}", flush=True)

for candidate_file in candidate_files:
    candidate_file.close()
with args.selection.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=selection_rows[0])
    writer.writeheader()
    writer.writerows(selection_rows)
print(
    {
        alpha: sum(row["selected_alpha"] == alpha for row in selection_rows)
        for alpha in args.alphas
    }
)