import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from src.equivariant_residual import SEGMENTS, apply_patch_residual, build_bases


parser = argparse.ArgumentParser()
parser.add_argument("--q-cache", type=Path, required=True)
parser.add_argument("--ligand-coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--split", type=Path, required=True)
parser.add_argument("--partition-column", default="partition")
parser.add_argument("--coefficients", type=Path, required=True)
parser.add_argument("--metrics", type=Path, required=True)
args = parser.parse_args()

for output_path in (args.coefficients, args.metrics):
    if output_path.exists():
        raise FileExistsError(output_path)

with args.split.open(encoding="utf-8", newline="") as handle:
    split_rows = list(csv.DictReader(handle))
fit_ids = [
    row["complex_id"]
    for row in split_rows
    if row[args.partition_column] == "fit"
]
validation_ids = [
    row["complex_id"]
    for row in split_rows
    if row[args.partition_column] == "internal_validation"
]
gram = np.zeros((4, 4, 4), dtype=np.float64)
rhs = np.zeros((4, 4), dtype=np.float64)


with h5py.File(args.q_cache, "r") as q_cache, h5py.File(
    args.ligand_coordinate_cache, "r"
) as coordinates, h5py.File(args.graph_cache, "r") as graphs:
    for system_index, complex_id in enumerate(fit_ids, start=1):
        frames = coordinates[complex_id]["ligand_coordinates"][18:100]
        previous, current = frames[:2]
        target = frames[2:]
        prediction = q_cache[complex_id]["q_prediction"][:]
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        system_bases = build_bases(previous, current, prediction, heavy)
        error = target - prediction
        for segment_index, (start, stop) in enumerate(SEGMENTS):
            design = system_bases[start:stop, heavy].reshape(-1, 4).astype(
                np.float64
            )
            response = error[start:stop, heavy].reshape(-1).astype(np.float64)
            gram[segment_index] += design.T @ design
            rhs[segment_index] += design.T @ response
        if system_index % 1000 == 0 or system_index == len(fit_ids):
            print(f"fit accumulated {system_index}/{len(fit_ids)}", flush=True)

    fitted_coefficients = np.stack(
        [np.linalg.solve(gram[index], rhs[index]) for index in range(4)]
    )
    coefficients = np.zeros((4, 4), dtype=np.float64)
    coefficients[:, 2:] = fitted_coefficients[:, 2:]
    baseline_mean = []
    baseline_final = []
    corrected_mean = []
    corrected_final = []
    for system_index, complex_id in enumerate(validation_ids, start=1):
        frames = coordinates[complex_id]["ligand_coordinates"][18:100]
        previous, current = frames[:2]
        target = frames[2:]
        prediction = q_cache[complex_id]["q_prediction"][:]
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        corrected = apply_patch_residual(
            previous, current, prediction, heavy, coefficients
        )
        if not np.isfinite(corrected).all():
            raise FloatingPointError(f"{complex_id}: non-finite corrected trajectory")
        baseline_rmsd = np.sqrt(
            np.mean(np.sum((prediction[:, heavy] - target[:, heavy]) ** 2, axis=-1), axis=1)
        )
        corrected_rmsd = np.sqrt(
            np.mean(np.sum((corrected[:, heavy] - target[:, heavy]) ** 2, axis=-1), axis=1)
        )
        baseline_mean.append(float(baseline_rmsd.mean()))
        baseline_final.append(float(baseline_rmsd[-1]))
        corrected_mean.append(float(corrected_rmsd.mean()))
        corrected_final.append(float(corrected_rmsd[-1]))
        if system_index % 200 == 0 or system_index == len(validation_ids):
            print(
                f"internal validation {system_index}/{len(validation_ids)}",
                flush=True,
            )

args.coefficients.parent.mkdir(parents=True, exist_ok=True)
with args.coefficients.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "segment",
            "q_internal_deformation",
            "observed_internal_velocity",
        ]
    )
    for index, values in enumerate(coefficients[:, 2:], start=1):
        writer.writerow([index, *values])

metrics = {
    "fit_system_count": len(fit_ids),
    "internal_validation_system_count": len(validation_ids),
    "baseline_mean_rmsd": float(np.mean(baseline_mean)),
    "corrected_mean_rmsd": float(np.mean(corrected_mean)),
    "baseline_final_rmsd": float(np.mean(baseline_final)),
    "corrected_final_rmsd": float(np.mean(corrected_final)),
}
with args.metrics.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=metrics.keys())
    writer.writeheader()
    writer.writerow(metrics)
print(metrics)
