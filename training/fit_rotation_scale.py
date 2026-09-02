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


def fitted_rotations(source, target):
    covariance = np.einsum("fai,faj->fij", source, target)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    reflected = np.linalg.det(rotation) < 0
    left[reflected, :, -1] *= -1
    return left @ right_t


def mode_scales(trajectory):
    centered = trajectory - trajectory.mean(axis=1, keepdims=True)
    reference = centered[19]
    source = np.broadcast_to(reference, centered.shape)
    rotation = fitted_rotations(source, centered)
    rigid = np.einsum("ai,fij->faj", reference, rotation)
    internal = centered - rigid
    rigid_scale = np.sqrt(
        np.mean(np.sum((rigid - rigid.mean(axis=0)) ** 2, axis=-1))
    )
    internal_scale = np.sqrt(np.mean(np.sum(internal**2, axis=-1)))
    return rigid_scale, internal_scale


with args.internal_split.open(encoding="utf-8", newline="") as handle:
    partitions = {
        row["complex_id"]: row[args.partition_column]
        for row in csv.DictReader(handle)
    }

values = {"fit": [], "internal_validation": []}
shape_energy = 0.0
rotation_energy = 0.0
internal_energy = 0.0
with h5py.File(args.ligand_coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs:
    for complex_id, partition in partitions.items():
        heavy = graphs[complex_id]["ligand_atomic_numbers"][:] != 1
        trajectory = coordinates[complex_id]["ligand_coordinates"][:100, heavy]
        observed_rotation, observed_internal = mode_scales(trajectory[:20])
        future_rotation, future_internal = mode_scales(trajectory[20:])
        values[partition].append(
            (
                observed_rotation,
                future_rotation,
                observed_internal,
                future_internal,
            )
        )

        if partition == "fit":
            future = trajectory[20:]
            centered = future - future.mean(axis=1, keepdims=True)
            source = centered[:-1]
            target = centered[1:]
            rotation = fitted_rotations(source, target)
            rigid_target = np.einsum("fai,fij->faj", source, rotation)
            shape_step = target - source
            rotation_step = rigid_target - source
            internal_step = target - rigid_target
            shape_energy += float(np.sum(shape_step**2))
            rotation_energy += float(np.sum(rotation_step**2))
            internal_energy += float(np.sum(internal_step**2))

fit = np.asarray(values["fit"])
validation = np.asarray(values["internal_validation"])
row = {
    "fit_system_count": len(fit),
    "validation_system_count": len(validation),
    "future_shape_rotation_energy_fraction": rotation_energy / shape_energy,
    "future_shape_internal_energy_fraction": internal_energy / shape_energy,
}
for name, observed_column, future_column in (
    ("rotation", 0, 1),
    ("internal", 2, 3),
):
    observed_fit = fit[:, observed_column]
    future_fit = fit[:, future_column]
    observed_validation = validation[:, observed_column]
    future_validation = validation[:, future_column]
    design = np.column_stack((np.ones(len(fit)), np.log(observed_fit)))
    intercept, slope = np.linalg.lstsq(
        design, np.log(future_fit), rcond=None
    )[0]
    prediction = np.exp(intercept + slope * np.log(observed_validation))
    row.update(
        {
            f"{name}_fit_pearson": float(
                np.corrcoef(observed_fit, future_fit)[0, 1]
            ),
            f"{name}_observed_mean": float(observed_validation.mean()),
            f"{name}_future_mean": float(future_validation.mean()),
            f"{name}_log_intercept": float(intercept),
            f"{name}_log_slope": float(slope),
            f"{name}_validation_mae": float(
                np.mean(np.abs(prediction - future_validation))
            ),
            f"{name}_validation_mean_prediction": float(prediction.mean()),
        }
    )

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
print(row)
