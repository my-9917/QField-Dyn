import argparse
import csv
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from src.clash import build_graph_clash_physics, clash_counts


TASKS = {
    "T1": (10, 10),
    "T2": (80, 20),
    "T3": (20, 80),
}


def pair_distances(coordinates):
    atom_i, atom_j = np.triu_indices(coordinates.shape[1], k=1)
    return np.linalg.norm(
        coordinates[:, atom_i] - coordinates[:, atom_j], axis=-1
    )


def radius_of_gyration(coordinates):
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))


def rmsf(coordinates):
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered**2, axis=-1)))


def nearest_distances(coordinates, pocket):
    return np.stack(
        [
            np.min(
                np.linalg.norm(frame[:, None] - pocket[None], axis=-1),
                axis=1,
            )
            for frame in coordinates
        ]
    )


def angle_triplets(atom_count, bonds):
    neighbors = [[] for _ in range(atom_count)]
    for atom_i, atom_j in bonds:
        neighbors[atom_i].append(atom_j)
        neighbors[atom_j].append(atom_i)
    return np.asarray(
        [
            (atom_i, center, atom_j)
            for center, atoms in enumerate(neighbors)
            for atom_i, atom_j in combinations(atoms, 2)
        ],
        dtype=np.int64,
    )


def angles(coordinates, triplets):
    left = coordinates[:, triplets[:, 0]] - coordinates[:, triplets[:, 1]]
    right = coordinates[:, triplets[:, 2]] - coordinates[:, triplets[:, 1]]
    cosine = np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def velocity_correlation(coordinates):
    velocity = np.diff(coordinates, axis=0)
    if len(velocity) < 2:
        return 0.0
    scale = np.mean(np.sum(velocity[:-1] ** 2, axis=-1))
    if scale == 0.0:
        return 0.0
    return float(
        np.mean(np.sum(velocity[:-1] * velocity[1:], axis=-1)) / scale
    )


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset-split", default="validation")
parser.add_argument("--internal-split", type=Path)
parser.add_argument("--partition-column", default="partition")
parser.add_argument("--partition", default="internal_validation")
parser.add_argument("--ligand-coordinate-cache", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--predictions", type=Path)
parser.add_argument(
    "--prediction-mode", choices=("stored", "static", "linear"), required=True
)
parser.add_argument("--model", required=True)
parser.add_argument("--task", choices=tuple(TASKS), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--per-complex", type=Path, required=True)
args = parser.parse_args()

if args.output.exists() or args.per_complex.exists():
    raise FileExistsError("joint evaluation output already exists")
if args.prediction_mode == "stored" and args.predictions is None:
    raise ValueError("stored mode requires --predictions")

with args.manifest.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["split"] == args.dataset_split
    ]
if args.internal_split is not None:
    with args.internal_split.open(encoding="utf-8", newline="") as handle:
        partition_ids = {
            row["complex_id"]
            for row in csv.DictReader(handle)
            if row[args.partition_column] == args.partition
        }
    ids = [complex_id for complex_id in ids if complex_id in partition_ids]

observed_count, predicted_count = TASKS[args.task]
stored = h5py.File(args.predictions, "r") if args.predictions else None
rows = []
with h5py.File(args.ligand_coordinate_cache, "r") as coordinates, h5py.File(
    args.graph_cache, "r"
) as graphs:
    for system_index, complex_id in enumerate(ids, start=1):
        graph = graphs[complex_id]
        trajectory = coordinates[complex_id]["ligand_coordinates"][:]
        observed = trajectory[:observed_count]
        truth = trajectory[observed_count : observed_count + predicted_count]
        if args.prediction_mode == "stored":
            prediction = stored[complex_id][:]
        elif args.prediction_mode == "static":
            prediction = np.repeat(observed[-1][None], predicted_count, axis=0)
        else:
            horizon = np.arange(1, predicted_count + 1)[:, None, None]
            prediction = observed[-1][None] + horizon * (
                observed[-1] - observed[-2]
            )[None]
        if prediction.shape != truth.shape:
            raise ValueError(
                f"{complex_id}: prediction {prediction.shape}, truth {truth.shape}"
            )
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"{complex_id}: non-finite prediction")

        ligand_z = graph["ligand_atomic_numbers"][:]
        protein_z = graph["protein_atomic_numbers"][:]
        bonds = graph["ligand_bonds"][:]
        heavy = ligand_z != 1
        protein_heavy = protein_z != 1
        pocket = graph["protein_coordinates"][:][protein_heavy]
        pred_heavy = prediction[:, heavy]
        truth_heavy = truth[:, heavy]

        frame_rmsd = np.sqrt(
            np.mean(np.sum((pred_heavy - truth_heavy) ** 2, axis=-1), axis=1)
        )
        pred_pairs = pair_distances(pred_heavy)
        truth_pairs = pair_distances(truth_heavy)
        pred_rg = radius_of_gyration(pred_heavy)
        truth_rg = radius_of_gyration(truth_heavy)

        pred_contact_distance = nearest_distances(pred_heavy, pocket)
        truth_contact_distance = nearest_distances(truth_heavy, pocket)
        pred_contact = np.mean(pred_contact_distance <= 4.5, axis=1)
        truth_contact = np.mean(truth_contact_distance <= 4.5, axis=1)
        pred_speed = np.linalg.norm(np.diff(pred_heavy, axis=0), axis=-1)
        truth_speed = np.linalg.norm(np.diff(truth_heavy, axis=0), axis=-1)
        pred_centroid = pred_heavy.mean(axis=1)
        truth_centroid = truth_heavy.mean(axis=1)

        reference = observed[-1]
        reference_bonds = np.linalg.norm(
            reference[bonds[:, 0]] - reference[bonds[:, 1]], axis=-1
        )
        predicted_bonds = np.linalg.norm(
            prediction[:, bonds[:, 0]] - prediction[:, bonds[:, 1]], axis=-1
        )
        triplets = angle_triplets(len(ligand_z), bonds)
        reference_angles = angles(reference[None], triplets)[0]
        predicted_angles = angles(prediction, triplets)
        physics = build_graph_clash_physics(protein_z, ligand_z, bonds)
        internal_clash, protein_clash = clash_counts(
            prediction, pocket, physics
        )
        reference_internal_clash, reference_protein_clash = clash_counts(
            reference[None], pocket, physics
        )

        internal_error = np.mean(np.abs(pred_pairs - truth_pairs), axis=1)
        rg_error = np.abs(pred_rg - truth_rg)
        pred_rmsf = rmsf(pred_heavy)
        truth_rmsf = rmsf(truth_heavy)
        pred_mean_speed = float(np.mean(pred_speed))
        truth_mean_speed = float(np.mean(truth_speed))
        pred_transition = float(np.mean(np.abs(np.diff(pred_contact))))
        truth_transition = float(np.mean(np.abs(np.diff(truth_contact))))
        reference_rg = float(radius_of_gyration(reference[None, heavy])[0])
        rows.append(
            {
                "complex_id": complex_id,
                "geo_mean_rmsd": float(np.mean(frame_rmsd)),
                "geo_final_rmsd": float(frame_rmsd[-1]),
                "geo_rmsd_auc": float(
                    np.trapezoid(frame_rmsd) / (predicted_count - 1)
                ),
                "phys_mean_bond_drift": float(
                    np.mean(np.abs(predicted_bonds - reference_bonds))
                ),
                "phys_mean_angle_drift_deg": float(
                    np.mean(np.abs(predicted_angles - reference_angles))
                ),
                "phys_internal_clash_frame_ratio": float(
                    np.mean(internal_clash > 0)
                ),
                "phys_protein_clash_frame_ratio": float(
                    np.mean(protein_clash > 0)
                ),
                "phys_internal_clash_excess_mean": float(
                    np.mean(np.maximum(internal_clash - reference_internal_clash[0], 0))
                ),
                "phys_protein_clash_excess_mean": float(
                    np.mean(np.maximum(protein_clash - reference_protein_clash[0], 0))
                ),
                "phys_clash_nonincrease_frame_ratio": float(
                    np.mean(
                        (internal_clash <= reference_internal_clash[0])
                        & (protein_clash <= reference_protein_clash[0])
                    )
                ),
                "dyn_contact_w1": float(
                    np.mean(
                        np.abs(
                            np.sort(pred_contact_distance, axis=1)
                            - np.sort(truth_contact_distance, axis=1)
                        )
                    )
                ),
                "dyn_contact_fraction_abs_error": float(
                    np.mean(np.abs(pred_contact - truth_contact))
                ),
                "dyn_pred_rmsf": float(pred_rmsf),
                "dyn_true_rmsf": float(truth_rmsf),
                "dyn_rmsf_abs_error": float(abs(pred_rmsf - truth_rmsf)),
                "dyn_pred_mean_speed": pred_mean_speed,
                "dyn_true_mean_speed": truth_mean_speed,
                "dyn_step_speed_w1": float(
                    np.mean(
                        np.abs(
                            np.sort(pred_speed, axis=None)
                            - np.sort(truth_speed, axis=None)
                        )
                    )
                ),
                "dyn_pred_contact_transition": pred_transition,
                "dyn_true_contact_transition": truth_transition,
                "dyn_contact_transition_abs_error": float(
                    abs(pred_transition - truth_transition)
                ),
                "dyn_velocity_correlation_abs_error": float(
                    abs(
                        velocity_correlation(pred_heavy)
                        - velocity_correlation(truth_heavy)
                    )
                ),
                "dyn_centroid_final_msd_abs_error": float(
                    abs(
                        np.sum((pred_centroid[-1] - pred_centroid[0]) ** 2)
                        - np.sum((truth_centroid[-1] - truth_centroid[0]) ** 2)
                    )
                ),
                "stab_mean_internal_distance_mae": float(
                    np.mean(internal_error)
                ),
                "stab_final_internal_distance_mae": float(internal_error[-1]),
                "stab_mean_rg_abs_error": float(np.mean(rg_error)),
                "stab_final_rg_abs_error": float(rg_error[-1]),
                "stab_rmsd_growth": float(frame_rmsd[-1] - frame_rmsd[0]),
                "stab_min_rg_ratio": float(np.min(pred_rg) / reference_rg),
                "stab_max_rg_ratio": float(np.max(pred_rg) / reference_rg),
                "stab_final_centroid_displacement": float(
                    np.linalg.norm(pred_centroid[-1] - pred_centroid[0])
                ),
            }
        )
        if system_index % 100 == 0 or system_index == len(ids):
            print(f"evaluated {system_index}/{len(ids)}", flush=True)

if stored is not None:
    stored.close()

aggregate = {
    "model": args.model,
    "task": args.task,
    "system_count": len(rows),
}
for key in rows[0]:
    if key != "complex_id":
        aggregate[key] = float(np.mean([row[key] for row in rows]))
aggregate["anti_rmsf_ratio"] = (
    aggregate["dyn_pred_rmsf"] / aggregate["dyn_true_rmsf"]
)
aggregate["anti_step_speed_ratio"] = (
    aggregate["dyn_pred_mean_speed"] / aggregate["dyn_true_mean_speed"]
)
aggregate["anti_contact_transition_ratio"] = (
    aggregate["dyn_pred_contact_transition"]
    / aggregate["dyn_true_contact_transition"]
)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.per_complex.parent.mkdir(parents=True, exist_ok=True)
with args.per_complex.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=aggregate.keys())
    writer.writeheader()
    writer.writerow(aggregate)
print(aggregate)

