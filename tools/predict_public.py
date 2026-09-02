import argparse
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src import QFieldDynPredictor
from src.clash import build_graph_clash_physics, clash_counts
from src.rotation_projection import project_rotation_without_new_clash
from src.translation_projection import project_translation_without_new_clash
from tools.public_io import (
    _read_xtc,
    load_public_system,
    rigid_fit,
    write_public_prediction,
)


def predict_t4(system, seed):
    observed = system["observed_ligand_coordinates"]
    heavy = system["ligand_atomic_numbers"] != 1
    centroids = observed[:, heavy].mean(axis=1)
    translation_increments = np.diff(centroids, axis=0)
    translation_innovations = (
        translation_increments - translation_increments.mean(axis=0)
    )
    centered_centroids = centroids - centroids.mean(axis=0)
    centroid_rho = np.sum(
        centered_centroids[:-1] * centered_centroids[1:]
    ) / np.sum(centered_centroids[:-1] ** 2)
    mean_reversion = float(np.clip(1.0 - centroid_rho, 0.05, 0.5))

    rotation_increments = []
    for frame_index in range(len(observed) - 1):
        row_rotation, _, _ = rigid_fit(
            observed[frame_index, heavy],
            observed[frame_index + 1, heavy],
        )
        rotation_increments.append(
            Rotation.from_matrix(row_rotation.T).as_rotvec()
        )
    rotation_increments = np.asarray(rotation_increments)

    physics = build_graph_clash_physics(
        system["protein_atomic_numbers"],
        system["ligand_atomic_numbers"],
        system["ligand_bonds"],
    )
    fixed_protein = system["protein_coordinates"][physics["protein_heavy"]]
    projection_physics = dict(physics)
    projection_physics["protein_ligand_threshold"] = (
        physics["protein_ligand_threshold"] + 0.02
    )
    rng = np.random.default_rng(
        seed + zlib.crc32(system["meta"]["id"].encode("utf-8"))
    )
    anchor_centroid = centroids[-1]
    step_count = system["meta"]["n_pred"]
    pair_count = (step_count + 1) // 2
    pair_rotation_choice = rng.integers(
        len(rotation_increments), size=pair_count
    )
    pair_rotation_sign = rng.choice((-1.0, 1.0), size=pair_count)
    pair_translation_choice = rng.integers(
        len(translation_innovations), size=pair_count
    )
    rotation_choice = np.repeat(pair_rotation_choice, 2)[:step_count]
    rotation_sign = np.column_stack(
        (pair_rotation_sign, -pair_rotation_sign)
    ).reshape(-1)[:step_count]
    translation_choice = np.repeat(pair_translation_choice, 2)[:step_count]
    translation_sign = np.tile((1.0, -1.0), pair_count)[:step_count]

    def generate(activity_scale):
        current = observed[-1].copy()
        prediction = []
        for step in range(step_count):
            previous = current.copy()
            rotation_vector = (
                activity_scale
                * rotation_sign[step]
                * rotation_increments[rotation_choice[step]]
            )
            current, _ = project_rotation_without_new_clash(
                current, rotation_vector, fixed_protein, projection_physics
            )
            current_centroid = current[heavy].mean(axis=0)
            displacement = (
                activity_scale
                * translation_sign[step]
                * translation_innovations[translation_choice[step]]
                - mean_reversion * (current_centroid - anchor_centroid)
            )
            current, _ = project_translation_without_new_clash(
                current, displacement, fixed_protein, projection_physics
            )
            _, protein_clash = clash_counts(
                current[None], fixed_protein, physics
            )
            if protein_clash[0] > 0:
                current = previous
            prediction.append(current.copy())
        return np.asarray(prediction)

    pilot = generate(1.0)
    observed_step_speed = np.linalg.norm(
        np.diff(observed[:, heavy], axis=0), axis=-1
    ).mean()
    pilot_step_speed = np.linalg.norm(
        np.diff(
            np.concatenate((observed[-1:, heavy], pilot[:, heavy]), axis=0),
            axis=0,
        ),
        axis=-1,
    ).mean()
    return generate(float(observed_step_speed / pilot_step_speed))


def project_to_nearest_safe_frame(system, prediction):
    ligand_z = system["ligand_atomic_numbers"]
    heavy = ligand_z != 1
    physics = build_graph_clash_physics(
        system["protein_atomic_numbers"],
        ligand_z,
        system["ligand_bonds"],
    )
    fixed_protein = system["protein_coordinates"][physics["protein_heavy"]]
    internal_clash, protein_clash = clash_counts(
        prediction, fixed_protein, physics
    )
    reference_internal_clash, reference_protein_clash = clash_counts(
        system["observed_ligand_coordinates"][-1:],
        fixed_protein,
        physics,
    )
    fixed_environment = system["fixed_environment_coordinates"]
    if len(fixed_environment) == 0:
        environment_clash = np.zeros(len(prediction), dtype=bool)
    else:
        distance = np.linalg.norm(
            prediction[:, :, None] - fixed_environment[None, None], axis=-1
        )
        environment_clash = np.any(distance < 0.4, axis=(1, 2))
    safe = (
        (internal_clash <= reference_internal_clash[0])
        & (protein_clash <= reference_protein_clash[0])
        & ~environment_clash
    )
    if np.all(safe):
        return prediction
    safe_indices = np.flatnonzero(safe)
    if len(safe_indices) == 0:
        raise ValueError(f"{system['meta']['id']}: no safe predicted frame")

    repaired = prediction.copy()
    for frame_index in np.flatnonzero(~safe):
        temporal_distance = np.abs(safe_indices - frame_index)
        candidates = safe_indices[
            temporal_distance == np.min(temporal_distance)
        ]
        difference = np.mean(
            np.sum(
                (
                    prediction[candidates][:, heavy]
                    - prediction[frame_index, heavy]
                )
                ** 2,
                axis=-1,
            ),
            axis=1,
        )
        repaired[frame_index] = prediction[candidates[np.argmin(difference)]]
    return repaired


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--tiers", nargs="+", choices=("T1", "T2", "T3", "T4"), required=True
)
parser.add_argument("--ids", nargs="+")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=20260825)
args = parser.parse_args()

project_directory = Path(__file__).resolve().parents[1]
predictor = QFieldDynPredictor(project_directory, args.device)
for tier in args.tiers:
    ids = (args.data / tier / "ids.txt").read_text(encoding="utf-8").splitlines()
    if args.ids:
        ids = [complex_id for complex_id in ids if complex_id in args.ids]
    for index, complex_id in enumerate(ids, start=1):
        system = load_public_system(args.data, tier, complex_id)
        if tier == "T4":
            prediction = predict_t4(system, args.seed)
            output_path = args.output / tier / f"{complex_id}_pred.xtc"
            raw_output = args.output / tier / f"{complex_id}_raw.xtc"
            write_public_prediction(
                system,
                prediction.astype(np.float32),
                raw_output,
            )
            quantized, _, _ = _read_xtc(raw_output)
            prediction = project_to_nearest_safe_frame(
                system, quantized[:, system["ligand_indices"]]
            )
            write_public_prediction(
                system,
                prediction.astype(np.float32),
                output_path,
            )
            raw_output.unlink()
        else:
            prediction, _ = predictor.predict(
                system["protein_atomic_numbers"],
                system["protein_coordinates"],
                system["ligand_atomic_numbers"],
                system["ligand_bonds"],
                system["ligand_structure_coordinates"],
                system["observed_ligand_coordinates"],
                tier,
            )
            write_public_prediction(
                system,
                prediction.astype(np.float32),
                args.output / tier / f"{complex_id}_pred.xtc",
            )
        print(f"{tier}: predicted {index}/{len(ids)}", flush=True)
