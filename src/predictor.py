import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .clash import build_graph_clash_physics, clash_counts
from .history_quantum_predictor import HistoryQuantumTrajectoryPredictor
from .mean_predictor import MeanTrajectoryPredictor
from .quantum_predictor import QuantumTrajectoryPredictor
from .rotation_projection import project_rotation_without_new_clash
from .stochastic_dynamics import bootstrap_residuals
from .torsion_geometry import graph_data, kinematic_tree, rigid_fit
from .torsion_projection import project_to_torsion_manifold
from .translation_projection import project_translation_without_new_clash


class QFieldDynPredictor:
    def __init__(self, project_directory, device="cuda"):
        project_directory = Path(project_directory)
        artifacts = project_directory / "artifacts"
        with (project_directory / "config" / "model.json").open(
            encoding="utf-8"
        ) as handle:
            parameters = json.load(handle)

        self.mean_predictor = MeanTrajectoryPredictor(
            artifacts / "quantum_encoder.pt",
            artifacts / "quantum_statistics.hdf5",
            artifacts / "trajectory_model.pt",
            artifacts / "expert_gate.pt",
            artifacts / "geometry_coefficients.csv",
            parameters["max_rms_displacement"],
            device,
        )
        self.t1_predictor = QuantumTrajectoryPredictor(
            artifacts / "quantum_encoder.pt",
            artifacts / "quantum_statistics.hdf5",
            artifacts / "trajectory_model_t1.pt",
            device,
        )
        self.t2_predictor = QuantumTrajectoryPredictor(
            artifacts / "quantum_encoder.pt",
            artifacts / "quantum_statistics.hdf5",
            artifacts / "trajectory_model_t2.pt",
            device,
        )
        self.history_t2_predictor = HistoryQuantumTrajectoryPredictor(
            project_directory,
            artifacts / "history_trajectory_t2.pt",
            device,
        )
        self.history_t1_predictor = HistoryQuantumTrajectoryPredictor(
            project_directory,
            artifacts / "history_trajectory_t1.pt",
            device,
        )
        with (artifacts / "velocity_correlation_t1.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            t1_mapping = next(
                row for row in csv.DictReader(handle) if row["lag"] == "1"
            )
        self.t1_correlation_intercept = float(t1_mapping["intercept"])
        self.t1_correlation_slope = float(t1_mapping["slope"])
        with (artifacts / "t1_phase_targets.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            phase_targets = {row["metric"]: row for row in csv.DictReader(handle)}
        self.t1_phase_targets = {
            lag: float(phase_targets[f"lag{lag}"]["target_value"])
            for lag in (2, 4)
        }
        with (artifacts / "velocity_correlation.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            correlation_mapping = next(csv.DictReader(handle))
        self.velocity_correlation_intercept = float(
            correlation_mapping["intercept"]
        )
        self.velocity_correlation_slope = float(
            correlation_mapping["slope"]
        )
        self.position_correlation = parameters["position_correlation"]
        self.translation_log_intercept = parameters[
            "translation_log_intercept"
        ]
        self.translation_log_slope = parameters["translation_log_slope"]
        self.rotation_log_intercept = parameters["rotation_log_intercept"]
        self.rotation_log_slope = parameters["rotation_log_slope"]
        self.seed = parameters["seed"]

    def predict(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
        task,
    ):
        inputs = (
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed_ligand_coordinates,
        )
        if task == "T1":
            return self._predict_short_trajectory(*inputs), None
        if task == "T2":
            return self._predict_medium_trajectory(*inputs), None
        if task == "T3":
            return self._predict_long_trajectory(*inputs)
        raise ValueError(task)

    def _predict_medium_trajectory(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
    ):
        inputs = (
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed_ligand_coordinates,
        )
        baseline = self.mean_predictor.dual_expert.quantum_predictor.predict(
            *inputs, "T2"
        )
        target = self.t2_predictor.predict(*inputs, "T2")
        ligand_z = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        bonds = np.asarray(ligand_bonds, dtype=np.int64)
        heavy = ligand_z != 1
        rotatable, fragment, fragments = graph_data(len(ligand_z), bonds)
        _, order, subtree_atoms = kinematic_tree(
            baseline[0], rotatable, fragment, fragments
        )

        transferred = np.empty_like(target, dtype=np.float64)
        for frame_index, (source, target_frame) in enumerate(
            zip(baseline, target)
        ):
            rotation, translation = rigid_fit(
                source[heavy], target_frame[heavy]
            )
            aligned = source @ rotation + translation
            transferred[frame_index], _ = project_to_torsion_manifold(
                aligned,
                target_frame,
                heavy,
                order,
                subtree_atoms,
            )
        transferred = transferred.astype(np.float32)

        protein_z = np.asarray(protein_atomic_numbers, dtype=np.int64)
        protein_x = np.asarray(protein_coordinates, dtype=np.float32)
        physics = build_graph_clash_physics(protein_z, ligand_z, bonds)
        fixed_protein = protein_x[physics["protein_heavy"]]
        baseline_internal, baseline_protein = clash_counts(
            baseline, fixed_protein, physics
        )
        alphas = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
        candidates = []
        for alpha in alphas:
            candidate = (
                baseline + alpha * (transferred - baseline)
            ).astype(np.float32)
            candidate_internal, candidate_protein = clash_counts(
                candidate, fixed_protein, physics
            )
            accept = (candidate_internal <= baseline_internal) & (
                candidate_protein <= baseline_protein
            )
            candidates.append(
                np.where(accept[:, None, None], candidate, baseline)
            )

        observed = np.asarray(
            observed_ligand_coordinates, dtype=np.float64
        )[:, heavy]
        observed_velocity = np.diff(observed, axis=0)
        observed_correlation = np.mean(
            np.sum(
                observed_velocity[:-1] * observed_velocity[1:], axis=-1
            )
        ) / np.mean(np.sum(observed_velocity[:-1] ** 2, axis=-1))
        correlation_target = (
            self.velocity_correlation_intercept
            + self.velocity_correlation_slope * observed_correlation
        )
        candidate_correlations = []
        for candidate in candidates:
            velocity = np.diff(candidate[:, heavy], axis=0)
            candidate_correlations.append(
                np.mean(np.sum(velocity[:-1] * velocity[1:], axis=-1))
                / np.mean(np.sum(velocity[:-1] ** 2, axis=-1))
            )
        choice = int(
            np.argmin(
                np.abs(
                    np.asarray(candidate_correlations) - correlation_target
                )
            )
        )
        selected = candidates[choice].astype(np.float32)
        history_target = self.history_t2_predictor.predict(*inputs, steps=20)
        _, history_order, history_subtree_atoms = kinematic_tree(
            selected[0], rotatable, fragment, fragments
        )
        history_transferred = np.empty_like(history_target, dtype=np.float64)
        for frame_index, (source, target_frame) in enumerate(
            zip(selected, history_target)
        ):
            rotation, translation = rigid_fit(
                source[heavy], target_frame[heavy]
            )
            aligned = source @ rotation + translation
            history_transferred[frame_index], _ = project_to_torsion_manifold(
                aligned,
                target_frame,
                heavy,
                history_order,
                history_subtree_atoms,
            )
        selected_internal, selected_protein = clash_counts(
            selected, fixed_protein, physics
        )
        history_internal, history_protein = clash_counts(
            history_transferred, fixed_protein, physics
        )
        accept = (history_internal <= selected_internal) & (
            history_protein <= selected_protein
        )
        history_candidate = np.where(
            accept[:, None, None], history_transferred, selected
        ).astype(np.float32)

        history_velocity = np.diff(history_candidate[:, heavy], axis=0)
        history_correlation = np.mean(
            np.sum(
                history_velocity[:-1] * history_velocity[1:], axis=-1
            )
        ) / np.mean(np.sum(history_velocity[:-1] ** 2, axis=-1))
        selected_distance = np.linalg.norm(
            selected[:, heavy, None] - fixed_protein[None, None], axis=-1
        ).min(axis=-1)
        history_distance = np.linalg.norm(
            history_candidate[:, heavy, None] - fixed_protein[None, None],
            axis=-1,
        ).min(axis=-1)
        selected_transition = np.mean(
            np.abs(np.diff(np.mean(selected_distance <= 4.5, axis=1)))
        )
        history_transition = np.mean(
            np.abs(np.diff(np.mean(history_distance <= 4.5, axis=1)))
        )
        if (
            abs(history_correlation - correlation_target)
            < abs(candidate_correlations[choice] - correlation_target)
            and history_transition >= selected_transition
        ):
            return history_candidate
        return selected
    def _predict_short_trajectory(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
    ):
        inputs = (
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed_ligand_coordinates,
        )
        baseline = self.mean_predictor.dual_expert.quantum_predictor.predict(
            *inputs, "T1"
        )
        target = self.t1_predictor.predict(*inputs, "T1")
        ligand_z = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        bonds = np.asarray(ligand_bonds, dtype=np.int64)
        heavy = ligand_z != 1
        rotatable, fragment, fragments = graph_data(len(ligand_z), bonds)
        _, order, subtree_atoms = kinematic_tree(
            baseline[0], rotatable, fragment, fragments
        )

        candidate = np.empty_like(target, dtype=np.float64)
        for frame_index, (source, target_frame) in enumerate(
            zip(baseline, target)
        ):
            rotation, translation = rigid_fit(
                source[heavy], target_frame[heavy]
            )
            aligned = source @ rotation + translation
            candidate[frame_index], _ = project_to_torsion_manifold(
                aligned,
                target_frame,
                heavy,
                order,
                subtree_atoms,
            )

        protein_z = np.asarray(protein_atomic_numbers, dtype=np.int64)
        protein_x = np.asarray(protein_coordinates, dtype=np.float32)
        physics = build_graph_clash_physics(protein_z, ligand_z, bonds)
        fixed_protein = protein_x[physics["protein_heavy"]]
        baseline_internal, baseline_protein = clash_counts(
            baseline, fixed_protein, physics
        )
        candidate_internal, candidate_protein = clash_counts(
            candidate, fixed_protein, physics
        )
        accept = (candidate_internal <= baseline_internal) & (
            candidate_protein <= baseline_protein
        )
        deployable = np.where(
            accept[:, None, None], candidate, baseline
        ).astype(np.float32)

        history_target = self.history_t1_predictor.predict(*inputs, steps=10)
        _, history_order, history_subtree_atoms = kinematic_tree(
            deployable[0], rotatable, fragment, fragments
        )
        history_transferred = np.empty_like(history_target, dtype=np.float64)
        for frame_index, (source, target_frame) in enumerate(
            zip(deployable, history_target)
        ):
            rotation, translation = rigid_fit(
                source[heavy], target_frame[heavy]
            )
            aligned = source @ rotation + translation
            history_transferred[frame_index], _ = project_to_torsion_manifold(
                aligned,
                target_frame,
                heavy,
                history_order,
                history_subtree_atoms,
            )
        deployable_internal, deployable_protein = clash_counts(
            deployable, fixed_protein, physics
        )
        history_internal, history_protein = clash_counts(
            history_transferred, fixed_protein, physics
        )
        history_accept = (history_internal <= deployable_internal) & (
            history_protein <= deployable_protein
        )
        history_candidate = np.where(
            history_accept[:, None, None], history_transferred, deployable
        ).astype(np.float32)

        observed = np.asarray(
            observed_ligand_coordinates, dtype=np.float32
        )[:, heavy]
        observed_velocity = np.diff(observed, axis=0)

        def correlation(velocity, lag=1):
            return np.mean(
                np.sum(velocity[:-lag] * velocity[lag:], axis=-1)
            ) / np.mean(np.sum(velocity[:-lag] ** 2, axis=-1))

        target_correlation = (
            self.t1_correlation_intercept
            + self.t1_correlation_slope * correlation(observed_velocity)
        )
        deployable_velocity = np.diff(
            np.concatenate((observed[-1:], deployable[:, heavy]), axis=0),
            axis=0,
        )
        history_velocity = np.diff(
            np.concatenate(
                (observed[-1:], history_candidate[:, heavy]), axis=0
            ),
            axis=0,
        )
        deployable_distance = np.linalg.norm(
            deployable[:, heavy, None] - fixed_protein[None, None], axis=-1
        ).min(axis=-1)
        history_distance = np.linalg.norm(
            history_candidate[:, heavy, None] - fixed_protein[None, None],
            axis=-1,
        ).min(axis=-1)
        deployable_transition = np.mean(
            np.abs(np.diff(np.mean(deployable_distance <= 4.5, axis=1)))
        )
        history_transition = np.mean(
            np.abs(np.diff(np.mean(history_distance <= 4.5, axis=1)))
        )
        deployable_speed = np.linalg.norm(
            np.diff(deployable[:, heavy], axis=0), axis=-1
        ).mean()
        history_speed = np.linalg.norm(
            np.diff(history_candidate[:, heavy], axis=0), axis=-1
        ).mean()
        deployable_centroid = deployable[:, heavy].mean(axis=1)
        history_centroid = history_candidate[:, heavy].mean(axis=1)
        use_history = (
            abs(correlation(history_velocity) - target_correlation)
            < abs(correlation(deployable_velocity) - target_correlation)
            and history_transition >= deployable_transition
            and history_speed >= deployable_speed
            and np.sum((history_centroid[-1] - history_centroid[0]) ** 2)
            >= np.sum(
                (deployable_centroid[-1] - deployable_centroid[0]) ** 2
            )
        )
        if not use_history:
            return deployable

        shift = (
            deployable[:, heavy].mean(axis=1)
            - history_candidate[:, heavy].mean(axis=1)
        )
        restored = (history_candidate + shift[:, None]).astype(np.float32)
        observed_distance = np.stack(
            [
                np.min(
                    np.linalg.norm(
                        frame[:, None] - fixed_protein[None], axis=-1
                    ),
                    axis=1,
                )
                for frame in observed
            ]
        )
        restored_distance = np.linalg.norm(
            restored[:, heavy, None] - fixed_protein[None, None], axis=-1
        ).min(axis=-1)
        deployable_contact_error = np.mean(
            np.abs(
                np.sort(deployable_distance, axis=None)
                - np.sort(observed_distance, axis=None)
            )
        )
        restored_contact_error = np.mean(
            np.abs(
                np.sort(restored_distance, axis=None)
                - np.sort(observed_distance, axis=None)
            )
        )
        restored_velocity = np.diff(
            np.concatenate((observed[-1:], restored[:, heavy]), axis=0),
            axis=0,
        )
        deployable_phase_error = np.mean(
            [
                abs(
                    correlation(deployable_velocity, lag)
                    - self.t1_phase_targets[lag]
                )
                for lag in (2, 4)
            ]
        )
        restored_phase_error = np.mean(
            [
                abs(
                    correlation(restored_velocity, lag)
                    - self.t1_phase_targets[lag]
                )
                for lag in (2, 4)
            ]
        )
        if (
            restored_contact_error <= deployable_contact_error
            and restored_phase_error <= deployable_phase_error
        ):
            return restored
        return deployable
    def _predict_long_trajectory(
        self,
        protein_atomic_numbers,
        protein_coordinates,
        ligand_atomic_numbers,
        ligand_bonds,
        ligand_structure_coordinates,
        observed_ligand_coordinates,
    ):
        baseline, weights = self.mean_predictor.predict(
            protein_atomic_numbers,
            protein_coordinates,
            ligand_atomic_numbers,
            ligand_bonds,
            ligand_structure_coordinates,
            observed_ligand_coordinates,
        )
        protein_z = np.asarray(protein_atomic_numbers, dtype=np.int64)
        protein_x = np.asarray(protein_coordinates, dtype=np.float32)
        ligand_z = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        bonds = np.asarray(ligand_bonds, dtype=np.int64)
        observed = np.asarray(observed_ligand_coordinates, dtype=np.float32)
        heavy = ligand_z != 1
        physics = build_graph_clash_physics(protein_z, ligand_z, bonds)
        fixed_protein = protein_x[physics["protein_heavy"]]

        centroids = observed[:, heavy].mean(axis=1)
        centered_centroids = centroids - centroids.mean(0)
        observed_scale = np.sqrt(
            np.mean(np.sum(centered_centroids**2, axis=1))
        )
        target_scale = np.exp(
            self.translation_log_intercept
            + self.translation_log_slope * np.log(observed_scale)
        )
        translation = bootstrap_residuals(
            centered_centroids,
            self.position_correlation,
            self.seed,
            len(baseline),
        )
        translation *= target_scale / np.sqrt(
            np.mean(np.sum(translation**2, axis=1))
        )
        translated = np.empty_like(baseline)
        for frame_index, displacement in enumerate(translation):
            translated[frame_index], _ = project_translation_without_new_clash(
                baseline[frame_index], displacement, fixed_protein, physics
            )

        reference = observed[-1, heavy]
        reference_centered = reference - reference.mean(0)
        rotation_vectors = []
        rigid_frames = []
        for frame in observed[:, heavy]:
            rotation, _ = rigid_fit(reference, frame)
            rotation_vectors.append(
                Rotation.from_matrix(rotation.T).as_rotvec()
            )
            rigid_frames.append(reference_centered @ rotation)
        rotation_vectors = np.asarray(rotation_vectors)
        rotation_vectors -= rotation_vectors.mean(0)
        rigid_frames = np.asarray(rigid_frames)
        observed_scale = np.sqrt(
            np.mean(
                np.sum((rigid_frames - rigid_frames.mean(0)) ** 2, axis=-1)
            )
        )
        target_scale = np.exp(
            self.rotation_log_intercept
            + self.rotation_log_slope * np.log(observed_scale)
        )
        rotation_residuals = bootstrap_residuals(
            rotation_vectors,
            self.position_correlation,
            self.seed,
            len(baseline),
        )
        generated_rigid = np.asarray(
            [
                Rotation.from_rotvec(vector).apply(reference_centered)
                for vector in rotation_residuals
            ]
        )
        generated_scale = np.sqrt(
            np.mean(
                np.sum(
                    (generated_rigid - generated_rigid.mean(0)) ** 2,
                    axis=-1,
                )
            )
        )
        rotation_residuals *= target_scale / generated_scale

        prediction = np.empty_like(translated)
        for frame_index, rotation_vector in enumerate(rotation_residuals):
            prediction[frame_index], _ = project_rotation_without_new_clash(
                translated[frame_index],
                rotation_vector,
                fixed_protein,
                physics,
            )
        return prediction.astype(np.float32), weights
