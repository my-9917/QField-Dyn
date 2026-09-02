import numpy as np
from scipy.spatial.transform import Rotation


def project_rotation_without_new_clash(
    frame, rotation_vector, fixed_protein, physics
):
    heavy_indices = physics["ligand_heavy_indices"]
    threshold = physics["protein_ligand_threshold"]
    center = frame[heavy_indices].mean(axis=0)
    centered = frame - center
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(fixed_protein))),
        float(np.max(np.abs(frame))),
    )
    safe_threshold = threshold + 8.0 * np.finfo(np.float32).eps * coordinate_scale

    def rotate(scale):
        if scale == 0.0:
            return frame
        return Rotation.from_rotvec(scale * rotation_vector).apply(centered) + center

    baseline_ligand = frame[heavy_indices]

    def safe(scale, candidate):
        if scale == 0.0:
            return True
        ligand = candidate[heavy_indices]
        max_threshold = float(np.max(safe_threshold))
        lower = ligand.min(0) - max_threshold
        upper = ligand.max(0) + max_threshold
        protein_mask = np.all(
            (fixed_protein >= lower) & (fixed_protein <= upper), axis=1
        )
        baseline_distance = np.linalg.norm(
            fixed_protein[protein_mask, None] - baseline_ligand[None], axis=-1
        )
        candidate_distance = np.linalg.norm(
            fixed_protein[protein_mask, None] - ligand[None], axis=-1
        )
        return not np.any(
            (baseline_distance >= threshold[protein_mask])
            & (candidate_distance < safe_threshold[protein_mask])
        )

    candidate = rotate(1.0)
    if safe(1.0, candidate):
        return candidate, 1.0

    grid = np.linspace(15.0 / 16.0, 0.0, 16)
    lower = next(scale for scale in grid if safe(scale, rotate(scale)))
    upper = lower + 1.0 / 16.0
    for _ in range(12):
        middle = (lower + upper) / 2.0
        if safe(middle, rotate(middle)):
            lower = middle
        else:
            upper = middle
    return rotate(lower), lower
