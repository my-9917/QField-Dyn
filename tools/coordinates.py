import gzip
from pathlib import Path

import numpy as np


def amber_cell(path: Path) -> np.ndarray:
    lines = gzip.open(path, "rt").readlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("%FLAG BOX_DIMENSIONS"))
    values = []
    for line in lines[index + 2 :]:
        if line.startswith("%FLAG"):
            break
        values.extend(float(value) for value in line.split())
    beta, a, b, c = values[:4]
    angle = np.deg2rad(beta)
    cosine, sine = np.cos(angle), np.sin(angle)
    c_y = c * (cosine - cosine**2) / sine
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cosine, b * sine, 0.0],
            [c * cosine, c_y, np.sqrt(c**2 - (c * cosine) ** 2 - c_y**2)],
        ]
    )


def kabsch(moving: np.ndarray, reference: np.ndarray):
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (moving - moving_center).T @ (reference - reference_center)
    )
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt, moving_center, reference_center


def corrected_ligand_trajectory(
    trajectory: np.ndarray,
    atomic_numbers: np.ndarray,
    molecule_starts: np.ndarray,
    cell: np.ndarray,
):
    starts = molecule_starts.astype(int)
    ends = np.append(starts[1:], trajectory.shape[1])
    ligand_start = starts[-1]
    ligand_heavy = atomic_numbers[ligand_start:] != 1
    receptor_index = int(
        np.argmax(
            [
                np.sum(atomic_numbers[start:end] != 1)
                for start, end in zip(starts[:-1], ends[:-1])
            ]
        )
    )
    receptor_start, receptor_end = starts[receptor_index], ends[receptor_index]
    receptor_heavy = atomic_numbers[receptor_start:receptor_end] != 1
    reference_receptor = trajectory[0, receptor_start:receptor_end][receptor_heavy]
    inverse_cell = np.linalg.inv(cell)
    previous_receptor_center = trajectory[0, receptor_start:receptor_end].mean(axis=0)
    previous_ligand_center = trajectory[0, ligand_start:][ligand_heavy].mean(axis=0)
    corrected = []
    receptor_rmsd = []

    for frame in trajectory:
        raw_receptor = frame[receptor_start:receptor_end]
        receptor_shift = -np.rint(
            (raw_receptor.mean(axis=0) - previous_receptor_center) @ inverse_cell
        )
        receptor = raw_receptor + receptor_shift @ cell
        previous_receptor_center = receptor.mean(axis=0)
        rotation, moving_center, reference_center = kabsch(
            receptor[receptor_heavy], reference_receptor
        )
        aligned_receptor = (
            receptor[receptor_heavy] - moving_center
        ) @ rotation + reference_center
        receptor_rmsd.append(
            np.sqrt(
                np.mean(np.sum((aligned_receptor - reference_receptor) ** 2, axis=1))
            )
        )

        raw_ligand = frame[ligand_start:]
        ligand = (raw_ligand - moving_center) @ rotation + reference_center
        rotated_cell = cell @ rotation
        ligand_shift = -np.rint(
            (ligand[ligand_heavy].mean(axis=0) - previous_ligand_center)
            @ np.linalg.inv(rotated_cell)
        )
        ligand = ligand + ligand_shift @ rotated_cell
        previous_ligand_center = ligand[ligand_heavy].mean(axis=0)
        corrected.append(ligand)

    return np.asarray(corrected), receptor_index, np.asarray(receptor_rmsd)
