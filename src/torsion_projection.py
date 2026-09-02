import numpy as np

from .torsion_geometry import apply_torsion_patch


def project_to_torsion_manifold(
    frame,
    target,
    heavy,
    order,
    subtree_atoms,
    ridge=1e-4,
    angle_limit=0.1,
):
    frame = np.asarray(frame, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not order:
        return frame.copy(), 0.0
    jacobian = np.zeros((*frame.shape, len(order)), dtype=np.float64)
    for index, (_, child_fragment, parent_atom, child_atom) in enumerate(order):
        axis = frame[child_atom] - frame[parent_atom]
        axis /= np.linalg.norm(axis)
        atoms = subtree_atoms[child_fragment]
        jacobian[atoms, :, index] = np.cross(
            axis, frame[atoms] - frame[child_atom]
        )
    design = jacobian[heavy].reshape(-1, len(order))
    response = (target - frame)[heavy].reshape(-1)
    angles = np.linalg.solve(
        design.T @ design + ridge * np.eye(len(order)),
        design.T @ response,
    )
    angles = np.clip(angles, -angle_limit, angle_limit)
    corrected = apply_torsion_patch(frame, order, subtree_atoms, angles)
    corrected += frame[heavy].mean(0) - corrected[heavy].mean(0)
    return corrected, float(np.max(np.abs(angles)))
