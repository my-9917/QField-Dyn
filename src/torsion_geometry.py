from collections import deque

import numpy as np


def graph_data(atom_count, bonds):
    neighbors = [[] for _ in range(atom_count)]
    for atom_i, atom_j in bonds:
        neighbors[atom_i].append(atom_j)
        neighbors[atom_j].append(atom_i)
    discovery = np.full(atom_count, -1, dtype=np.int64)
    low = np.empty(atom_count, dtype=np.int64)
    bridges = set()
    time = 0

    def visit(atom, parent):
        nonlocal time
        discovery[atom] = low[atom] = time
        time += 1
        for other in neighbors[atom]:
            if other == parent:
                continue
            if discovery[other] == -1:
                visit(other, atom)
                low[atom] = min(low[atom], low[other])
                if low[other] > discovery[atom]:
                    bridges.add(tuple(sorted((atom, other))))
            else:
                low[atom] = min(low[atom], discovery[other])

    visit(0, -1)
    rotatable = {
        edge
        for edge in bridges
        if len(neighbors[edge[0]]) > 1 and len(neighbors[edge[1]]) > 1
    }
    fragment = np.full(atom_count, -1, dtype=np.int64)
    fragments = []
    for start in range(atom_count):
        if fragment[start] >= 0:
            continue
        fragment_id = len(fragments)
        atoms = []
        stack = [start]
        fragment[start] = fragment_id
        while stack:
            atom = stack.pop()
            atoms.append(atom)
            for other in neighbors[atom]:
                if (
                    tuple(sorted((atom, other))) in rotatable
                    or fragment[other] >= 0
                ):
                    continue
                fragment[other] = fragment_id
                stack.append(other)
        fragments.append(np.asarray(sorted(atoms), dtype=np.int64))
    return rotatable, fragment, fragments


def kinematic_tree(reference, rotatable, fragment, fragments):
    ranks = [
        np.linalg.matrix_rank(reference[atoms] - reference[atoms].mean(0))
        for atoms in fragments
    ]
    root = max(
        range(len(fragments)),
        key=lambda index: (ranks[index], len(fragments[index])),
    )
    fragment_edges = [[] for _ in fragments]
    for atom_i, atom_j in rotatable:
        fragment_edges[fragment[atom_i]].append(
            (fragment[atom_j], atom_i, atom_j)
        )
        fragment_edges[fragment[atom_j]].append(
            (fragment[atom_i], atom_j, atom_i)
        )
    parent = np.full(len(fragments), -1, dtype=np.int64)
    parent[root] = root
    order = []
    queue = deque([root])
    while queue:
        parent_fragment = queue.popleft()
        for child_fragment, parent_atom, child_atom in fragment_edges[
            parent_fragment
        ]:
            if parent[child_fragment] >= 0:
                continue
            parent[child_fragment] = parent_fragment
            order.append(
                (parent_fragment, child_fragment, parent_atom, child_atom)
            )
            queue.append(child_fragment)
    children = [[] for _ in fragments]
    for parent_fragment, child_fragment, _, _ in order:
        children[parent_fragment].append(child_fragment)
    subtree_atoms = {}

    def collect(fragment_id):
        atoms = list(fragments[fragment_id])
        for child in children[fragment_id]:
            atoms.extend(collect(child))
        subtree_atoms[fragment_id] = np.asarray(atoms, dtype=np.int64)
        return atoms

    collect(root)
    return root, order, subtree_atoms


def rigid_fit(reference, target):
    reference_center = reference.mean(0)
    target_center = target.mean(0)
    left, _, right_t = np.linalg.svd(
        (reference - reference_center).T @ (target - target_center)
    )
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    translation = target_center - reference_center @ rotation
    return rotation, translation


def rotate_about_axis(coordinates, atom_indices, origin, axis, angle):
    points = coordinates[atom_indices] - origin
    cosine = np.cos(angle)
    sine = np.sin(angle)
    coordinates[atom_indices] = (
        points * cosine
        + np.cross(axis, points) * sine
        + np.outer(points @ axis, axis) * (1.0 - cosine)
        + origin
    )


def torsion_angles(reference, target, fragments, root, order, subtree_atoms):
    rotation, translation = rigid_fit(
        reference[fragments[root]], target[fragments[root]]
    )
    decoded = reference @ rotation + translation
    angles = []
    for _, child_fragment, parent_atom, child_atom in order:
        axis = decoded[child_atom] - decoded[parent_atom]
        axis /= np.linalg.norm(axis)
        atoms = subtree_atoms[child_fragment]
        current = decoded[atoms] - decoded[child_atom]
        desired = target[atoms] - target[child_atom]
        current -= np.outer(current @ axis, axis)
        desired -= np.outer(desired @ axis, axis)
        angle = np.arctan2(
            np.sum(np.cross(current, desired) @ axis),
            np.sum(current * desired),
        )
        angles.append(angle)
        rotate_about_axis(decoded, atoms, decoded[child_atom], axis, angle)
    return np.asarray(angles, dtype=np.float64)


def torsion_bases(frame, order, subtree_atoms, observed_angles, q_angles):
    basis_observed = np.zeros_like(frame, dtype=np.float64)
    basis_q = np.zeros_like(frame, dtype=np.float64)
    for index, (_, child_fragment, parent_atom, child_atom) in enumerate(order):
        axis = frame[child_atom] - frame[parent_atom]
        axis /= np.linalg.norm(axis)
        atoms = subtree_atoms[child_fragment]
        derivative = np.cross(axis, frame[atoms] - frame[child_atom])
        basis_observed[atoms] += observed_angles[index] * derivative
        basis_q[atoms] += q_angles[index] * derivative
    return np.stack((basis_observed, basis_q), axis=-1)


def apply_torsion_patch(frame, order, subtree_atoms, angles):
    corrected = np.asarray(frame, dtype=np.float64).copy()
    for angle, (_, child_fragment, parent_atom, child_atom) in zip(
        angles, order
    ):
        axis = corrected[child_atom] - corrected[parent_atom]
        axis /= np.linalg.norm(axis)
        rotate_about_axis(
            corrected,
            subtree_atoms[child_fragment],
            corrected[child_atom],
            axis,
            angle,
        )
    return corrected

