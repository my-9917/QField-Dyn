import numpy as np


def project_translation_without_new_clash(
    frame, displacement, fixed_protein, physics
):
    ligand = frame[physics["ligand_heavy_indices"]]
    threshold = physics["protein_ligand_threshold"]
    max_threshold = float(np.max(threshold))
    candidate = ligand + displacement
    lower = np.minimum(ligand.min(0), candidate.min(0)) - max_threshold
    upper = np.maximum(ligand.max(0), candidate.max(0)) + max_threshold
    protein_mask = np.all(
        (fixed_protein >= lower) & (fixed_protein <= upper), axis=1
    )
    vectors = ligand[None] - fixed_protein[protein_mask, None]
    local_threshold = threshold[protein_mask]
    baseline_safe = np.sum(vectors**2, axis=-1) >= local_threshold**2
    candidate_clash = (
        np.sum((vectors + displacement) ** 2, axis=-1)
        < local_threshold**2
    )
    if not np.any(baseline_safe & candidate_clash):
        return frame + displacement, 1.0

    vectors = vectors[baseline_safe]
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(fixed_protein[protein_mask]))),
        float(np.max(np.abs(ligand))),
    )
    safety_margin = 8.0 * np.finfo(np.float32).eps * coordinate_scale
    radii = local_threshold[baseline_safe] + safety_margin
    quadratic = float(np.dot(displacement, displacement))
    linear = 2.0 * (vectors @ displacement)
    constant = np.sum(vectors**2, axis=-1) - radii**2
    discriminant = linear**2 - 4.0 * quadratic * constant
    crossing = discriminant > 0.0
    root = np.sqrt(discriminant[crossing])
    entry = (-linear[crossing] - root) / (2.0 * quadratic)
    exit = (-linear[crossing] + root) / (2.0 * quadratic)
    intersects = (entry < 1.0) & (exit > 0.0)
    intervals = sorted(
        zip(np.maximum(entry[intersects], 0.0), np.minimum(exit[intersects], 1.0))
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    unsafe_at_one = next(start for start, end in merged if end >= 1.0)
    scale = float(
        np.nextafter(np.float32(unsafe_at_one), np.float32(0.0))
    )
    return frame + scale * displacement, scale
