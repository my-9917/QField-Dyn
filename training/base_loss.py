import torch


def cross_index(current_coordinates, protein_coordinates, ligand_ptr, protein_ptr, cutoff):
    indices = []
    for ligand_start, ligand_end, protein_start, protein_end in zip(
        ligand_ptr[:-1], ligand_ptr[1:], protein_ptr[:-1], protein_ptr[1:]
    ):
        if protein_start == protein_end:
            continue
        pairs = torch.nonzero(
            torch.cdist(
                current_coordinates[ligand_start:ligand_end],
                protein_coordinates[protein_start:protein_end],
            )
            <= cutoff,
            as_tuple=False,
        )
        pairs[:, 0] += ligand_start
        pairs[:, 1] += protein_start
        indices.append(pairs.T)
    if not indices:
        return torch.empty(
            (2, 0), dtype=torch.long, device=current_coordinates.device
        )
    return torch.cat(indices, dim=1)


def rollout_loss(
    model, batch, cutoff, coordinate_weight, bond_weight, angle_weight
):
    current = batch["ligand_coordinates"]
    velocity = batch["ligand_velocity"]
    bond_target, bond_source = batch["bond_index"]
    angle_a, angle_center, angle_c = batch["angle_index"]
    coordinate_losses = []
    bond_losses = []
    angle_losses = []

    for target in batch["target_coordinates"]:
        current_cross_index = cross_index(
            current,
            batch["protein_coordinates"],
            batch["ligand_ptr"],
            batch["protein_ptr"],
            cutoff,
        )
        acceleration = model(
            batch["ligand_atomic_numbers"],
            batch["protein_atomic_numbers"],
            current,
            batch["protein_coordinates"],
            velocity,
            batch["bond_index"],
            current_cross_index,
        )
        prediction = current + velocity + acceleration
        coordinate_losses.append(torch.mean((prediction - target) ** 2))
        prediction_bond_length = torch.linalg.vector_norm(
            prediction[bond_source] - prediction[bond_target], dim=-1
        )
        target_bond_length = torch.linalg.vector_norm(
            target[bond_source] - target[bond_target], dim=-1
        )
        bond_losses.append(
            torch.mean((prediction_bond_length - target_bond_length) ** 2)
        )
        prediction_vector_a = prediction[angle_a] - prediction[angle_center]
        prediction_vector_c = prediction[angle_c] - prediction[angle_center]
        target_vector_a = target[angle_a] - target[angle_center]
        target_vector_c = target[angle_c] - target[angle_center]
        prediction_cosine = torch.sum(
            prediction_vector_a * prediction_vector_c, dim=-1
        ) / (
            torch.linalg.vector_norm(prediction_vector_a, dim=-1)
            * torch.linalg.vector_norm(prediction_vector_c, dim=-1)
        )
        target_cosine = torch.sum(target_vector_a * target_vector_c, dim=-1) / (
            torch.linalg.vector_norm(target_vector_a, dim=-1)
            * torch.linalg.vector_norm(target_vector_c, dim=-1)
        )
        angle_losses.append(
            torch.mean((prediction_cosine - target_cosine) ** 2)
        )
        velocity = prediction - current
        current = prediction

    coordinate_loss = torch.stack(coordinate_losses).mean()
    bond_loss = torch.stack(bond_losses).mean()
    angle_loss = torch.stack(angle_losses).mean()
    total_loss = (
        coordinate_weight * coordinate_loss
        + bond_weight * bond_loss
        + angle_weight * angle_loss
    )
    return total_loss, coordinate_loss, bond_loss, angle_loss
