import torch

from .base_loss import cross_index


def rollout_loss(model, batch, cutoff, coordinate_weight, bond_weight, angle_weight):
    current = batch["ligand_coordinates"]
    velocity = batch["ligand_velocity"]
    bond_target, bond_source = batch["bond_index"]
    angle_a, angle_center, angle_c = batch["angle_index"]
    coordinate_losses = []
    bond_losses = []
    angle_losses = []
    for rollout_step, target in enumerate(batch["target_coordinates"], start=1):
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
            batch["quantum_features"],
        )
        if not torch.isfinite(acceleration).all():
            raise FloatingPointError(
                f"rollout step {rollout_step}: non-finite acceleration"
            )
        prediction = current + velocity + acceleration
        coordinate_loss = torch.mean((prediction - target) ** 2)
        prediction_bond_length = torch.linalg.vector_norm(
            prediction[bond_source] - prediction[bond_target], dim=-1
        )
        target_bond_length = torch.linalg.vector_norm(
            target[bond_source] - target[bond_target], dim=-1
        )
        bond_loss = torch.mean((prediction_bond_length - target_bond_length) ** 2)
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
        angle_loss = torch.mean((prediction_cosine - target_cosine) ** 2)
        losses = torch.stack((coordinate_loss, bond_loss, angle_loss))
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                f"rollout step {rollout_step}: non-finite component loss"
            )
        coordinate_losses.append(coordinate_loss)
        bond_losses.append(bond_loss)
        angle_losses.append(angle_loss)
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
