import torch

from .base_loss import cross_index


def rmsf(coordinates):
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    return torch.sqrt(torch.mean(torch.sum(centered**2, dim=-1)))


def radius_of_gyration(coordinates):
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    return torch.sqrt(torch.mean(torch.sum(centered**2, dim=-1), dim=1))


def velocity_correlation(coordinates):
    velocity = torch.diff(coordinates, dim=0)
    return torch.mean(torch.sum(velocity[:-1] * velocity[1:], dim=-1)) / torch.mean(
        torch.sum(velocity[:-1] ** 2, dim=-1)
    )


def rollout_components(model, batch, cutoff, contact_temperature):
    current = batch["ligand_coordinates"]
    velocity = batch["ligand_velocity"]
    bond_target, bond_source = batch["bond_index"]
    angle_a, angle_center, angle_c = batch["angle_index"]
    coordinate_losses = []
    bond_losses = []
    angle_losses = []
    predictions = []

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
        predictions.append(prediction)
        coordinate_losses.append(torch.mean((prediction - target) ** 2))

        prediction_bond = torch.linalg.vector_norm(
            prediction[bond_source] - prediction[bond_target], dim=-1
        )
        target_bond = torch.linalg.vector_norm(
            target[bond_source] - target[bond_target], dim=-1
        )
        bond_losses.append(torch.mean((prediction_bond - target_bond) ** 2))

        prediction_a = prediction[angle_a] - prediction[angle_center]
        prediction_c = prediction[angle_c] - prediction[angle_center]
        target_a = target[angle_a] - target[angle_center]
        target_c = target[angle_c] - target[angle_center]
        prediction_cosine = torch.sum(prediction_a * prediction_c, dim=-1) / (
            torch.linalg.vector_norm(prediction_a, dim=-1)
            * torch.linalg.vector_norm(prediction_c, dim=-1)
        )
        target_cosine = torch.sum(target_a * target_c, dim=-1) / (
            torch.linalg.vector_norm(target_a, dim=-1)
            * torch.linalg.vector_norm(target_c, dim=-1)
        )
        angle_losses.append(torch.mean((prediction_cosine - target_cosine) ** 2))
        velocity = prediction - current
        current = prediction

    prediction = torch.stack(predictions)
    target = batch["target_coordinates"]
    rmsf_losses = []
    speed_losses = []
    contact_losses = []
    transition_losses = []
    correlation_losses = []
    rg_losses = []

    for ligand_start, ligand_end, protein_start, protein_end in zip(
        batch["ligand_ptr"][:-1],
        batch["ligand_ptr"][1:],
        batch["protein_ptr"][:-1],
        batch["protein_ptr"][1:],
    ):
        ligand_heavy = (
            batch["ligand_atomic_numbers"][ligand_start:ligand_end] != 1
        )
        protein_heavy = (
            batch["protein_atomic_numbers"][protein_start:protein_end] != 1
        )
        pred = prediction[:, ligand_start:ligand_end, :][:, ligand_heavy]
        truth = target[:, ligand_start:ligand_end, :][:, ligand_heavy]
        protein = batch["protein_coordinates"][protein_start:protein_end][
            protein_heavy
        ]

        rmsf_losses.append((rmsf(pred) - rmsf(truth)) ** 2)
        pred_speed = torch.sort(
            torch.linalg.vector_norm(torch.diff(pred, dim=0), dim=-1).flatten()
        ).values
        true_speed = torch.sort(
            torch.linalg.vector_norm(torch.diff(truth, dim=0), dim=-1).flatten()
        ).values
        speed_losses.append(torch.mean(torch.abs(pred_speed - true_speed)))

        if len(protein):
            pred_distance = torch.stack(
                [torch.cdist(frame, protein).min(dim=1).values for frame in pred]
            )
            true_distance = torch.stack(
                [torch.cdist(frame, protein).min(dim=1).values for frame in truth]
            )
            contact_losses.append(
                torch.mean(
                    torch.abs(
                        torch.sort(pred_distance, dim=1).values
                        - torch.sort(true_distance, dim=1).values
                    )
                )
            )
            pred_contact = torch.sigmoid(
                (4.5 - pred_distance) / contact_temperature
            ).mean(dim=1)
            true_contact = torch.sigmoid(
                (4.5 - true_distance) / contact_temperature
            ).mean(dim=1)
            transition_losses.append(
                torch.mean(
                    torch.abs(
                        torch.abs(torch.diff(pred_contact))
                        - torch.abs(torch.diff(true_contact))
                    )
                )
            )

        correlation_losses.append(
            (
                velocity_correlation(pred)
                - velocity_correlation(truth)
            )
            ** 2
        )
        rg_losses.append(
            torch.mean(
                torch.abs(
                    radius_of_gyration(pred) - radius_of_gyration(truth)
                )
            )
        )

    components = {
        "coordinate": torch.stack(coordinate_losses).mean(),
        "bond": torch.stack(bond_losses).mean(),
        "angle": torch.stack(angle_losses).mean(),
        "rmsf": torch.stack(rmsf_losses).mean(),
        "speed_w1": torch.stack(speed_losses).mean(),
        "contact_w1": torch.stack(contact_losses).mean(),
        "contact_transition": torch.stack(transition_losses).mean(),
        "velocity_correlation": torch.stack(correlation_losses).mean(),
        "radius_of_gyration": torch.stack(rg_losses).mean(),
    }
    if not torch.isfinite(torch.stack(tuple(components.values()))).all():
        raise FloatingPointError("non-finite joint trajectory loss")
    return components
