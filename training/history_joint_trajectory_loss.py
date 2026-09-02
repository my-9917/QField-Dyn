import torch


def atom_rmsf(coordinates):
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    return torch.sqrt(torch.mean(torch.sum(centered**2, dim=-1), dim=0))


def global_rmsf(coordinates):
    return torch.sqrt(torch.mean(atom_rmsf(coordinates) ** 2))


def radius_of_gyration(coordinates):
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    return torch.sqrt(torch.mean(torch.sum(centered**2, dim=-1), dim=1))


def lag_correlation(velocity, lag):
    return torch.mean(
        torch.sum(velocity[:-lag] * velocity[lag:], dim=-1)
    ) / torch.mean(torch.sum(velocity[:-lag] ** 2, dim=-1))


def rollout_components(model, batch, cutoff, contact_temperature):
    observed = batch["observed_coordinates"]
    target = batch["target_coordinates"]
    current = observed[-1]
    velocity = observed[-1] - observed[-2]
    memory = batch["memory_state"]
    bond_target, bond_source = batch["bond_index"]
    angle_a, angle_center, angle_c = batch["angle_index"]
    coordinate_losses = []
    velocity_losses = []
    bond_losses = []
    angle_losses = []
    predictions = []
    predicted_velocities = []
    truth_previous = observed[-1]

    for rollout_step, truth in enumerate(target, start=1):
        cross_index = torch.nonzero(
            torch.cdist(current, batch["protein_coordinates"]) <= cutoff,
            as_tuple=False,
        ).T
        acceleration, memory_decay = model(
            batch["ligand_atomic_numbers"],
            batch["protein_atomic_numbers"],
            current,
            batch["protein_coordinates"],
            velocity,
            memory,
            batch["history_features"],
            batch["bond_index"],
            cross_index,
            batch["quantum_features"],
        )
        if not torch.isfinite(acceleration).all():
            raise FloatingPointError(
                f"rollout step {rollout_step}: non-finite acceleration"
            )
        prediction = current + velocity + acceleration
        predicted_velocity = prediction - current
        true_velocity = truth - truth_previous
        predictions.append(prediction)
        predicted_velocities.append(predicted_velocity)
        coordinate_losses.append(torch.mean((prediction - truth) ** 2))
        velocity_losses.append(
            torch.mean((predicted_velocity - true_velocity) ** 2)
        )

        prediction_bond = torch.linalg.vector_norm(
            prediction[bond_source] - prediction[bond_target], dim=-1
        )
        target_bond = torch.linalg.vector_norm(
            truth[bond_source] - truth[bond_target], dim=-1
        )
        bond_losses.append(torch.mean((prediction_bond - target_bond) ** 2))

        prediction_a = prediction[angle_a] - prediction[angle_center]
        prediction_c = prediction[angle_c] - prediction[angle_center]
        target_a = truth[angle_a] - truth[angle_center]
        target_c = truth[angle_c] - truth[angle_center]
        prediction_cosine = torch.sum(prediction_a * prediction_c, dim=-1) / (
            torch.linalg.vector_norm(prediction_a, dim=-1)
            * torch.linalg.vector_norm(prediction_c, dim=-1)
        )
        target_cosine = torch.sum(target_a * target_c, dim=-1) / (
            torch.linalg.vector_norm(target_a, dim=-1)
            * torch.linalg.vector_norm(target_c, dim=-1)
        )
        angle_losses.append(
            torch.mean((prediction_cosine - target_cosine) ** 2)
        )
        memory = memory_decay * memory + (1.0 - memory_decay) * predicted_velocity
        velocity = predicted_velocity
        current = prediction
        truth_previous = truth

    prediction = torch.stack(predictions)
    predicted_velocity = torch.stack(predicted_velocities)
    ligand_heavy = batch["ligand_atomic_numbers"] != 1
    protein_heavy = batch["protein_atomic_numbers"] != 1
    pred = prediction[:, ligand_heavy]
    truth = target[:, ligand_heavy]
    pred_velocity = predicted_velocity[:, ligand_heavy]
    truth_velocity = torch.diff(
        torch.cat((observed[-1:, ligand_heavy], truth), dim=0), dim=0
    )
    protein = batch["protein_coordinates"][protein_heavy]

    pred_speed = torch.sort(
        torch.linalg.vector_norm(pred_velocity, dim=-1).flatten()
    ).values
    true_speed = torch.sort(
        torch.linalg.vector_norm(truth_velocity, dim=-1).flatten()
    ).values
    zero = prediction.sum() * 0.0
    if len(protein):
        pred_distance = torch.stack(
            [torch.cdist(frame, protein).min(dim=1).values for frame in pred]
        )
        true_distance = torch.stack(
            [torch.cdist(frame, protein).min(dim=1).values for frame in truth]
        )
        pred_contact = torch.sigmoid(
            (4.5 - pred_distance) / contact_temperature
        ).mean(dim=1)
        true_contact = torch.sigmoid(
            (4.5 - true_distance) / contact_temperature
        ).mean(dim=1)
        contact_w1 = torch.mean(
            torch.abs(
                torch.sort(pred_distance, dim=1).values
                - torch.sort(true_distance, dim=1).values
            )
        )
        contact_transition = torch.mean(
            torch.abs(
                torch.abs(torch.diff(pred_contact))
                - torch.abs(torch.diff(true_contact))
            )
        )
    else:
        contact_w1 = zero
        contact_transition = zero

    pred_atom_rmsf = atom_rmsf(pred)
    true_atom_rmsf = atom_rmsf(truth)
    legacy_rmsf = (global_rmsf(pred) - global_rmsf(truth)) ** 2
    lag_losses = torch.stack(
        [
            (
                lag_correlation(pred_velocity, lag)
                - lag_correlation(truth_velocity, lag)
            )
            ** 2
            for lag in (1, 2, 4, 8)
        ]
    )
    components = {
        "coordinate": torch.stack(coordinate_losses).mean(),
        "velocity_vector": torch.stack(velocity_losses).mean(),
        "bond": torch.stack(bond_losses).mean(),
        "angle": torch.stack(angle_losses).mean(),
        "atom_rmsf": torch.mean(torch.abs(pred_atom_rmsf - true_atom_rmsf)),
        "speed_w1": torch.mean(torch.abs(pred_speed - true_speed)),
        "contact_w1": contact_w1,
        "contact_transition": contact_transition,
        "multi_lag_velocity_correlation": lag_losses.mean(),
        "radius_of_gyration": torch.mean(
            torch.abs(
                radius_of_gyration(pred) - radius_of_gyration(truth)
            )
        ),
        "legacy_rmsf": legacy_rmsf,
        "legacy_velocity_correlation": lag_losses[0],
    }
    if not torch.isfinite(torch.stack(tuple(components.values()))).all():
        raise FloatingPointError("non-finite history trajectory loss")
    return components
