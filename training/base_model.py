import torch
from torch import nn


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class LigandMessageLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        edge_dim = 2 * hidden_dim + 1
        self.bond_message = mlp(edge_dim, hidden_dim, hidden_dim)
        self.cross_message = mlp(edge_dim, hidden_dim, hidden_dim)
        self.update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(
        self,
        ligand_hidden,
        protein_hidden,
        ligand_coordinates,
        protein_coordinates,
        bond_index,
        cross_index,
    ):
        aggregate = torch.zeros_like(ligand_hidden)

        bond_target, bond_source = bond_index
        bond_distance = torch.linalg.vector_norm(
            ligand_coordinates[bond_source] - ligand_coordinates[bond_target],
            dim=-1,
            keepdim=True,
        )
        bond_message = self.bond_message(
            torch.cat(
                (
                    ligand_hidden[bond_target],
                    ligand_hidden[bond_source],
                    bond_distance,
                ),
                dim=-1,
            )
        )
        aggregate.index_add_(0, bond_target, bond_message)

        cross_ligand, cross_protein = cross_index
        cross_distance = torch.linalg.vector_norm(
            protein_coordinates[cross_protein]
            - ligand_coordinates[cross_ligand],
            dim=-1,
            keepdim=True,
        )
        cross_message = self.cross_message(
            torch.cat(
                (
                    ligand_hidden[cross_ligand],
                    protein_hidden[cross_protein],
                    cross_distance,
                ),
                dim=-1,
            )
        )
        aggregate.index_add_(0, cross_ligand, cross_message)

        return ligand_hidden + self.update(
            torch.cat((ligand_hidden, aggregate), dim=-1)
        )


class BaseTrajectoryModel(nn.Module):
    def __init__(self, hidden_dim=128, layer_count=3):
        super().__init__()
        self.atomic_embedding = nn.Embedding(128, hidden_dim)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.speed_embedding = mlp(1, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            LigandMessageLayer(hidden_dim) for _ in range(layer_count)
        )
        edge_dim = 2 * hidden_dim + 1
        self.bond_force = mlp(edge_dim, hidden_dim, 1)
        self.cross_force = mlp(edge_dim, hidden_dim, 1)
        self.damping = mlp(hidden_dim, hidden_dim, 1)
        nn.init.zeros_(self.bond_force[-1].weight)
        nn.init.zeros_(self.bond_force[-1].bias)
        nn.init.zeros_(self.cross_force[-1].weight)
        nn.init.zeros_(self.cross_force[-1].bias)
        nn.init.zeros_(self.damping[-1].weight)
        nn.init.zeros_(self.damping[-1].bias)

    def forward(
        self,
        ligand_atomic_numbers,
        protein_atomic_numbers,
        ligand_coordinates,
        protein_coordinates,
        ligand_velocity,
        bond_index,
        cross_index,
    ):
        ligand_hidden = (
            self.atomic_embedding(ligand_atomic_numbers)
            + self.role_embedding.weight[0]
            + self.speed_embedding(
                torch.linalg.vector_norm(
                    ligand_velocity, dim=-1, keepdim=True
                )
            )
        )
        protein_hidden = (
            self.atomic_embedding(protein_atomic_numbers)
            + self.role_embedding.weight[1]
        )

        for layer in self.layers:
            ligand_hidden = layer(
                ligand_hidden,
                protein_hidden,
                ligand_coordinates,
                protein_coordinates,
                bond_index,
                cross_index,
            )

        acceleration = torch.zeros_like(ligand_coordinates)

        bond_target, bond_source = bond_index
        bond_vector = (
            ligand_coordinates[bond_source] - ligand_coordinates[bond_target]
        )
        bond_distance = torch.linalg.vector_norm(
            bond_vector, dim=-1, keepdim=True
        )
        bond_coefficient = self.bond_force(
            torch.cat(
                (
                    ligand_hidden[bond_target],
                    ligand_hidden[bond_source],
                    bond_distance,
                ),
                dim=-1,
            )
        )
        acceleration.index_add_(
            0, bond_target, bond_coefficient * bond_vector / bond_distance
        )

        cross_ligand, cross_protein = cross_index
        cross_vector = (
            protein_coordinates[cross_protein]
            - ligand_coordinates[cross_ligand]
        )
        cross_distance = torch.linalg.vector_norm(
            cross_vector, dim=-1, keepdim=True
        )
        cross_coefficient = self.cross_force(
            torch.cat(
                (
                    ligand_hidden[cross_ligand],
                    protein_hidden[cross_protein],
                    cross_distance,
                ),
                dim=-1,
            )
        )
        acceleration.index_add_(
            0,
            cross_ligand,
            cross_coefficient * cross_vector / cross_distance,
        )

        damping = torch.nn.functional.softplus(self.damping(ligand_hidden))
        return acceleration - damping * ligand_velocity
