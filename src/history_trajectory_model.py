import torch
from torch import nn

from .base_dynamics import LigandMessageLayer, mlp


class HistoryQuantumTrajectoryModel(nn.Module):
    def __init__(self, hidden_dim=128, layer_count=3, quantum_feature_dim=132):
        super().__init__()
        self.atomic_embedding = nn.Embedding(128, hidden_dim)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.speed_embedding = mlp(1, hidden_dim, hidden_dim)
        self.quantum_projection = mlp(
            quantum_feature_dim, hidden_dim, hidden_dim
        )
        self.history_projection = mlp(4, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            LigandMessageLayer(hidden_dim) for _ in range(layer_count)
        )
        edge_dim = 2 * hidden_dim + 1
        self.bond_force = mlp(edge_dim, hidden_dim, 1)
        self.cross_force = mlp(edge_dim, hidden_dim, 1)
        self.damping = mlp(hidden_dim, hidden_dim, 1)
        self.memory_force = mlp(hidden_dim, hidden_dim, 1)
        self.memory_decay = mlp(hidden_dim, hidden_dim, 1)
        for module in (
            self.quantum_projection,
            self.history_projection,
            self.memory_force,
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(
        self,
        ligand_atomic_numbers,
        protein_atomic_numbers,
        ligand_coordinates,
        protein_coordinates,
        ligand_velocity,
        memory_state,
        history_features,
        bond_index,
        cross_index,
        quantum_features,
    ):
        ligand_hidden = (
            self.atomic_embedding(ligand_atomic_numbers)
            + self.role_embedding.weight[0]
            + self.speed_embedding(
                torch.linalg.vector_norm(
                    ligand_velocity, dim=-1, keepdim=True
                )
            )
            + self.quantum_projection(quantum_features)
            + self.history_projection(history_features)
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
            ligand_coordinates[bond_source]
            - ligand_coordinates[bond_target]
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
        memory_coefficient = torch.tanh(self.memory_force(ligand_hidden))
        memory_decay = torch.sigmoid(self.memory_decay(ligand_hidden))
        acceleration = (
            acceleration
            - damping * ligand_velocity
            + memory_coefficient * memory_state
        )
        return acceleration, memory_decay
