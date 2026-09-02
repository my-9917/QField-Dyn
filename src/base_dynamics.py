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

