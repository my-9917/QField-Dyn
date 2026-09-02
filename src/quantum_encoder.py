import torch
from torch import nn


class QuantumEncoder(nn.Module):
    def __init__(self, hidden_dim, layer_count):
        super().__init__()
        self.embedding = nn.Embedding(119, hidden_dim)
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layer_count)
            ]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layer_count)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layer_count)]
        )
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.molecule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, z, pos, edge_index, batch):
        hidden = self.embedding(z)
        target, source = edge_index
        distance = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1, keepdim=True)
        for message_layer, update_layer, norm in zip(
            self.messages, self.updates, self.norms
        ):
            message = message_layer(
                torch.cat((hidden[target], hidden[source], distance), dim=-1)
            )
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, target, message)
            hidden = norm(hidden + update_layer(torch.cat((hidden, aggregate), dim=-1)))
        atom_prediction = self.atom_head(hidden).squeeze(-1)
        molecule_hidden = torch.zeros(
            int(batch.max()) + 1,
            hidden.shape[1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        molecule_hidden.index_add_(0, batch, hidden)
        counts = torch.bincount(batch, minlength=len(molecule_hidden)).to(hidden.dtype)
        molecule_hidden = molecule_hidden / counts[:, None]
        molecule_prediction = self.molecule_head(molecule_hidden)
        return atom_prediction, molecule_prediction, hidden
