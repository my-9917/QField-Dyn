import torch

from .quantum_encoder import QuantumEncoder


class QuantumFeatureEncoder(QuantumEncoder):
    def forward_layers(self, z, pos, edge_index, batch):
        hidden = self.embedding(z)
        layers = [hidden]
        target, source = edge_index
        distance = torch.linalg.vector_norm(
            pos[source] - pos[target], dim=-1, keepdim=True
        )
        for message_layer, update_layer, norm in zip(
            self.messages, self.updates, self.norms
        ):
            message = message_layer(
                torch.cat((hidden[target], hidden[source], distance), dim=-1)
            )
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, target, message)
            hidden = norm(hidden + update_layer(torch.cat((hidden, aggregate), dim=-1)))
            layers.append(hidden)
        atom_prediction = self.atom_head(hidden).squeeze(-1)
        molecule_hidden = torch.zeros(
            int(batch.max()) + 1,
            hidden.shape[1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        molecule_hidden.index_add_(0, batch, hidden)
        counts = torch.bincount(batch, minlength=len(molecule_hidden)).to(hidden.dtype)
        molecule_prediction = self.molecule_head(molecule_hidden / counts[:, None])
        return atom_prediction, molecule_prediction, torch.stack(layers, dim=1)
