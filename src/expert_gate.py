import torch
from torch import nn


class ExpertGate(nn.Module):
    def __init__(self, context_dim=1033, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context):
        return torch.softmax(self.network(context).reshape(-1, 4, 2), dim=-1)
