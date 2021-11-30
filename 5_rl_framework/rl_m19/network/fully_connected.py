import torch
import torch.nn as nn
from rl_m19.network import core


class FullyConnected(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(32, 32), device=None):
        super().__init__()
        fc_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.fc = core.mlp(fc_sizes, bias=False)
        self.to(device)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
