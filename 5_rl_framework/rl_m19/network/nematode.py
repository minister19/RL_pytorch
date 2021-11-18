import torch
import torch.nn as nn
from rl_m19.network import core


class Nematode(nn.Module):
    def __init__(self, state_dim, action_dim, device=None):
        super().__init__()
        self.net = core.mlp((state_dim, state_dim // 2, action_dim), bias=False)
        self.to(device)

    def forward(self, x: torch.Tensor):
        return self.net(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 21
    action_dim = 3
    nematode = Nematode(state_dim, action_dim, device)
    x = torch.rand((1, state_dim), device=device)
    y = nematode(x)
    print(y)

    from torchviz import make_dot
    dot = make_dot(y, params=dict(nematode.named_parameters()), show_attrs=True, show_saved=True)
    dot.render('data/round-table.gv', view=True)
