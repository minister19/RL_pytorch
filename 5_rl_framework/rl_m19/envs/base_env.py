import torch


class BaseEnv():
    def __init__(self, device):
        self.device = device
        self.states_dim = None
        self.actions_dim = None

    def _unsqueeze_tensor(self, x):
        x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = torch.unsqueeze(x, 0)
        return x

    def step(self, action: int):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
