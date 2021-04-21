import torch


class BaseEnv():
    def __init__(self, device):
        self.device = device

    def unsqueeze_tensor(self, x):
        x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = torch.unsqueeze(x, 0)
        return x
