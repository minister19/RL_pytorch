import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim

from itertools import count


class BaseAgent():
    def __init__(self, config):
        self.config = config
        self.eps_fn = config.EPS_fn
        self.policy_net = config.policy_net_fn()
        self.target_net = config.target_net_fn()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = config.optimizer_fn(self.policy_net.parameters(), self.config.LR)
        self.loss_fn = config.loss_fn
        self.episode_t = []

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def tensor2str(self, x):
        y = torch.squeeze(x).numpy()
        y = map(lambda n: str(n), y)
        string = "_".join(y)
        return string

    def tensor2number(self, x):
        number = torch.squeeze(x).item()
        return number
