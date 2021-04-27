import math
import random
import numpy
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

    @staticmethod
    def tensor2str(x):
        y = torch.squeeze(x).numpy()
        if y.size <= 1:
            string = str(y)
        else:
            y = map(lambda n: str(n), y)
            string = "_".join(y)
        return string

    @staticmethod
    def tensor2number(x):
        number = torch.squeeze(x).item()
        return number

    @staticmethod
    def heavisde(number):
        if number < 0:
            return -1
        elif number == 0:
            return 0
        elif number > 0:
            return 1
        else:
            raise NotImplementedError

    @staticmethod
    def rectify(number):
        if number < -1:
            return -1
        elif -1 <= number <= 1:
            return number
        elif number > 1:
            return 1
        else:
            raise NotImplementedError
