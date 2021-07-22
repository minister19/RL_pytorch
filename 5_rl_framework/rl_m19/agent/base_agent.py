import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim


class BaseAgent():
    def __init__(self, config):
        self.config = config
        self.eps_fn = config.EPS_fn
        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = config.optimizer
        self.loss_fn = config.loss_fn
        self.episode_t = []

    def select_action(self, state):
        raise NotImplemented

    def learn(self, state, action, reward, next_state, next_action):
        raise NotImplemented

    def episode_learn(self, i_episode):
        raise NotImplemented

    def episodes_learn(self):
        raise NotImplemented


class AgentUtils():
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
        else:
            return 1

    @staticmethod
    def rectify(number):
        if number < -1:
            return -1
        elif -1 <= number <= 1:
            return number
        else:
            return 1
