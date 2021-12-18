import math
from typing import List

Core = {
    'logger': None,
    'plotter': None,
    'device': None,
    'env': None,
    'test_env': None,
    'state_dim': None,
    'action_dim': None
}

CNN = {
    'cnn_image_width': 0,
    'cnn_image_height': 0
}

Q = {
    'episode_lifespan': 1e3,
    'episodes': 1000,
    'eps_fn': lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000),

    'gamma': 0.999,
    'lr': 0.001
}

DQN = {
    'episode_lifespan': 10**3,
    'episodes': 1000,
    'eps_fn': lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000),

    'replay_size': 10**5,
    'batch_size': 10**3,
    'gamma': 0.999,
    'lr': 0.001,
    'target_update_freq': 10,

    'policy_net': None,
    'target_net': None,
    'optimizer': None,
    'loss_fn': None,
}


class Config:
    def __init__(self, dicts: List[dict]):
        for d in dicts:
            for k, v in d.items():
                setattr(self, k, v)
