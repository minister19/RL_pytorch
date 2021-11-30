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
    'eps_fn': None,

    'gamma': 0.999,
    'lr': 0.001
}

DQN = {
    'episode_lifespan': 10**3,
    'episodes': 1000,
    'eps_fn': None,

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

DDPG = {
    'episode_lifespan': 10**3,
    'test_episode_lifespan': 10**3,

    'replay_size': 10**6,
    'batch_size': 10**4,
    'pi_lr': 1e-3,
    'q_lr': 1e-3,
    'gamma': 0.99,
    'polyak': 0.995,

    'epoch_lifespan': 10**3*5,
    'epochs': 100,
    'init_wander': 10**3*10,
    'act_noise': 0.1,
    'update_after': 10**3*1,
    'update_every': 10**3*0.1,
}


class Config:
    def __init__(self, dicts: List[dict]):
        for d in dicts:
            for k, v in d.items():
                setattr(self, k, v)
