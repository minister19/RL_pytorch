from rl_m19.config import Config
from rl_m19.agent.ddpg_agent import DDPGAgent

config = Config()
config.seed = 0

if __name__ == '__main__':
    import numpy as np
    import torch

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    agent = DDPGAgent(config)
    agent.episodes_learn()
    config.env.render()
    config.env.close()
    config.plotter.plot_end()
