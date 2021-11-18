import math
import torch
from rl_m19.config import Config
from rl_m19.network import PureLinear
from rl_m19.utils import Plotter
from cart_pole_v0_env import CartPole_v0
from dqn_agent_ext import DQNAgentExt

config = Config()
config.episode_lifespan = 10**3
config.episodes = 10**3
config.BATCH_SIZE = 64
config.GAMMA = 0.999
# config.EPS_fn = lambda s: 0.9
config.EPS_fn = lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000)
config.LR = 0.01  # LEARNING_RATE
config.MC = 1000  # MEMORY_CAPACITY
config.TUF = 5  # TARGET_UPDATE_FREQUENCY

config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = CartPole_v0(config.device)
config.test_env = CartPole_v0(config.device)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.policy_net = PureLinear(config)
config.target_net = PureLinear(config)
config.optimizer = torch.optim.RMSprop(config.policy_net.parameters(), config.LR)
config.loss_fn = torch.nn.MSELoss()


if __name__ == '__main__':
    agent = DQNAgentExt(config)
    agent.episodes_learn()
    config.env.render()
    config.env.close()
    config.plotter.plot_end()
