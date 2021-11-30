import math
import torch
from rl_m19.config import Config
from rl_m19.agent import QLearningAgent, SarsaAgent
from rl_m19.network import PureLinear
from rl_m19.utils import Plotter
from maze_2d import TwoDimensionMaze

config = Config()
config.episode_lifespan = 10**3
config.episodes = 10**5
config.batch_size = 64
config.gamma = 0.999
# config.eps_fn = lambda s: 0.9
config.eps_fn = lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000)
config.lr = 0.001
config.replay_size = 1000
config.target_update_freq = 10

config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = TwoDimensionMaze(config.device)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.policy_net = PureLinear(config)
config.target_net = PureLinear(config)
config.optimizer = torch.optim.RMSprop(config.policy_net.parameters(), config.lr)
config.loss_fn = torch.nn.MSELoss()


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == 'sarsa':
        agent = SarsaAgent(config)
    else:
        agent = QLearningAgent(config)
    agent.episodes_learn()
    config.env.render()
    config.env.close()
    config.plotter.plot_end()
