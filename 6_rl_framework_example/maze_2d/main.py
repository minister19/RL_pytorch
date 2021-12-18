import math
import torch
from rl_m19.config import core
from rl_m19.network.fully_connected import FullyConnected
from rl_m19.utils import Logger, Plotter
from rl_m19.agent.q_learning_agent import QLearningAgent
from rl_m19.agent.sarsa_agent import SarsaAgent
from maze_2d import TwoDimensionMaze

config = core.Config([core.Core, core.Q])

config.logger = Logger()
config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = TwoDimensionMaze(config.device)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.policy_net = FullyConnected(config.state_dim,
                                   config.action_dim,
                                   hidden_sizes=(config.state_dim*4, config.state_dim*4, ),
                                   device=config.device)
config.target_net = FullyConnected(config.state_dim,
                                   config.action_dim,
                                   hidden_sizes=(config.state_dim*4, config.state_dim*4, ),
                                   device=config.device)
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
