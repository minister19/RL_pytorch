import math
import torch
from rl_m19.config import core
from rl_m19.network.fully_connected import FullyConnected
from rl_m19.utils import Logger, Plotter
from rl_m19.agent.dqn_agent import DQNAgent
from cart_pole_v0_env import CartPole_v0

config = core.Config([core.Core, core.DQN])

config.logger = Logger()
config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = CartPole_v0(config.device)
config.test_env = CartPole_v0(config.device)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.episode_lifespan = 10**3
config.episodes = 1000
# config.eps_fn = lambda s: 0.9
config.eps_fn = lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000)

config.replay_size = 10**5
config.batch_size = 10**2
config.gamma = 0.999
config.lr = 0.001
config.target_update_freq = 10

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
    agent = DQNAgent(config)
    agent.episodes_learn(step_render=True)
    config.env.render()
    config.env.close()
    config.plotter.plot_end()
