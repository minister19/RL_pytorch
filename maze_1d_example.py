import math
import torch
from rl_m19.config import Config
from rl_m19.agent import QLearningAgent, SarsaAgent
from rl_m19.envs import OneDimensionMaze
from rl_m19.network import ReplayMemory, PureLinear
from rl_m19.utils import Plotter

config = Config()
config.episode_lifespan = 1000
config.episodes = 1000
config.BATCH_SIZE = 64
config.GAMMA = 0.999
# config.EPS_fn = lambda s: 0.9
config.EPS_fn = lambda s: 0.05 + (0.9 - 0.05) * math.exp(-1. * s / 1000)
config.LR = 0.1  # LEARNING_RATE
config.MC = 1000  # MEMORY_CAPACITY
config.TUF = 10  # TARGET_UPDATE_FREQUENCY

config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = OneDimensionMaze(config.device)
config.states_dim = config.env.states_dim
config.actions_dim = config.env.actions_dim

config.memory_fn = lambda: ReplayMemory(config.MC)
config.policy_net_fn = lambda: PureLinear(config)
config.target_net_fn = lambda: PureLinear(config)
config.optimizer_fn = torch.optim.RMSprop
config.loss_fn = torch.nn.MSELoss()

agent = QLearningAgent(config)
# agent = SarsaAgent(config)
agent.episodes_learn()
config.env.render()
config.env.close()
config.plotter.plot_end()
