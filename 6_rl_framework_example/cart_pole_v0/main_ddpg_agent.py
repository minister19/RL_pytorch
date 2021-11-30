import torch
from rl_m19.config import core
from rl_m19.utils import Logger, Plotter
from rl_m19.agent.ddpg_agent import DDPGAgent
from cart_pole_v0_env import CartPole_v0

config = core.Config([core.Core, core.DDPG])

config.logger = Logger()
config.plotter = Plotter()
# config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.device = torch.device("cpu")
config.env = CartPole_v0(config.device)
config.test_env = CartPole_v0(config.device)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.episode_lifespan = 500
config.test_episode_lifespan = 500

config.replay_size = 10**4
config.batch_size = 32
config.pi_lr = 0.001
config.q_lr = 0.001
config.gamma = 0.99
config.polyak = 0.98

config.epoch_lifespan = 10**3*5
config.epochs = 100
config.init_wander = 10**3*1
config.act_noise = 0
config.update_after = 10**3
config.update_every = 10**2

if __name__ == '__main__':
    agent = DDPGAgent(config)
    agent.episodes_learn(step_render=True)
    config.env.render()
    config.env.close()
    config.plotter.plot_end()
