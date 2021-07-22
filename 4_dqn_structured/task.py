# 2 actions: move left or right.
# 4 inputs：position, velocity, etc.
# reward: better performing scenarios will run for longer duration, accumulating larger return.

# First, let’s import needed packages. Firstly, we need gym for the environment (Install using pip install gym). We’ll also use the following from PyTorch:
# - neural networks (torch.nn)
# - optimization (torch.optim)
# - automatic differentiation (torch.autograd)
# - utilities for vision tasks (torchvision - a separate package).

import gym
import torch
from agent import Agent
from config import Config
from env_wrapper import EnvWrapper
from network import DQN


class Task:
    def __init__(self, name, num_episodes=500):
        self.name = name
        self.num_episodes = num_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(name).unwrapped
        self.env.reset()
        self.env_w = EnvWrapper(self.env, self.device)
        self.cfg = Config()
        self.cfg.n_actions = self.env.action_space.n
        self.cfg.policy_net = DQN(self.env_w.screen_height, self.env_w.screen_width,
                                  self.cfg.n_actions).to(self.device)
        self.cfg.target_net = DQN(self.env_w.screen_height, self.env_w.screen_width,
                                  self.cfg.n_actions).to(self.device)
        self.agent = Agent(self.env, self.env_w, self.device, self.cfg)

    def train(self):
        for i_episode in range(self.num_episodes):
            self.agent.step(i_episode)
        print('Complete')


if __name__ == '__main__':
    task = Task('CartPole-v0', 5000)
    state = task.env.reset()
    task.train()
    task.env.render()
    task.env.close()
    task.env_w.plot_end()
