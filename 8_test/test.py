import gym
import torch

x = torch.rand(2)
print(x)

x.uniform_(0, 2)
print(x)

y = torch.tensor(1)
print(y)

env = gym.make('Pendulum-v1')
print(env)

env = gym.make('CartPole-v0')
print(env)
