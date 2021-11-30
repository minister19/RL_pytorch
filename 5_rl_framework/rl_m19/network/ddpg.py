import numpy as np
import torch
import torch.nn as nn
from rl_m19.network import core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, device=None):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = core.mlp(pi_sizes, bias=False, activation=nn.ReLU, output_activation=nn.Tanh)
        self.to(device)

    def forward(self, obs):
        return self.pi(obs)


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, device=None):
        super().__init__()
        q_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q = core.mlp(q_sizes, bias=False, activation=nn.ReLU, output_activation=nn.Identity)
        self.to(device)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class DDPGActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), device=None):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, device)
        self.q = QFunction(obs_dim, act_dim, hidden_sizes, device)
        self.to(device)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 21
    action_dim = 2
    ddpg = DDPGActorCritic(state_dim, action_dim, (state_dim//2, ), device=device)
    x = torch.rand((1, state_dim), device=device)
    print(x)
    y = ddpg.act(x)
    print(y)
    z = y.cpu().numpy()
    print(z)
