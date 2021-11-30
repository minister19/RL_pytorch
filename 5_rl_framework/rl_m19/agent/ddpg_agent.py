import logging
import numpy as np
import torch
from copy import deepcopy
from rl_m19.network import core
from rl_m19.network.replay_memory import ReplayMemory
from rl_m19.network.ddpg import DDPGActorCritic, ReplayBuffer
from rl_m19.agent.core import BaseAgent, AgentUtils


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.episode_t = []

        # Create actor-critic module and target networks
        # mlp_out = math.ceil(math.log(2, 2))
        self.ac = DDPGActorCritic(config.state_dim, config.action_dim, (256, 256, ), config.device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.memory = ReplayMemory(config.replay_size)
        # self.memory = ReplayBuffer(config.state_dim, config.action_dim, config.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=config.pi_lr)
        self.q_optimizer = torch.optim.Adam(self.ac.q.parameters(), lr=config.q_lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        logging.info(f'var_counts: {var_counts}')

    def select_action(self, state: torch.Tensor, is_random=False, noise_scale=0):
        if is_random:
            action = torch.empty(self.config.action_dim, device=self.config.device)
            action.uniform_(-2, 2).unsqueeze_(0)
        else:
            action = self.ac.act(state)
            action += noise_scale * torch.randn(self.config.action_dim, device=self.config.device)
        action_argmax = action.argmax() % self.config.action_dim
        return action, action_argmax

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['state_batch'], data['action_batch'], data['reward_batch'], data['next_state_batch'], data['done_batch']

        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.config.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['state_batch']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def sample_batch(self):
        batch = self.memory.sample_batch(self.config.batch_size)
        data = {
            "state_batch": torch.cat(batch.state),
            "action_batch": torch.cat(batch.action),
            "reward_batch": torch.cat(batch.reward),
            "next_state_batch": torch.cat(batch.next_state),
            "done_batch": torch.cat(batch.done)
        }
        return data

    def gradient_descent(self, data):
        # First run one gradient descent step for Q.
        loss_q, loss_info = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        # for param in self.ac.q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        loss_pi = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        # for param in self.ac.pi.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)
                # p_targ.data.clamp_(-1, 1)

    def test_agent(self, step_render=False):
        state, done, ep_ret, ep_len = self.config.test_env.reset(), False, 0, 0
        while not(done or (ep_len == self.config.test_episode_lifespan)):
            if step_render:
                self.config.test_env.render()

            # Take deterministic actions at test time (noise_scale=0)
            action, action_argmax = self.select_action(state, False, 0)

            state, reward, done, info = self.config.test_env.step(np.array(action))
            ep_ret += reward
            ep_len += 1

    def on_episode_done(self):
        self.config.plotter.plot_single_with_mean({
            'id': 'episode_t',
            'title': 'episode_t',
            'xlabel': 'iteration',
            'ylabel': 'lifespan',
            'x_data': range(len(self.episode_t)),
            'y_data': self.episode_t,
            'm': 100
        })

        # self.config.plotter.plot_single_with_mean({
        #     'id': 'loss',
        #     'title': 'loss',
        #     'xlabel': 'iteration',
        #     'ylabel': 'loss',
        #     'x_data': range(len(self.train_loss)),
        #     'y_data': self.train_loss,
        #     'm': 10
        # })

    # Ref: relationship between epoch and episode.
    # https://stats.stackexchange.com/questions/250943/what-is-the-difference-between-episode-and-epoch-in-deep-q-learning
    def episodes_learn(self, step_render=False):
        total_steps = self.config.epoch_lifespan * self.config.epochs
        state, ep_ret, ep_len = self.config.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            if step_render:
                self.config.env.render()

            # Until init_wander have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            is_random = (t <= self.config.init_wander)
            action, action_argmax = self.select_action(state, is_random, self.config.act_noise)

            # Step the env
            # next_state, reward, done, info = self.config.env.step(action.item())
            next_state, reward, done, info = self.config.env.step(action_argmax.item())
            ep_ret += reward
            ep_len += 1

            done |= (ep_len == self.config.episode_lifespan)

            # Store experience to replay buffer
            r = torch.as_tensor(reward, dtype=torch.float32, device=self.config.device).unsqueeze(0)
            d = torch.as_tensor(done, dtype=torch.float32, device=self.config.device).unsqueeze(0)
            # r = reward
            # d = done
            self.memory.push(state, action, r, next_state, d)

            # Super critical, easy to overlook step: make sure to update most recent observation!
            state = next_state

            # End of trajectory handling
            if done:
                print(ep_ret)
                self.episode_t.append(ep_len)
                # self.on_episode_done()
                state, ep_ret, ep_len = self.config.env.reset(), 0, 0

            # Update handling
            # if t >= self.config.update_after and t % self.config.update_every == 0:
            #     for _ in range(self.config.update_every):

            if t >= self.config.batch_size:
                # sample batch
                data = self.sample_batch()

                # compute loss, gradient descent
                self.gradient_descent(data)

            # End of epoch handling
            if (t+1) % self.config.epoch_lifespan == 0:
                epoch = (t+1) // self.config.epoch_lifespan

                # Test the performance of the deterministic version of the agent.
                # self.test_agent(step_render)
