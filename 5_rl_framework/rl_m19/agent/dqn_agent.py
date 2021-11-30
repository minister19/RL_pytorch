import numpy as np
import torch
from itertools import count
from rl_m19.network.replay_memory import ReplayMemory
from rl_m19.agent.core import BaseAgent, AgentUtils


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.eps_steps = 0
        self.episode_t = []
        self.iteration_loss = []
        self.train_loss = []

        self.memory = ReplayMemory(config.replay_size)
        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = config.optimizer
        self.loss_fn = config.loss_fn

    def select_action(self, state):
        sample = np.random.random()
        eps = self.config.eps_fn(self.eps_steps)
        self.eps_steps += 1
        if sample > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # t.argmax() will return largest value's index of t.
                action = self.policy_net(state).argmax() % self.config.action_dim
        else:
            action = np.random.randint(0, self.config.action_dim)
        return torch.tensor([[action]], device=self.config.device, dtype=torch.long)

    def sample_batch(self):
        # sample batch
        batch = self.memory.sample_batch(self.config.batch_size)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # evaluation value
        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        # target value
        mask_tuple = tuple(map(lambda s: s is not None, batch.next_state))
        non_final_mask = torch.tensor(mask_tuple, device=self.config.device, dtype=torch.bool)
        # 2020-08-11 Shawn: if done, next_state should be None.
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        q_next = torch.zeros(self.config.batch_size, device=self.config.device)
        q_next[non_final_mask] = self.target_net(non_final_next_states).max(1).values.detach()

        q_target = reward_batch + self.config.gamma * q_next

        # compute loss
        loss = self.loss_fn(q_eval, q_target.unsqueeze(1))

        # record loss
        self.iteration_loss.append(loss.item())

        return loss

    # q_eval, q_target: torch.tensor
    def gradient_descent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # 2020-08-13 Shawn: Sometimes, no clamp is better.
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def episode_learn(self, i_episode, step_render=False):
        state = self.config.env.reset()

        for t in count():
            if step_render:
                self.config.env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            self.memory.push(state, action, reward, next_state, done)

            if len(self.memory) >= self.config.batch_size:
                # sample batch, compute loss
                train_loss = self.sample_batch()

                # gradient descent
                self.gradient_descent(train_loss)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                train_loss = np.average(self.iteration_loss) if len(self.iteration_loss) > 0 else 0
                self.train_loss.append(train_loss)
                self.iteration_loss.clear()
                break
            else:
                # update state
                state = next_state

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

        self.config.plotter.plot_single_with_mean({
            'id': 'loss',
            'title': 'loss',
            'xlabel': 'iteration',
            'ylabel': 'loss',
            'x_data': range(len(self.train_loss)),
            'y_data': self.train_loss,
            'm': 10
        })

    def episodes_learn(self, step_render=False):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode, step_render)

            if i_episode % self.config.target_update_freq == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.on_episode_done()
