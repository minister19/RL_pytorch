import numpy as np
import torch
from itertools import count
from rl_m19.network.replay_memory import ReplayMemory
from rl_m19.agent.base_agent import BaseAgent, AgentUtils


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.select_action_counter = 0

        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = config.optimizer
        self.loss_fn = config.loss_fn

        self.train_memory = ReplayMemory(config.MC)
        self.train_loss = []
        self.train_losses = []

        self.test_memory = ReplayMemory(config.MC)
        self.test_loss = []
        self.test_losses = []

    def select_action(self, state):
        sample = np.random.random()
        eps = self.eps_fn(self.select_action_counter)
        self.select_action_counter += 1
        if sample > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # t.argmax() will return largest value's index of t.
                action = self.policy_net(state).argmax() % self.config.action_dim
        else:
            action = np.random.randint(0, self.config.action_dim)
        return torch.tensor([[action]], device=self.config.device, dtype=torch.long)

    def sample_minibatch(self, memory: ReplayMemory):
        # sample batch
        batch = memory.sample2Batch(self.config.BATCH_SIZE)
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

        q_next = torch.zeros(self.config.BATCH_SIZE, device=self.config.device)
        q_next[non_final_mask] = self.target_net(non_final_next_states).max(1).values.detach()

        q_target = reward_batch + self.config.GAMMA * q_next

        return q_eval, q_target.unsqueeze(1)

    # q_eval, q_target: torch.tensor
    def gradient_descent(self, q_eval, q_target):
        loss = self.loss_fn(q_eval, q_target)  # compute loss
        self.optimizer.zero_grad()
        loss.backward()
        # 2020-08-13 Shawn: Sometimes, no clamp is better.
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def episode_learn(self, i_episode, step_render=True):
        state = self.config.env.reset()

        for t in count():
            if step_render:
                self.config.env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            self.train_memory.push(state, action, reward, next_state)

            if len(self.train_memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch(self.train_memory)

                # gradient descent
                loss = self.gradient_descent(q_eval, q_target)
                self.train_loss.append(loss)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                avg_loss = np.average(self.train_loss) if len(self.train_loss) > 0 else None
                self.train_losses.append(avg_loss)
                self.train_loss.clear()
                break
            else:
                # update state
                state = next_state

    def episode_test(self, i_episode, step_render=True):
        state = self.config.test_env.reset()

        for t in count():
            if step_render:
                self.config.test_env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.test_env.step(action.item())

            # store transition
            self.test_memory.push(state, action, reward, next_state)

            if len(self.test_memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch(self.test_memory)

                # compute loss
                loss_t = self.loss_fn(q_eval, q_target)
                loss = loss_t.item()
                self.test_loss.append(loss)

            if done or t >= self.config.episode_lifespan:
                self.test_memory.clear()
                avg_loss = np.average(self.test_loss) if len(self.test_loss) > 0 else None
                self.test_losses.append(avg_loss)
                self.test_loss.clear()
                break
            else:
                # update state
                state = next_state

    def on_episode_done(self):
        self.config.plotter.plot_single_with_mean({
            'id': 1,
            'title': 'episode_t',
            'xlabel': 'iteration',
            'ylabel': 'lifespan',
            'x_data': range(len(self.episode_t)),
            'y_data': self.episode_t,
            'm': 100
        })

        self.config.plotter.plot_multiple({
            'id': 'loss',
            'title': 'loss',
            'xlabel': 'step',
            'ylabel': ['train_loss', 'test_loss'],
            'x_data': [range(len(self.train_losses)), range(len(self.test_losses))],
            'y_data': [self.train_losses, self.test_losses],
            'color': ['blue', 'red'],
        })

    def episodes_learn(self):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode)
            if i_episode % self.config.TUF == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.episode_test(i_episode)

            self.on_episode_done()
