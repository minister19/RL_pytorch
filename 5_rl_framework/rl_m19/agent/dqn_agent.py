import torch
import random
from itertools import count
from .base_agent import BaseAgent, AgentUtils


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.memory = config.memory
        self.select_action_counter = 0

    def select_action(self, state):
        sample = random.random()
        eps = self.eps_fn(self.select_action_counter)
        self.select_action_counter += 1
        if sample > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # t.argmax() will return largest value's index of t.
                action = self.policy_net(state).argmax() % self.config.actions_dim
        else:
            action = random.randrange(self.config.actions_dim)
        return torch.tensor([[action]], device=self.config.device, dtype=torch.long)

    def sample_minibatch(self):
        # sample batch
        batch = self.memory.sample2Batch(self.config.BATCH_SIZE)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # evaluation value
        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        # target value
        mask_tuple = tuple(map(lambda s: s is not None, batch.next_state))
        non_final_mask = torch.tensor(mask_tuple, device=self.config.device, dtype=torch.bool)
        # 2020-08-11 Shawn: if done, next_state should be None
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
        self.optimizer.step()

    def episode_learn(self, i_episode):
        state = self.config.env.reset()

        for t in count():
            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            self.memory.push(state, action, reward, next_state)

            if len(self.memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch()

                # gradient descent
                self.gradient_descent(q_eval, q_target)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                self.config.plotter.plot_single_with_mean({
                    'id': 1,
                    'title': 'episode_t',
                    'xlabel': 'iteration',
                    'ylabel': 'lifespan',
                    'x_data': range(len(self.episode_t)),
                    'y_data': self.episode_t,
                    'm': 100
                })
                break
            else:
                # update state
                state = next_state

    def episodes_learn(self):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode)
            if i_episode % self.config.TUF == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())
