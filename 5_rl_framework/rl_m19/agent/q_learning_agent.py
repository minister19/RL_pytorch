import numpy as np
import pandas as pd
from itertools import count
from rl_m19.agent.core import BaseAgent, AgentUtils


class QLearningAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.eps_steps = 0
        self.episode_t = []

        self.actions = range(self.config.action_dim)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)

    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            new_state_row = pd.Series(
                np.zeros(self.config.action_dim),
                index=self.actions,
                name=state,
            )
            self.q_table = self.q_table.append(new_state_row)

    def select_action(self, state):
        # convert tensor to string, number
        state = AgentUtils.tensor2str(state.cpu())

        # record unknown state
        self._check_state_exist(state)

        sample = np.random.random()
        eps = self.config.eps_fn(self.eps_steps)
        self.eps_steps += 1
        if sample > eps:
            # some actions may have the same value, randomly choose on in these actions
            state_action = self.q_table.loc[state, :]
            best_actions = state_action[state_action == max(state_action)].index
            action = np.random.choice(best_actions)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state, next_action):
        # convert tensor to string, number
        state = AgentUtils.tensor2str(state.cpu())
        reward = AgentUtils.tensor2number(reward.cpu())
        next_state = AgentUtils.tensor2str(next_state.cpu())

        # record unknown state
        self._check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        if next_state != None:
            # next state is not terminal
            q_target = reward + self.config.gamma * self.q_table.loc[next_state, :].max()
        else:
            # next state is terminal
            q_target = reward
        self.q_table.loc[state, action] += self.config.lr * (q_target - q_predict)

    def episode_learn(self, i_episode, step_render=False):
        state = self.config.env.reset()

        for t in count():
            if step_render:
                self.config.env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                break
            else:
                # learn
                self.learn(state, action, reward, next_state, None)

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

    def episodes_learn(self, step_render=False):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode, step_render)

            self.on_episode_done()
