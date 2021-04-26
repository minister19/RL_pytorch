import numpy as np
import pandas as pd
import random
import time

from itertools import count
from .q_learning_agent import QLearningAgent


class SarsaAgent(QLearningAgent):
    def __init__(self, config):
        super().__init__(config)
        self.actions = range(self.config.actions_dim)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)
        self.select_action_counter = 0

    def learn(self, state, action, reward, next_state, next_action):
        # convert tensor to string, number
        state = self.tensor2str(state.cpu())
        reward = self.tensor2number(reward.cpu())
        reward = self._unify_reward(reward)
        next_state = self.tensor2str(next_state.cpu())

        # record unknown state
        self._check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        if next_state != None:
            # next state is not terminal
            q_target = reward + self.config.GAMMA * self.q_table.loc[next_state, next_action]
        else:
            # next state is terminal
            q_target = reward
        self.q_table.loc[state, action] += self.config.LR * (q_target - q_predict)

    def episode_learn(self, i_episode):
        state = self.config.env.reset()

        for t in count():
            self.config.env.render()
            time.sleep(0.05)

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action)

            # choose next_action
            next_action = self.select_action(state)

            if done or t >= self.config.episode_lifespan:
                self.config.env.render()
                print(f'\r\n{self.q_table}', end='\r\n')

                self.episode_t.append(t)
                # self.config.plotter.plot_list_ndarray(self.episode_t)
                break
            else:
                # learn
                self.learn(state, action, reward, next_state, next_action)

                # update state
                state = next_state
