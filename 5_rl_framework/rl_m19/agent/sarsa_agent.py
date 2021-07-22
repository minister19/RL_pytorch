import numpy as np
import pandas as pd
import random
from itertools import count
from .q_learning_agent import AgentUtils, QLearningAgent


class SarsaAgent(QLearningAgent):
    def __init__(self, config):
        super().__init__(config)

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
            q_target = reward + self.config.GAMMA * self.q_table.loc[next_state, next_action]
        else:
            # next state is terminal
            q_target = reward
        self.q_table.loc[state, action] += self.config.LR * (q_target - q_predict)

    def episode_learn(self, i_episode):
        state = self.config.env.reset()

        for t in count():
            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action)

            # choose next_action
            next_action = self.select_action(state)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                self.config.plotter.plot_list_ndarray(self.episode_t, 100)
                break
            else:
                # learn
                self.learn(state, action, reward, next_state, next_action)

                # update state
                state = next_state
