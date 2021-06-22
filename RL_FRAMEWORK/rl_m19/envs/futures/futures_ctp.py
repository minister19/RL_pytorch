import math
import numpy as np
import numpy.random as ran

from ..base_env import BaseEnv
from .account import Account

'''
Description:
    Futures CTP is a trading account for trader to open/close short/long positions.
Source:
    Shuang Gao
Observation - Indicators:
    Type: Box
    Num     Obersvation     Min     Max     Discrete            Sum
    0       Emas trend                      -1/0/1              3
    1       Emas support                    0/1                 2
    2       Qianlon lon                     -1/0/1              3
    3       Qianlon vel                     -1/0/1              3
    4       Boll sig                        -4/-3/-2/0/2/3/4    7
    5       Period sig                      -2/-1/0/1/2         5
    6       RSI sig                         -2/-1/0/1/2         5
Observation - Account:
    Type: Box
    Num     Obersvation     Min     Max     Discrete                        Sum
    0       Fund            -inf    inf                                     
    1       Position                        -1.0/-0.5/0/0.5/1.0             5
    2       Margin (%)                      -9/-7/-5/-3/-1/0/1/3/5/7/9      11
    3       Margin_vel (%)                  -5/-3/-2/-1/-0.5/0/0.5/1/2/3/5  11
Actions:
    Type: Discrete
    Num     Action
    0       Long 0.5
    1       Long 1.0
    2       Short 0.5
    3       Short 1.0
    4       Neither
Reward:
    Consider steps and margin, reward = 1 (one step forward) + margin
Starting State:
    Fund = 10K
    Steps >= 100 (ignore EMA, SMA's beginning values)
Episode Termination:
    Fun <= 5K
    Steps >= len(history data)
'''


class FuturesCTP(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.states_dim = 10**2
        self.actions_dim = 4  # move left/right/up/down
        self.steps = None

    def __get_state(self):
        # TODO: regard qianlon lon/vel ripples ~= 0
        return [self.posi_x, self.posi_y]

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        # 2020-08-18 Shawn: TODO: punish frequent trade.
        if action == 0:
            self.posi_x = max(0, self.posi_x - 1)
        elif action == 1:
            self.posi_x = min(self.maze_length-1, self.posi_x + 1)
        elif action == 2:
            self.posi_y = max(0, self.posi_y - 1)
        elif action == 3:
            self.posi_y = min(self.maze_length-1, self.posi_y + 1)

        # 3. get next state
        next_state = self.__get_state()

        # 4. update reward basing on next state
        if next_state == self.trap:
            reward = -10
        elif next_state == state:
            reward = -5
        elif next_state == self.terminal:
            reward = 100
        else:
            reward = -1

        # 5. test if done
        if next_state == self.terminal or self.steps >= 1000:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.posi_x = ran.randint(self.maze_length)
        self.posi_y = ran.randint(self.maze_length)
        self.steps = 0
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        maze = '.' * self.maze_length**2
        posi = self.trap[0]*self.maze_length + self.trap[1]
        maze = maze[0:posi] + 'o' + maze[posi+1:]
        posi = self.posi_x*self.maze_length + self.posi_y
        maze = maze[0:posi] + 'x' + maze[posi+1:]
        print(maze)

    def close(self):
        pass
