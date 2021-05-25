import math
import numpy as np
import numpy.random as ran

from .base_env import BaseEnv

'''
Description:
    Future CTP is a trading account for trader to open/close short/long positions.
Source:
    Shuang Gao
Observation:
    Type: Box(10)
    Num     Obersvation     Min     Max     Discrete
    0       Fund            -inf    inf
    1       Position                        -1.0/-0.5/0/0.5/1.0
    2       Margin (%)      -inf    inf
    3       Emas                            -1/0/1
    4       Mas                             -1/0/1
    5       Qianlon lon                     -1/0/1
    6       Qianlon vel                     -1/0/1
    7       Boll sig                        -4/-3/-2/0/2/3/4
    8       Period sig                      -2/-1/0/1/2
    9       RSI sig                         -2/-1/0/1/2
    开仓以后价差百分比化（止损）也应作为状态观察
Actions:
    Type: Discrete(6)
    Num   Action
    0     Open long 0.5
    1     Open long 1.0
    2     Open short 0.5
    3     Open short 1.0
    4     Close long
    5     Close short
    6     Reverse short to long 0.5
    7     Reverse short to long 1.0
    7     Reverse long to short 0.5
    8     Reverse long to short 1.0
    9     Standby
Reward:
    Type: Discrete(3)
    Reward      Reason
    0           abs(Margin) <= 1.0
    Margin      Margin < 1.0
    Margin      Margin > 1.0
Starting State:
    Fund = 10K
    Steps >= 100 (ignore EMA, SMA's beginning values)
Episode Termination:
    Fun <= 5K
    Steps >= len(history data)
'''

# TODO: research action Close short/long 0.5
# TODO: research when should qianlon lon/vel ~= 0


class FuturesCTP(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.states_dim = self.maze_length**2
        self.actions_dim = 4  # move left/right/up/down
        self.steps = None

    @property
    def trap(self): return [6, 6]

    @property
    def terminal(self): return [self.maze_length-1, self.maze_length-1]

    def __get_state(self):
        return [self.posi_x, self.posi_y]

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
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
