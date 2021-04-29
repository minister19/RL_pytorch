import math
import numpy as np

from .base_env import BaseEnv

'''
Description:
    Two dimension maze is a square for agent to move left/right/up/down until get out (right-bottom-corner).
Source:
    Shuang Gao
Observation:
    Type: Box(2)
    Num     Obersvation     Min     Max
    0       Position x      0       9
    1       Position y      0       9
Actions:
    Type: Discrete(4)
    Num   Action
    0     Move left
    1     Move right
    2     Move up
    3     Move down
Reward:
    Type: Discrete(4)
    Reward      Reason
    -10         Moved to trap (7,7)
    -1          Stand still
    0           Moved to non-trap
    100         Terminate
Starting State:
    Position is assigned a uniform random value in (x, y) with x/y drops in [0..9]
Episode Termination:
    Position = (9, 9)
    Steps >= 100
'''


class TwoDimensionMaze(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.maze_length = 10  # .....x...., x is the position
        self.states_dim = self.maze_length**2
        self.actions_dim = 4  # move left/right/up/down
        self.posi_x = None
        self.posi_y = None
        self.steps = None

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
        next_state = self.posi_to_state(self.posi)

        # 4. update reward basing on next state
        if next_state == [7, 7]:
            reward = -10
        elif next_state == state:
            reward = -1
        elif next_state == self.terminal:
            reward = 100
        else:
            reward = 0

        # 5. test if done
        if next_state == self.terminal or self.steps >= 100:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.posi = self.entry
        self.steps = 0
        state = self.posi_to_state(self.posi)
        return self._unsqueeze_tensor(state)

    def render(self):
        maze = '.' * self.maze_length**2
        posi = 7*7-1
        maze = maze[posi:posi+1] + 'o' + maze[posi+1:]
        posi = self.posi_x*self.posi_y-1
        maze = maze[posi:posi+1] + 'x' + maze[posi+1:]
        print(f'\r{maze}', end='')

    def close(self):
        pass
