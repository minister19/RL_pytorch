import math
import numpy as np
import numpy.random as ran

from .base_env import BaseEnv

'''
Description:
    One dimension maze is a line for agent to move left/right until get out (right-most).
Source:
    Shuang Gao
Observation:
    Type: Box(1)
    Num     Obersvation     Min     Max
    0       Position        0       9
Actions:
    Type: Discrete(2)
    Num   Action
    0     Move left
    1     Move right
Reward:
    Type: Discrete(4)
    Reward      Reason
    -1          Moved left
    0           Stand still
    1           Moved right
    100         Terminate
Starting State:
    Position is assigned a uniform random value in [0..9]
Episode Termination:
    Position = 9
    Steps >= 100
'''


class OneDimensionMaze(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.maze_length = 10  # .....x...., x is the position
        self.states_dim = self.maze_length
        self.actions_dim = 2  # move left, move right
        self.posi = None
        self.steps = None

    @property
    def entry(self): return [0]

    @property
    def terminal(self): return [self.maze_length - 1]

    def __get_state(self):
        return [self.posi]

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        if action == 0:
            if self.posi > self.entry[0]:
                self.posi -= 1
        elif action == 1:
            if self.posi < self.terminal[0]:
                self.posi += 1

        # 3. get next state
        next_state = self.__get_state()

        # 4. update reward basing on next state
        if next_state == self.terminal:
            reward = 100
        else:
            reward = next_state[0] - state[0]
            # TODO: research that tuning reward such that q table reaches convergence.
            # phenomenon: for Q Learning, Q(state=1, action=0) === -2.0
            # reward = next_state[0] - state[0] - 1

        # 5. test if done
        if next_state == self.terminal or self.steps >= 100:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.posi = ran.randint(self.maze_length)
        self.steps = 0
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        maze = '.' * self.maze_length
        maze = maze[0:self.posi] + 'x' + maze[self.posi+1:]
        print(maze)

    def close(self):
        pass
