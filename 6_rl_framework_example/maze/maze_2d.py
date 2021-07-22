import random
from rl_m19.envs import BaseEnv

'''
Description:
    Two dimension maze is a square for agent to move left/right/up/down until get out (right-bottom-corner).
Source:
    Shuang Gao
Observation:
    Type: Box
    Num     Obersvation     Min     Max
    1       Position x      0       9
    2       Position y      0       9
Actions:
    Type: Discrete
    Num   Action
    1     Move left
    2     Move right
    3     Move up
    4     Move down
Reward:
    Type: Discrete
    Reward      Reason
    -10         Moved to trap (6,6)
    -1          Stand still
    0           Moved to non-trap
    100         Terminate
Starting State:
    Position is assigned a uniform random value in (x, y) with x/y drops in [0..9]
Episode Termination:
    Position = (9, 9)
    Steps >= 1000
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
        self.posi_x = random.randint(0, self.maze_length-1)
        self.posi_y = random.randint(0, self.maze_length-1)
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
