from .base_env import BaseEnv

'''
Description:
    One dimension maze is a line for agent to move left and right until get out (right-most).
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
    Type: Discrete(2)
    Reward      Reason
    -1          Moved left
    0           Stand still
    1           Moved right
Starting State:
    Position is assigned a uniform random value in [0..9]
Episode Termination:
    Position = 9
    Steps >= 100
'''


class OneDimensionMaze(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.states_dim = 3  # entry-point, in-process, terminal
        self.actions_dim = 2  # move left, move right
        self.maze_length = 10  # .....x...., x is the position
        self.posi = self.entry
        self.steps = 0

    @property
    def entry(self):
        return 0

    @property
    def terminal(self):
        return self.maze_length - 1

    def posi_to_state(self, posi):
        if posi == self.entry:
            state = [0]
        elif 0 < posi < self.terminal:
            state = [1]
        else:
            state = [2]
        return state

    def step(self, action: int):
        # 1. take action
        pre_posi = self.posi
        if action == 0:
            self.posi = max(self.entry, self.posi - 1)
        elif action == 1:
            self.posi = min(self.terminal, self.posi + 1)

        # 2. get next state
        next_state = self.posi_to_state(self.posi)

        # 3. update reward basing on next state
        reward = self.posi - pre_posi

        # 4. test if done
        self.steps += 1
        if self.steps >= 100:
            done = True
        elif self.posi == self.terminal:
            done = True
        else:
            done = False

        # 5. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.posi = self.entry
        self.steps = 0
        state = self.posi_to_state(self.posi)
        return self._unsqueeze_tensor(state)

    def render(self):
        maze = '.' * (self.posi-self.entry) + 'x' + '.' * (self.terminal-self.posi)
        print('\r{}'.format(maze), end='')
        # print(maze)

    def close(self):
        pass
