import random
from rl_m19.envs import BaseEnv
from account import Account
from backtest_data import BacktestData

'''
Description:
    Futures CTP is a trading account for trader to open/close short/long positions.
Source:
    Shuang Gao
Observation - Indicators:
    Type: Box
    Num     Obersvation     Min     Max     Discrete
    1       Emas sign                       -1/0/1
    2       Emas support                    -1/0/1
    3       Qianlon sign                    -1/0/1
    4       Qianlon trend                   -1/0/1
    5       Qianlon vel                     -1/0/1
    6       Boll sig                        -4/-3/-2/0/2/3/4
    7       Period sig                      -2/-1/0/1/2
    8       RSI sig                         -2/-1/0/1/2
    9       Withdraw                        -1/0/1
Observation - Account:
    Type: Box
    Num     Obersvation     Min     Max     Discrete
    0       Fund            -inf    inf     
    1       Position                        -1.0/-0.5/0/0.5/1.0
    2       Margin (%)                      -2/-1/0/1/2
Actions:
    Type: Discrete
    Num     Action
    0       Long 0.5
    1       Long 1.0
    2       Short 0.5
    3       Short 1.0
    4       Neutral
Reward:
    Consider steps and margin, reward = 1 + margin, 1 for if margin >=0, one step forward)
Starting State:
    Indicators = history[100] (skip EMA, SMA's beginning values)
    Fund = 10K (ignored because of continuity)
    Position = 0
    Margin = 0
Episode Termination:
    Indicators = history[-1]
    Fund <= 5K
'''


class FuturesCTP(BaseEnv):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.states_dim = 9*2 + 2  # Indicators along with feedback, Postion and Margin
        self.actions_dim = 5
        self.steps = None
        self.rewards = []
        self.account = Account(10000)
        self.backtest_data = BacktestData()

    def __get_state(self):
        self.backtest_data.state

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
        self.rewards.append(reward)

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
