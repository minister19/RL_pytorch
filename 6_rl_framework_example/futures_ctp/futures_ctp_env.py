import asyncio
import copy
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
    1       Emas trend                      -1/0/1
    2       Emas support                    -1/0/1
    3       Qianlon sign                    -1/0/1
    4       Qianlon trend                   -1/0/1
    5       Qianlon vel sign                -1/0/1
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
        self.steps = 0
        self.rewards = []
        self.account = Account()
        self.backtest_data = BacktestData()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.backtest_data.sync())

    def __get_state(self):
        s1 = copy.copy(self.backtest_data.state[1:])
        s2 = copy.copy(self.account.state[1:])
        return s1.extend(s2)

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        self.account.take_action(action, state[0])

        # 3. get next state
        self.backtest_data.forward()
        self.account.update_margin(state[0])
        next_state = self.__get_state()

        # 4. update reward basing on next state
        margin = self.account.margins[-1]
        if margin >= 0:
            reward = 1 + margin
        else:
            reward = margin
        self.rewards.append(reward)

        # 5. test if done
        if self.account.terminated or self.backtest_data.terminated:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.steps = 0
        self.account.reset()
        self.backtest_data.reset()
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        print(self.account.actions)
        print(self.account.margins)
        print(self.rewards)

    def close(self):
        pass
