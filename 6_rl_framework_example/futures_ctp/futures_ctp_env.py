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
Observation - Account:
    Type: Box
    Num     Obersvation     Min     Max     Discrete
    0       Fund            -inf    inf     
    1       Position                        -1.0/-0.5/0/0.5/1.0
    2       Margin (%)                      -2/-1/0/1/2
Observation - Indicators:
    Type: Box
    Num     Obersvation     Min     Max     Discrete
    0       id
    1       close
    2       Emas trend                      -1/0/1
    3       Emas support                    -1/0/1
    4       Qianlon sign                    -1/0/1
    5       Qianlon trend                   -1/0/1
    6       Qianlon vel sign                -1/0/1
    7       Boll sig                        -4/-3/-2/0/2/3/4
    8       Period sig                      -2/-1/0/1/2
    9       RSI sig                         -2/-1/0/1/2
    10      Withdraw                        -1/0/1
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
        self.account = Account()
        self.backtest_data = BacktestData()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.backtest_data.sync())
        self.states_dim = 2 + len(self.backtest_data.indicators)*2  # postion, margin, indicators along with feedback
        self.actions_dim = 5
        self.steps = 0
        self.rewards = []

    def __get_state(self):
        s1 = copy.copy(self.account.state[1:])
        s2 = copy.copy(self.backtest_data.state[1:])
        s3 = s1 + s2
        return s3

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        self.account.take_action(action, self.backtest_data.state[1])

        # 3. get next state
        self.backtest_data.forward()
        self.account.update_margin(self.backtest_data.state[1])
        next_state = self.__get_state()

        # 4. update reward basing on next state
        margin = self.account.margins[-1] * 100
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
        self.rewards.clear()
        self.account.reset()
        self.backtest_data.reset()
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        # print(self.account.actions)
        # print(self.account.margins)
        # 2020-08-18 Shawn: 打印 reward 曲线，验证指标.
        # print(self.rewards)
        pass

    def close(self):
        pass
