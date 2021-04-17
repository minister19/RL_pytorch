import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from .base_env import BaseEnv
from .futures.account import Account
from .futures.frequency import Frequency

'''
Observations:
0   price, i.e. bar.close
1   volume
    1. original data
    2. 0, aims to be ignored by network
2   qianlon
3   qianlon_ma
4   rsi_ma
5   rsi_smh_ma

Actions v1:
0   Wait
    平仓观察
1   Positive_1
2   Positive_2
3   Positive_3
    立即持多仓
4   Negative_1
5   Negative_2
6   Negative_3
    立即持空仓

Actions v2:
0   Wait
    平仓观察
1   Unilateral
    单边，boll 15%置信区间开仓
2   Oscillated
    振荡，boll 95%置信区间开仓

Reward:
Change in decimal (rather than %) of account's fund.

Starting state:
Ignore the first n points (due to Exponential Moving Average, Simple Moving Average, etc.)

Episode termination:
1. freq's data stream out.
2. account's fund goes below 50%, i.e. reward <= -0.5.
'''


class FuturesCTP(BaseEnv):
    def __init__(self, device, account, freq):
        super().__init__(device)
        self.account = account
        self.freq = freq
        self.states_dim = 4
        self.actions_dim = 7
        self.step_to_run = len(freq.data.open) - 1
        self.step_index = None

    def __get_state(self):
        i = self.step_index
        d = self.freq.data
        # price = d.close[i]
        # volume = d.volume[i]
        # ql = d.qianlon.lon[i]
        # ql_ma = d.qianlon.lon_ma[i]
        # rsi_ma = d.rsi.rsi_ma[i]
        # rsi_smh_ma = d.rsi.rsi_smh_ma[i]
        # return [price, volume, ql, ql_ma, rsi_ma, rsi_smh_ma]

        d1 = d.bbands.location[i]
        d2 = d.period.period[i]
        d3 = d.qianlon.region[i]
        d4 = d.qianlon.ma_isup[i]
        return [d1, d2, d3, d4]

    def __get_current_price(self):
        i = self.step_index
        d = self.freq.data
        price = d.close[i]
        return price

    def step(self, action):
        # 1. take action
        self.account.take_action_v1(action)
        # self.account.take_action_v2(action, self.freq.data)

        # 2. get next state
        self.step_index += 1
        next_state = self.__get_state()

        # 3. update reward basing on next state
        reward = self.account.update_reward(self.__get_current_price()) / self.account.fund * 100

        # 4. test if done
        if self.step_index >= self.step_to_run:
            done = True
        elif self.account.fund < 0.50*self.account.init_fund:
            done = True
        else:
            done = False

        # 5. currently, info is None
        info = None

        return self.unsqueeze_tensor(next_state), self.unsqueeze_tensor(reward), done, info

    def reset(self):
        self.step_index = self.freq.data.ignore_len  # Ignore the first n points
        self.account.reset(self.__get_current_price())
        state = self.__get_state()
        return self.unsqueeze_tensor(state)

    def render(self):
        close = self.freq.data.close[self.freq.data.ignore_len:self.step_index]
        reward = self.account.reward

        fig1 = plt.figure(1)
        plt.clf()
        ax1 = fig1.add_subplot(111)

        # 2020-08-20 Shawn: TODO: 打印开平仓标志，便于分析持仓时间。
        color = 'tab:red'
        ax1.set_xlabel('Time (' + self.freq.freq + ')')
        ax1.set_ylabel('Close', color=color)
        ax1.plot(close, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Reward', color=color)  # we already handled the x-label with ax1
        ax2.plot(reward, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig1.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.pause(0.001)

    def close(self):
        state = np.zeros(self.states_dim)
        return self.unsqueeze_tensor(state)
