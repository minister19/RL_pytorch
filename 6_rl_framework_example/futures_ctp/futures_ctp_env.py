import asyncio
import copy
from rl_m19.envs import BaseEnv
from account import ActionTable, Account
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
    2       Margin (%)      -inf    inf     -2/-1/0/1/2
Observation - Indicators:
    Type: Box
    Num     Obersvation     Min     Max     Discrete
    0       kline                           -
    1       Emas trend                      -1/0/1
            Emas support                    -1/0/1
    2       Qianlon sign                    -1/0/1
            Qianlon vel sign                -1/0/1
    3       Boll sig                        -4/-3/-2/0/2/3/4
    4       Period sig                      -2/-1/0/1/2
    5       RSI sig                         -2/-1/0/1/2
    6       RSV trend                       -1/0/1
            RSV sig                         -1/0/1
    7       Withdraw sig                    -1/0/1
Actions:
    Type: Discrete
    Num     Action
    0       Long 0.5
    1       Long 1.0
    2       Short 0.5
    3       Short 1.0
    4       Neutral
Reward:
    Consider steps and margin, reward = margin + ACTION_PENALTY
Starting State:
    Indicators = history[60] (skip EMA, SMA's beginning values, 5*6*2=60)
    Fund = 1.0
    Position = 'N'
    Margin = 0
Episode Termination:
    Indicators = history[-1]
    Fund <= 0.5
'''


class FuturesCTP(BaseEnv):
    def __init__(self, device, plotter=None) -> None:
        super().__init__(device, plotter)
        self.account = Account()
        self.train_data = BacktestData()
        asyncio.run(self.train_data.sync())
        self.states_dim = self.account.states_dim + self.train_data.states_dim
        self.actions_dim = self.account.actions_dim
        self.steps = 0
        self.__action_long_pc = None
        self.__action_short_pc = None
        self.__action_neutral_pc = None

    def __get_state(self):
        s1 = copy.copy(self.account.states[1:])
        s2 = copy.copy(self.train_data.states[1:])
        return s1 + s2

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        self.account.take_action(action, self.train_data.states[0]['close'])

        # 3. get next state
        self.train_data.forward()
        self.account.update_margin(self.train_data.states[0]['close'])
        next_state = self.__get_state()

        # 4. update reward basing on next state
        if not self.account.action_transits:
            reward = (self.account.nominal_margin + Account.TRADE_FEE) * 100
        else:
            reward = (self.account.nominal_margin - self.account.action_penalty) * 100

        # 5. test if done
        if self.account.terminated or self.train_data.terminated:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.account.reset()
        self.train_data.reset()
        self.steps = 0
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        _actions = self.account.actions_real
        _fund_totals = self.account.fund_totals
        _klines = self.train_data.klines
        # self.render_klines_and_funds(_actions, _fund_totals, _klines)
        # self.render_actions_and_funds(_actions, _fund_totals, _klines)

    def render_klines_and_funds(self, _actions, _fund_totals, _klines):
        partial = len(_actions)
        time = range(partial)
        close = []
        for i in time:
            close.append(_klines[i]['close'])
        fund_totals = _fund_totals[0:partial]
        axes = self.plotter.plot_multiple({
            'id': 'kline',
            'title': 'kline',
            'xlabel': 'time',
            'ylabel': ['close', 'fund_total'],
            'x_data': [time, time],
            'y_data': [close, fund_totals],
            'color': ['blue', 'red'],
        })
        return

    def render_actions_and_funds(self, _actions, _fund_totals, _klines):
        partial = len(_actions)
        actions = [[], []]
        fund_totals_step = [[], []]
        action_long = [[], []]
        action_short = [[], []]
        action_neutral = [[], []]
        time = range(partial)
        for i in time:
            if i == 0 or _actions[i-1] != _actions[i]:
                actions[0].append(i)
                actions[1].append(_klines[i]['close'])
                fund_totals_step[0].append(i)
                fund_totals_step[1].append(_fund_totals[i])
                if ActionTable[_actions[i]].posi == 'L':
                    action_long[0].append(i)
                    action_long[1].append(_klines[i]['close'])
                elif ActionTable[_actions[i]].posi == 'S':
                    action_short[0].append(i)
                    action_short[1].append(_klines[i]['close'])
                elif ActionTable[_actions[i]].posi == 'N':
                    action_neutral[0].append(i)
                    action_neutral[1].append(_klines[i]['close'])
        if self.__action_long_pc is not None:
            self.__action_long_pc.remove()
        if self.__action_short_pc is not None:
            self.__action_short_pc.remove()
        if self.__action_neutral_pc is not None:
            self.__action_neutral_pc.remove()
        axes = self.plotter.plot_multiple({
            'id': 'action',
            'title': 'action',
            'xlabel': 'time',
            'ylabel': ['action', 'fund_totals_step'],
            'x_data': [actions[0], fund_totals_step[0]],
            'y_data': [actions[1], fund_totals_step[1]],
            'color': ['blue', 'red'],
        })
        self.__action_long_pc = self.plotter.plot_scatter({
            'id': 'action',
            'axes': axes[0],
            'x_data': action_long[0],
            'y_data': action_long[1],
            's': 25,
            'c': 'red',
            'marker': '^'
        })
        self.__action_short_pc = self.plotter.plot_scatter({
            'id': 'action',
            'axes': axes[0],
            'x_data': action_short[0],
            'y_data': action_short[1],
            's': 25,
            'c': 'green',
            'marker': 'v'
        })
        self.__action_neutral_pc = self.plotter.plot_scatter({
            'id': 'action',
            'axes': axes[0],
            'x_data': action_neutral[0],
            'y_data': action_neutral[1],
            's': 25,
            'c': 'orange',
            'marker': 'o'
        })
        return

    def close(self):
        return
