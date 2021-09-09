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
    Consider steps and margin, reward = margin + trade_fee
Starting State:
    Indicators = history[60] (skip EMA, SMA's beginning values, 5*6*2=60)
    Fund = 10K (ignored because of continuity)
    Position = 0
    Margin = 0
Episode Termination:
    Indicators = history[-1]
    Fund <= 5K
'''


class FuturesCTP(BaseEnv):
    def __init__(self, device, plotter=None) -> None:
        super().__init__(device, plotter)
        self.account = Account()
        self.backtest_data = BacktestData()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.backtest_data.sync())
        self.states_dim = self.account.states_dim + self.backtest_data.states_dim
        self.actions_dim = self.account.actions_dim
        self.steps = 0
        self.__action_long_pc = None
        self.__action_short_pc = None
        self.__action_neutral_pc = None

    def __get_state(self):
        s1 = copy.copy(self.account.states[1:])
        s2 = copy.copy(self.backtest_data.states[1:])
        s3 = s1 + s2
        return s3

    def step(self, action: int):
        self.steps += 1

        # 1. snapshot state
        state = self.__get_state()

        # 2. take action
        self.account.take_action(action, self.backtest_data.states[0]['close'])

        # 3. get next state
        self.backtest_data.forward()
        self.account.update_margin(self.backtest_data.states[0]['close'])
        next_state = self.__get_state()

        # 4. update reward basing on next state
        margin = self.account.margins[-1]
        reward = (margin + self.account.trade_fee)*100

        # 5. test if done
        if self.account.terminated or self.backtest_data.terminated:
            done = True
        else:
            done = False

        # 6. currently, info is None
        info = None

        return self._unsqueeze_tensor(next_state), self._unsqueeze_tensor(reward), done, info

    def reset(self):
        self.account.reset()
        self.backtest_data.reset()
        self.steps = 0
        state = self.__get_state()
        return self._unsqueeze_tensor(state)

    def render(self):
        _actions = self.account.actions
        _fund_totals = self.account.fund_totals
        _klines = self.backtest_data.klines

        # plot klines and funds
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
        })

        # plot actions and funds
        action = [[], []]
        fund_totals_step = [[], []]
        action_long = [[], []]
        action_short = [[], []]
        action_neutral = [[], []]
        time = range(partial)
        for i in time:
            if i == 0 or _actions[i-1] != _actions[i]:
                action[0].append(i)
                action[1].append(_klines[i]['close'])
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
            'x_data': [action[0], fund_totals_step[0]],
            'y_data': [action[1], fund_totals_step[1]],
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
