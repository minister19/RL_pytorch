import math
from .action import ActionTable
from .history import History


class Account:
    def __init__(self, init_fund):
        self.init_fund = init_fund
        self.fund_total = init_fund
        self.pos = 0
        self.vol = 0
        self.cost = None

        self.actions = []
        self.rewards = []

    @property
    def fund_fixed(self):
        return self.vol * self.cost

    @property
    def fund_liquid(self):
        return self.fund_total - self.fund_fixed

    def reset(self):
        self.fund_total = self.init_fund
        self.pos = 0
        self.vol = 0
        self.cost = None
        self.actions = []
        self.rewards = []

    def take_action(self, action):
        action'
        vol_max = self.fund_total / (self.current_price * self.trade_ratio)
        base = 3
        if action == 0:  # 平仓观察
            self.__update_position(0, 0)
        elif action > 0 and action <= base:  # 立即持多仓
            vol_ratio = action / base
            vol = math.ceil(vol_max * vol_ratio)
            self.__update_position(1, vol)
        elif action > base and action <= 2*base:  # 立即持空仓
            vol_ratio = (action-base) / base
            vol = math.ceil(vol_max * vol_ratio)
            self.__update_position(-1, vol)
        else:
            raise RuntimeError("Invalid action")

    def __update_position(self, pos, vol):
        if pos == 0:
            if self.vol > 0:
                self.__close(self.pos, self.vol)
            else:
                pass
        elif self.pos == 0:
            self.__open(pos, vol)
        elif self.pos == pos:
            if self.vol < vol:
                self.__open(pos, vol - self.vol)
            elif self.vol > vol:
                self.__close(pos, self.vol - vol)
        else:
            self.__close(self.pos, self.vol)
            self.__open(pos, vol)
        self.pos = pos

    def __open(self, pos, vol):
        # 开仓价格
        open_cost = self.current_price
        if self.vol == 0:
            new_vol = vol
            new_cost = open_cost
        else:
            amount = self.vol * self.cost + vol * open_cost
            new_vol = self.vol + vol
            new_cost = amount / new_vol
        self.vol = new_vol
        self.cost = new_cost

        # 更新资金
        f_open = vol * open_cost * self.trade_ratio
        self.fund_fixed += f_open

    def __close(self, pos, vol):
        # 平仓价格
        close_cost = self.current_price
        if self.vol < vol:
            raise RuntimeError("Not enough vol to close")
        else:
            new_vol = self.vol - vol
        self.vol = new_vol

        # 更新资金
        f_close = vol * self.cost * self.trade_ratio
        f_close_gain = self.__get_gain(self.cost) * vol
        self.fund_total += f_close_gain
        self.fund_fixed -= f_close

    def __get_gain(self, cost):
        gain = (1 if self.pos == 1 else -1) * (self.current_price - cost) * self.board_lot
        return gain

    def update_reward(self, current_price):
        self.current_price = current_price
        if self.pos == 0:
            # 2020-08-18 Shawn: TODO: analyze small bonus if Wait, because bet too frequently is expensive.
            gain = 0
        else:
            gain = self.__get_gain(self.cost) * self.vol
        reward = self.fund_total + gain - self.init_fund
        self.rewards.append(reward)
        return reward
