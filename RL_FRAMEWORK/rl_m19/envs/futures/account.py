import math


class Account:
    def __init__(self, init_fund, board_lot, margin_level, trade_slip):
        self.init_fund = init_fund
        self.fund = init_fund
        self.f_liquid = init_fund
        self.f_fixed = 0  # f_fixed = vol * cost
        self.pos = 0
        self.vol = 0
        self.cost = None
        self.current_price = None
        self.reward = []
        self.board_lot = board_lot
        self.margin_level = margin_level
        self.trade_ratio = board_lot * margin_level
        self.trade_slip = trade_slip  # 最小变动单位*1

    def reset(self, current_price):
        self.fund = self.init_fund
        self.f_liquid = self.init_fund
        self.f_fixed = 0
        self.pos = 0
        self.vol = 0
        self.current_price = current_price
        self.reward = []

    def take_action_v1(self, action):
        vol_max = self.fund / (self.current_price * self.trade_ratio)
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

    def take_action_v2(self, action, d):
        vol_max = self.fund / (self.current_price * self.trade_ratio)
        if action == 0:  # 休眠行情
            self.__update_position(0, 0)
        elif action == 1:  # 振荡行情
            if (d.bbands.location[-1] <= -4):
                self.__update_position(1, vol_max / 2)
            elif (d.bbands.location[-1] >= 4):
                self.__update_position(-1, vol_max / 2)
        elif action == 2:  # 单边行情
            if (d.qianlon.region[-1] == 1 and d.bbands.location[-1] <= 2):
                self.__update_position(1, vol_max)
            elif (d.qianlon.region[-1] == -1 and d.bbands.location[-1] >= -2):
                self.__update_position(-1, vol_max)
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
        trade_slip = (1 if pos == 1 else -1) * self.trade_slip
        open_cost = self.current_price + trade_slip
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
        self.f_liquid -= f_open
        self.f_fixed += f_open
        self.fund = self.f_liquid + self.f_fixed

    def __close(self, pos, vol):
        # 平仓价格
        trade_slip = (1 if pos == 1 else -1) * self.trade_slip
        close_cost = self.current_price - trade_slip
        if self.vol < vol:
            raise RuntimeError("Not enough vol to close")
        else:
            new_vol = self.vol - vol
        self.vol = new_vol

        # 更新资金
        f_close = vol * self.cost * self.trade_ratio
        f_close_gain = self.__get_gain(self.cost) * vol
        self.f_liquid += f_close + f_close_gain
        self.f_fixed -= f_close
        self.fund = self.f_liquid + self.f_fixed

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
        reward = self.fund + gain - self.init_fund
        self.reward.append(reward)
        return reward
