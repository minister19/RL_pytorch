from .action import ActionTable
from .history import History


class Account:
    def __init__(self, init_fund):
        self.init_fund = init_fund
        self.fund_total = init_fund
        self.history = History()
        self.price = None
        self.posi = 'N'
        # vol is modeled as percentage rather than real vol
        self.vol = 0
        self.cost_average = None
        self.actions = []
        self.margins = []
        self.margin_vels = []

    @property
    def fund_fixed(self):
        return self.cost_average * self.vol

    @property
    def fund_liquid(self):
        return self.fund_total - self.fund_fixed

    @property
    def nominal_posi(self):
        sign = -1 if self.posi == 'S' else 1
        return sign * self.vol

    @property
    def nominal_margin(self):
        margin = self.margins[-1]
        if margin < -0.02:
            return -2
        elif -0.02 <= margin < -0.01:
            return -1
        elif -0.01 <= margin <= 0.01:
            return 0
        elif 0.01 < margin <= 0.02:
            return 1
        else:  # margin > 0.02
            return 2

    @property
    def nominal_margin_vel(self):
        margin_vel = self.margin_vels[-1]
        if margin_vel < -0.01:
            return -2
        elif -0.01 <= margin_vel < -0.005:
            return -1
        elif -0.005 <= margin_vel <= 0.005:
            return 0
        elif 0.005 < margin_vel <= 0.01:
            return 1
        else:  # margin_vel > 0.01
            return 2

    def reset(self):
        self.fund_total = self.init_fund
        self.price = None
        self.posi = 'N'
        self.vol = 0
        self.cost_average = None
        self.actions = []
        self.margins = []
        self.margin_vels = []

    def take_action(self, action: int):
        act = ActionTable[action]
        if act.posi == 'N':  # 全平
            self.__close(1.0)
        elif act.posi == self.posi:  # 增减仓
            if act.vol > self.vol:  # 增仓
                self.__open(act.posi, act.vol - self.vol)
            elif act.vol < self.vol:  # 减仓
                self.__close(self.vol - act.vol)
            else:
                pass
        elif act.posi != self.posi:  # 反手
            self.__close(1.0)
            self.__open(act.posi, act.vol)
        else:
            raise RuntimeError("Invalid action", str(act))
        self.actions.append(action)

    def update_margin(self):
        if self.posi == 'L':
            margin = (self.price - self.cost_average) / self.cost_average * self.vol
        elif self.posi == 'S':
            margin = (self.cost_average - self.price) / self.cost_average * self.vol
        else:
            margin = 0
        if len(self.margins) <= 1:
            margin_vel = 0
        else:
            margin_vel = margin - self.margins[-1]
        self.margins.append(margin)
        self.margin_vels.append(margin_vel)

        self.fund_total += margin

    def __open(self, posi, vol_ratio):
        if self.posi != 'N' or self.posi != posi:
            raise RuntimeError("Invalid posi", posi)
        vol = 1.0 * vol_ratio
        self.cost_average = (self.cost_average*self.vol + self.price*vol)/(self.vol + vol)
        self.vol += vol

    def __close(self, vol_ratio):
        vol = self.vol * vol_ratio
        if self.vol > vol:
            self.vol -= vol
            if self.vol == 0:
                self.posi = 'N'
        else:
            raise RuntimeError("Not enough vol to close", self.vol, vol_ratio)

    def plog(self):
        # plot time, action, fund
        pass
