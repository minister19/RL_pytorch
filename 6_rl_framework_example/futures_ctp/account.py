class Action:
    def __init__(self, posi: str, vol: float) -> None:
        self.posi = posi
        self.vol = vol


ActionTable = {
    0: Action('L', 0.5),
    1: Action('L', 1.0),
    2: Action('S', 0.5),
    3: Action('S', 1.0),
    4: Action('N', 0),
}


class Account:
    def __init__(self, init_fund):
        self.init_fund = init_fund
        self.fund_total = init_fund
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
        if margin < -0.05:
            return -2
        elif -0.05 <= margin < -0.02:
            return -1
        elif -0.02 <= margin <= 0.02:
            return 0
        elif 0.02 < margin <= 0.05:
            return 1
        else:  # margin > 0.05
            return 2

    def reset(self):
        self.fund_total = self.init_fund
        self.price = None
        self.posi = 'N'
        self.vol = 0
        self.cost_average = None
        self.actions = []
        self.margins = []

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
        self.margins.append(margin)

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
