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
    5: Action('U', None),
}


class Account:
    def __init__(self):
        self.fund_total = 1.0
        self.posi = 'N'
        self.vol = 0  # vol is modeled as percentage rather than real vol
        self.cost_average = None
        self.actions = []
        self.margins = []

    def reset(self):
        self.fund_total = 1.0
        self.posi = 'N'
        self.vol = 0
        self.cost_average = None
        self.actions.clear()
        self.margins.clear()

    def take_action(self, action: int, price: float):
        action = ActionTable[action]
        if action.posi == 'U':  # 待机
            pass
        elif action.posi == 'N':  # 全平
            self.__close(1.0)
        elif action.posi == self.posi:  # 增减仓
            if action.vol > self.vol:  # 增仓
                self.__open(action.posi, action.vol - self.vol, price)
            elif action.vol < self.vol:  # 减仓
                self.__close(self.vol - action.vol)
        elif action.posi != self.posi:  # 反手
            self.__close(1.0)
            self.__open(action.posi, action.vol, price)
        else:
            raise RuntimeError("Invalid action", str(action))
        self.actions.append(action)

    def update_margin(self, price: float):
        if self.posi == 'L':
            margin = (price - self.cost_average) / self.cost_average * self.vol
        elif self.posi == 'S':
            margin = (self.cost_average - price) / self.cost_average * self.vol
        else:
            margin = 0
        self.margins.append(margin)

        self.fund_total += margin

    def __open(self, posi, d_vol, price: float):
        if self.posi != 'N' and self.posi != posi:
            raise RuntimeError("Invalid posi", posi)
        if self.cost_average:
            self.cost_average = (self.cost_average*self.vol + price*d_vol)/(self.vol + d_vol)
        else:
            self.cost_average = price
        self.posi = posi
        self.vol += d_vol
        return

    def __close(self, d_vol):
        if d_vol == 1.0:
            self.posi = 'N'
            self.vol = 0
        else:
            self.vol -= d_vol

    def plot(self):
        # plot time, action, fund
        pass

    @property
    def fund_fixed(self):
        return self.fund_total * self.vol

    @property
    def fund_liquid(self):
        return self.fund_total - self.fund_fixed

    @property
    def nominal_posi(self):
        if self.posi == 'L':
            return 1 * self.vol
        elif self.posi == 'S':
            return -1 * self.vol
        else:
            return 0

    @property
    def nominal_margin(self):
        if len(self.margins) <= 0:
            return 0
        margin = self.margins[-1]
        if margin < -0.05:
            return -3
        elif -0.05 <= margin < -0.02:
            return -2
        elif -0.02 <= margin < 0:
            return -1
        elif margin == 0:
            return 0
        elif 0 <= margin <= 0.02:
            return 1
        elif 0.02 < margin <= 0.05:
            return 2
        else:  # margin > 0.05
            return 3

    @property
    def state(self):
        return [self.fund_total, self.nominal_posi, self.nominal_margin]

    @property
    def terminated(self):
        return self.fund_total <= 0.5


if __name__ == '__main__':
    ac = Account()

    from numpy import random
    x = random.randint(100, 200)
    ac.take_action(0, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(5, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(1, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(5, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(2, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(5, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(3, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(5, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(4, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))

    x = random.randint(100, 200)
    ac.take_action(5, x)
    print(ac.nominal_posi, x, ac.cost_average, end=' ')
    x = random.randint(100, 200)
    ac.update_margin(x)
    print(x, round(ac.fund_total, 2))
