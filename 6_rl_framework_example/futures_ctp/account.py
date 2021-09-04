import logging


class Action:
    def __init__(self, posi: str, vol: float) -> None:
        self.posi = posi
        self.vol = vol


ActionTable = {
    # 0: Action('L', 1.0),
    # 1: Action('S', 1.0),

    0: Action('L', 1.0),
    1: Action('S', 1.0),
    2: Action('N', 0),

    # 0: Action('L', 0.5),
    # 1: Action('L', 1.0),
    # 2: Action('S', 0.5),
    # 3: Action('S', 1.0),
    # 4: Action('N', 0),
}

# TODO:
# 2. Design algorithms to control number of actions to a smaller range.
# 2. How about for every x steps, only y actions can be taken, if exceeds, done this episode.
# 2. If fixed number of actions are used up, the last action's state is calculated by last kline rather than next kline.
# 3. More accurate trade fee on both open and close action.
# 4. Margin basing on avg_cost rather than margin_base, otherwise model would not work for general purpose.
# 5. Tune trade fee and control how many actions taken.
# 6. Be aware if training more than enough times, model is overfitting (loss drops, rise and drop again).
# 7. More Klines to train.
# 8. Consider Action 'N'.
# 9. Consider norminal margin, stepped ones, rather than decimal ones.


class Account:
    def __init__(self):
        self.states_dim = 2
        self.actions_dim = 2
        self.fund_total = 1.0
        self.posi = 'N'
        self.vol = 0  # vol is modeled as percentage rather than real vol
        self.pre_cost = None
        self.actions = []
        self.margins = []
        self.fund_totals = []

    def reset(self):
        self.fund_total = 1.0
        self.posi = 'N'
        self.vol = 0
        self.pre_cost = None
        self.actions.clear()
        self.margins.clear()
        self.fund_totals.clear()

    def take_action(self, action: int, price: float):
        if len(self.actions) <= 0:
            self.trade_fee = 0
        elif self.actions[-1] != action:
            self.trade_fee = -0.01
        else:
            self.trade_fee = 0.00001

        _action = ActionTable[action]
        if _action.posi == 'N':  # 全平
            self.__close(1.0)
        elif _action.posi == self.posi:  # 增减仓
            if _action.vol > self.vol:  # 增仓
                self.__open(_action.posi, _action.vol - self.vol, price)
            elif _action.vol < self.vol:  # 减仓
                self.__close(self.vol - _action.vol)
        elif _action.posi != self.posi:  # 反手
            self.__close(1.0)
            self.__open(_action.posi, _action.vol, price)
        else:
            raise RuntimeError("Invalid action", str(_action))
        self.actions.append(action)

    def update_margin(self, price: float):
        margin_base = 50000
        if self.pre_cost:
            if self.posi == 'L':
                margin = (price - self.pre_cost) / margin_base * self.vol
            elif self.posi == 'S':
                margin = (self.pre_cost - price) / margin_base * self.vol
            else:
                margin = 0
        else:
            margin = 0
        _margin = round(margin, 5)
        logging.info(f'{self.pre_cost}\t{price}\t{self.nominal_posi}\t{_margin}')
        self.margins.append(_margin)
        self.fund_total += _margin
        self.fund_totals.append(self.fund_total)
        self.pre_cost = price

    def __open(self, posi, d_vol, price: float):
        if self.posi != 'N' and self.posi != posi:
            raise RuntimeError("Invalid posi", posi)
        self.posi = posi
        self.vol += d_vol
        return

    def __close(self, d_vol):
        if d_vol == 1.0:
            self.posi = 'N'
            self.vol = 0
        else:
            self.vol -= d_vol

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
        return self.margins[-1]

    @property
    def states(self):
        return [self.fund_total, self.nominal_posi, self.nominal_margin]

    @property
    def terminated(self):
        return self.fund_total <= 0.5


if __name__ == '__main__':
    ac = Account()

    from numpy import random
    x = random.randint(100, 200)

    for i in range(5):
        ac.take_action(i, x)
        print(f'posi: {ac.nominal_posi}, close: {x}, av_cost: {ac.av_cost}', end=' ')
        x = random.randint(100, 200)
        ac.update_margin(x)
        print(f'close: {x}, fund: {round(ac.fund_total, 2)}')

        ac.take_action(5, x)
        print(f'posi: {ac.nominal_posi}, close: {x}, av_cost: {ac.av_cost}', end=' ')
        x = random.randint(100, 200)
        ac.update_margin(x)
        print(f'close: {x}, fund: {round(ac.fund_total, 2)}')
