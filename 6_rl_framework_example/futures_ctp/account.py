import logging


class Action:
    def __init__(self, posi: str, vol: float) -> None:
        self.posi = posi
        self.vol = vol


ActionTable = [
    # Action('L', 0.5),
    Action('L', 1.0),
    # Action('S', 0.5),
    Action('S', 1.0),
    # Action('N', None),
    # Action('U', None),
]


class Account:
    TRADE_FEE = 0.001
    ACTION_PENALTY_RATIO = 10

    def __init__(self):
        self.states_dim = 2
        self.actions_dim = len(ActionTable)
        self.reset()

    def reset(self):
        self.trade_fee = 0
        self.action_transits = True
        self.fund_total = 1.0
        self.posi = 'N'
        self.vol = 0  # vol is modeled as percentage rather than real vol
        self.avg_cost = None
        self.pre_cost = None
        self.actions = []
        self.actions_real = []
        self.margins = []
        self.fund_totals = []

    def take_action(self, _action: int, _price: float):
        '''
        action_penalty algorithm
        accumulated_reward algorithm
        '''
        self.trade_fee = 0
        _action_real = _action
        action = ActionTable[_action]
        if action.posi == 'N':  # 全平
            self.__close(self.vol)
        elif action.posi == 'U':  # 持仓观望
            if len(self.actions) >= 1:
                _action_real = self.actions_real[-1]
        elif action.posi == self.posi:  # 增减仓
            if action.vol > self.vol:  # 增仓
                self.__open(action.posi, action.vol - self.vol, _price)
            elif action.vol < self.vol:  # 减仓
                self.__close(self.vol - action.vol)
        elif action.posi != self.posi:  # 反手
            self.__close(self.vol)
            self.__open(action.posi, action.vol, _price)
        else:
            raise RuntimeError("Invalid action", str(action))
        self.actions.append(_action)
        self.actions_real.append(_action_real)

        if action.posi in ['N', 'U']:
            self.action_transits = False
        else:
            self.action_transits = True

    def update_margin(self, price: float):
        if self.pre_cost:
            if self.posi == 'L':
                margin = (price - self.pre_cost) / self.avg_cost * self.vol
            elif self.posi == 'S':
                margin = (self.pre_cost - price) / self.avg_cost * self.vol
            else:
                margin = 0
        else:
            margin = 0
        _margin = round(margin, 5)
        self.margins.append(_margin)
        self.fund_total += (_margin - self.trade_fee)
        self.fund_totals.append(self.fund_total)
        # logging.info(f'{self.pre_cost}\t{price}\t{self.nominal_posi}\t{_margin}')
        self.pre_cost = price

    def __open(self, posi, d_vol, price: float):
        if self.posi != 'N' and self.posi != posi:
            raise RuntimeError("Invalid posi", posi)
        self.posi = posi
        self.vol += d_vol
        if self.avg_cost:
            self.avg_cost = (self.avg_cost*(self.vol-d_vol) + price*d_vol) / self.vol
        else:
            self.avg_cost = price
        self.trade_fee += Account.TRADE_FEE*d_vol

    def __close(self, d_vol):
        self.vol -= d_vol
        if self.vol == 0.0:
            self.posi = 'N'
            self.avg_cost = None
        self.trade_fee += Account.TRADE_FEE*d_vol

    @property
    def action_penalty(self):
        return self.trade_fee * Account.ACTION_PENALTY_RATIO

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

    for idx, val in enumerate(ActionTable):
        ac.take_action(idx, x)
        print(f'posi: {ac.nominal_posi}, close: {x}, av_cost: {ac.avg_cost}', end=' ')
        x = random.randint(100, 200)
        ac.update_margin(x)
        print(f'close: {x}, fund: {round(ac.fund_total, 2)}')
