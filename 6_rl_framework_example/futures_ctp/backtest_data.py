from numpy import sign


class SingleIndicator:
    def __init__(self) -> None:
        self.val = []
        self.feedback_await_reset = False
        self.feedback_cost = None

    def forward(self, val, close):
        self.val.append(val)
        return self.__feedback(val, close)

    def __feedback(self, val, close):
        if val == 0:
            if self.feedback_cost:
                feedback = (close - self.feedback_cost) * sign(val)
            else:
                feedback = 0
            self.feedback_await_reset = True
        else:
            if self.feedback_cost:
                feedback = (close - self.feedback_cost) * sign(val)
            else:
                feedback = 0
                self.feedback_cost = close
            if self.feedback_await_reset:
                self.feedback_cost = close
        return feedback


class Indicators:
    def __init__(self, collection: str) -> None:
        self.emas_sign = SingleIndicator()
        self.emas_support = SingleIndicator()
        self.qianlon_sign = SingleIndicator()
        self.qianlon_vel = SingleIndicator()
        self.boll_sig = SingleIndicator()
        self.period_sig = SingleIndicator()
        self.rsi_sig = SingleIndicator()
        self.withdraw = SingleIndicator()


class BacktestData:
    def __init__(self) -> None:
        self.indicators = Indicators()

    def sync(self):
        '''
        Use Websocket client to sync data from huobi_futures_python server.
        '''
        pass
