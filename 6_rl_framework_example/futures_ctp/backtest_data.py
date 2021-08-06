import asyncio
import json
from numpy import sign
from websockets import client


class SingleIndicator:
    def __init__(self) -> None:
        self.value = []
        self.feedback_await_reset = False
        self.feedback_cost = None

    def forward(self, i, close, new_cost):
        val = self.value[i]
        if val == 0:
            if self.feedback_cost:
                feedback = (close - self.feedback_cost) * sign(val)
            else:
                feedback = 0
            self.feedback_await_reset = True
        else:
            if self.feedback_await_reset:
                feedback = 0
                self.feedback_await_reset = False
                self.feedback_cost = new_cost
            elif self.feedback_cost:
                feedback = (close - self.feedback_cost) * sign(val)
            else:
                feedback = 0
                self.feedback_cost = new_cost
        return feedback


class Indicators:
    def __init__(self) -> None:
        self.i = 0
        self.emas_sign = SingleIndicator()
        self.emas_support = SingleIndicator()
        self.qianlon_sign = SingleIndicator()
        self.qianlon_trend = SingleIndicator()
        self.qianlon_vel = SingleIndicator()
        self.boll_sig = SingleIndicator()
        self.period_sig = SingleIndicator()
        self.rsi_sig = SingleIndicator()
        self.withdraw = SingleIndicator()

    def forward(self, kline):
        if self.i >= len(self.emas_sign):
            raise RuntimeError('Indicator data are exhausted.')
        else:
            ret = []
            for indic in self.indicators:
                ret.append(indic.value[self.i])
                if indic == self.withdraw:
                    if indic.value[self.i] == -1:
                        new_cost = kline.low
                    elif indic.value[self.i] == 1:
                        new_cost = kline.high
                else:
                    new_cost = kline.close
                    ret.append(indic.forward(self.i, kline.close, new_cost))
            self.i += 1
            return ret

    @property
    def indicators(self):
        return (self.emas_sign,
                self.emas_support,
                self.qianlon_sign,
                self.qianlon_trend,
                self.qianlon_vel,
                self.boll_sig,
                self.period_sig,
                self.rsi_sig,
                self.withdraw
                )


class BacktestData:
    def __init__(self) -> None:
        self.klines = []
        self.indicators = Indicators()

    async def sync(self):
        '''
        Use Websocket client to sync data from huobi_futures_python server.
        '''
        async with client.connect('ws://localhost:6801') as websocket:
            await websocket.send('kline_15min_sync')
            msg1 = await websocket.recv()
            data1 = json.loads(msg1)
            self.klines = data1['data']

            await websocket.send('indic_15min_sync')
            msg2 = await websocket.recv()
            data2 = json.loads(msg2)
            ema_3 = data2['data']['emas']['ema_3']


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    bd = BacktestData()
    loop.run_until_complete(bd.sync())
