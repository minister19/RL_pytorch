import asyncio
import json
import numpy as np
import websockets


class SingleIndicator:
    def __init__(self) -> None:
        self.value = []
        self.feedback_sign = None
        self.feedback_cost = None
        self.max_margin = None

    def forward(self, i, close, new_cost):
        s = np.sign(self.value[i])
        if self.feedback_sign:
            if s == 0 or s == self.feedback_sign:
                margin = round((close - self.feedback_cost) * self.feedback_sign * 100 / self.feedback_cost, 3)
                withdraw = self.max_margin - margin
                self.max_margin = max(self.max_margin, margin)
            else:
                self.feedback_sign = s
                self.feedback_cost = new_cost
                self.max_margin = 0
                margin = 0
                withdraw = 0
        else:
            if s != 0:
                self.feedback_sign = s
                self.feedback_cost = new_cost
            self.max_margin = 0
            margin = 0
            withdraw = 0
        return [margin, withdraw]


class BacktestData:
    FREQ = '60min'
    COUNT = 1060
    SKIPPED = 60
    TRAINED = COUNT - SKIPPED

    def __init__(self) -> None:
        self._klines = []
        self.i = BacktestData.SKIPPED  # ema_3 requires to skip 5*6*2=60 klines
        self.emas_trend = SingleIndicator()
        self.emas_support = SingleIndicator()
        self.qianlon_sign = SingleIndicator()
        self.qianlon_trend = SingleIndicator()
        self.qianlon_vel_sign = SingleIndicator()
        self.boll_sig = SingleIndicator()
        self.period_sig = SingleIndicator()
        self.rsi_sig = SingleIndicator()
        self.withdraw = SingleIndicator()
        self.states = []

    async def sync(self):
        '''
        Use Websocket client to sync data from huobi_futures_python server.
        '''
        async with websockets.connect('ws://localhost:6801', max_size=2**30) as websocket:
            await websocket.send('kline_'+BacktestData.FREQ+'_'+str(BacktestData.COUNT)+'_history')
            msg1 = await websocket.recv()
            data1 = json.loads(msg1)
            klines = data1['data']

            for i in range(len(klines)):
                klines[i]['id'] -= 8*60*60
            self._klines = klines

            await websocket.send('indic_'+BacktestData.FREQ+'_'+str(BacktestData.COUNT)+'_history')
            msg2 = await websocket.recv()
            data2 = json.loads(msg2)
            ema_3 = data2['data']['emas']['ema_3']
            ema_3_support = data2['data']['emas']['ema_3_support']
            lon = data2['data']['qianlon']['lon']
            lon_vel = data2['data']['qianlon']['lon_vel']
            boll_sig = data2['data']['boll_aux']['sig']
            period_sig = data2['data']['period']['sig']
            rsi_sig = data2['data']['rsi_smh']['sig']
            withdraw = data2['data']['withdraw']['sig']

            for i in range(len(ema_3)):
                if i == 0 or ema_3[i-1] == ema_3[i]:
                    trend = 0
                elif ema_3[i-1] < ema_3[i]:
                    trend = 1
                else:
                    trend = -1
                self.emas_trend.value.append(trend)

                self.qianlon_sign.value.append(np.sign(lon[i]))
                if i == 0 or lon[i-1] == lon[i]:
                    trend = 0
                elif lon[i-1] < lon[i]:
                    trend = 1
                else:
                    trend = -1
                self.qianlon_trend.value.append(trend)
                self.qianlon_vel_sign.value.append(np.sign(lon_vel[i]))
            self.emas_support.value = ema_3_support
            self.boll_sig.value = boll_sig
            self.period_sig.value = period_sig
            self.rsi_sig.value = rsi_sig
            self.withdraw.value = withdraw

    def reset(self):
        self.i = BacktestData.SKIPPED
        self.forward()

    def forward(self):
        if self.terminated:
            return
        else:
            kline = self._klines[self.i]
            self.states.clear()
            self.states.append(kline)
            for indic in self.indicators:
                val = indic.value[self.i]
                self.states.append(val)
                if indic == self.withdraw:
                    if val == -1:
                        new_cost = kline['low']
                    elif val == 1:
                        new_cost = kline['high']
                    else:
                        new_cost = kline['close']
                else:
                    new_cost = kline['close']
                feedback = indic.forward(self.i, kline['close'], new_cost)
                self.states.extend(feedback)
            self.i += 1

    @property
    def indicators(self):
        return (self.emas_trend,
                self.emas_support,
                self.qianlon_sign,
                self.qianlon_trend,
                self.qianlon_vel_sign,
                self.boll_sig,
                self.period_sig,
                self.rsi_sig,
                self.withdraw
                )

    @property
    def states_dim(self):
        return len(self.indicators)*3

    @property
    def terminated(self):
        return self.i >= len(self._klines)

    @property
    def klines(self):
        return self._klines[BacktestData.SKIPPED:]


if __name__ == '__main__':
    from datetime import datetime
    bd = BacktestData()
    asyncio.run(bd.sync())
    for i in range(len(bd._klines)):
        bd.forward()
        x = bd.states[0]
        print(f'{datetime.fromtimestamp(x["id"], tz=None)}, {x["close"]}', end='\t')

        x = bd.states[1:]
        print(f'e_trend {x[0]}, {x[1]:.3f}', end='\t')
        print(f'e_support {x[2]}, {x[3]:.3f}', end='\t')
        print(f'q_sign {x[4]}, {x[5]:.3f}', end='\t')
        print(f'q_trend {x[6]}, {x[7]:.3f}', end='\t')
        print(f'q_vel_sign {x[8]}, {x[9]:.3f}', end='\t')
        print(f'boll_sig {x[10]}, {x[11]:.3f}', end='\t')
        print(f'period_sig {x[12]}, {x[13]:.3f}', end='\t')
        print(f'rsi_sig {x[14]}, {x[15]:.3f}', end='\t')
        print(f'withdraw {x[16]}, {x[17]:.3f}')
