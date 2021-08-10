import asyncio
import json
import numpy as np
from websockets import client


class SingleIndicator:
    def __init__(self) -> None:
        self.value = []
        self.feedback_sign = None
        self.feedback_cost = None

    def forward(self, i, close, new_cost):
        s = np.sign(self.value[i])
        if self.feedback_sign:
            if s == 0 or s == self.feedback_sign:
                feedback = round((close - self.feedback_cost) / self.feedback_cost, 3)
            else:
                self.feedback_sign = s
                self.feedback_cost = new_cost
                feedback = 0
        else:
            if s != 0:
                self.feedback_sign = s
                self.feedback_cost = new_cost
            feedback = 0
        return feedback


class BacktestData:
    def __init__(self) -> None:
        self.i = 0
        self.klines = []
        self.emas_trend = SingleIndicator()
        self.emas_support = SingleIndicator()
        self.qianlon_sign = SingleIndicator()
        self.qianlon_trend = SingleIndicator()
        self.qianlon_vel_sign = SingleIndicator()
        self.boll_sig = SingleIndicator()
        self.period_sig = SingleIndicator()
        self.rsi_sig = SingleIndicator()
        self.withdraw = SingleIndicator()
        self.state = [0] * (1+len(self.indicators)*2)

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
        self.i = 0
        self.forward()

    def forward(self):
        if self.terminated:
            return
        else:
            kline = self.klines[self.i]
            self.state.clear()
            self.state.append(kline['close'])
            for indic in self.indicators:
                val = indic.value[self.i]
                self.state.append(val)
                if indic == self.withdraw:
                    if val == -1:
                        new_cost = kline['low']
                    elif val == 1:
                        new_cost = kline['high']
                else:
                    new_cost = kline['close']
                self.state.append(indic.forward(self.i, kline['close'], new_cost))
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
    def terminated(self):
        return self.i >= len(self.klines)


if __name__ == '__main__':
    bd = BacktestData()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bd.sync())
    for i in range(len(bd.klines)):
        bd.forward()
        x = bd.state[1:]
        print(f'e_trend {x[0]}, {x[1]:.3f}', end='\t')
        print(f'e_support {x[2]}, {x[3]:.3f}', end='\t')
        print(f'q_sign {x[4]}, {x[5]:.3f}', end='\t')
        print(f'q_trend {x[6]}, {x[7]:.3f}', end='\t')
        print(f'q_vel_sign {x[8]}, {x[9]:.3f}', end='\t')
        print(f'boll_sig {x[10]}, {x[11]:.3f}', end='\t')
        print(f'period_sig {x[12]}, {x[13]:.3f}', end='\t')
        print(f'rsi_sig {x[14]}, {x[15]:.3f}', end='\t')
        print(f'withdraw {x[16]}, {x[17]:.3f}')
