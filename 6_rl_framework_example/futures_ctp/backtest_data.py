import asyncio
import json
import numpy as np
import websockets


class SingleIndicator:
    def __init__(self) -> None:
        self.value = []
        self.feedback_sign = None
        self.feedback_cost = None
        self.max_margin = 0

    def forward(self, i, close, new_cost):
        s = np.sign(self.value[i])
        if self.feedback_sign:
            if s == 0 or s == self.feedback_sign:
                margin = round((close - self.feedback_cost) * 100 / self.feedback_cost, 3)
                withdraw = min(margin - self.max_margin, 0)
                self.max_margin = max(margin, self.max_margin)
                _margin = margin * self.feedback_sign
                _withdraw = withdraw * self.feedback_sign
            else:
                self.feedback_sign = s
                self.feedback_cost = new_cost
                self.max_margin = 0
                _margin = 0
                _withdraw = 0
        else:
            if s != 0:
                self.feedback_sign = s
                self.feedback_cost = new_cost
            _margin = 0
            _withdraw = 0
        return [_margin]
        # return [_margin, _withdraw]


class BacktestData:
    FREQ = '15min'
    COUNT = 1060
    SKIPPED = 60
    TRAINED = COUNT - SKIPPED

    def __init__(self) -> None:
        self.i = BacktestData.SKIPPED  # ema_3 requires to skip 5*6*2=60 klines
        self._states = []
        self._klines = []
        self.emas_trend = SingleIndicator()
        self.emas_support = SingleIndicator()
        self.qianlon_sign = SingleIndicator()
        self.qianlon_vel_sign = SingleIndicator()
        self.boll_sig = SingleIndicator()
        self.period_sig = SingleIndicator()
        self.rsi_sig = SingleIndicator()
        self.rsv_trend = SingleIndicator()
        self.rsv_sig = SingleIndicator()
        self.withdraw_sig = SingleIndicator()

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
            lon = data2['data']['qianlon']['lon']
            lon_vel = data2['data']['qianlon']['lon_vel']
            rsv_m = data2['data']['rsv_aux']['rsv_m']

            for i in range(len(ema_3)):
                if i == 0 or ema_3[i-1] == ema_3[i]:
                    trend = 0
                elif ema_3[i-1] < ema_3[i]:
                    trend = 1
                else:
                    trend = -1
                self.emas_trend.value.append(trend)
                self.qianlon_sign.value.append(np.sign(lon[i]))
                self.qianlon_vel_sign.value.append(np.sign(lon_vel[i]))

                if rsv_m[i] == 50:
                    trend = 0
                elif rsv_m[i] > 50:
                    trend = 1
                else:
                    trend = -1
                self.rsv_trend.value.append(trend)

                if i == 0:
                    sig = 0
                elif rsv_m[i-1] < 5.0 and rsv_m[i] >= 5.0:
                    sig = -1
                elif rsv_m[i-1] > 95.0 and rsv_m[i] <= 95.0:
                    sig = 1
                else:
                    sig = 0
                self.rsv_sig.value.append(sig)

            self.emas_support.value = data2['data']['emas']['ema_3_support']
            self.boll_sig.value = data2['data']['boll_aux']['sig']
            self.period_sig.value = data2['data']['period']['sig']
            self.rsi_sig.value = data2['data']['rsi_smh']['sig']
            self.withdraw_sig.value = data2['data']['withdraw']['sig']
        self.__preprocess()

    def __preprocess(self):
        self._states.clear()
        for i in range(BacktestData.COUNT):
            kline = self._klines[i]
            states = []
            states.append(kline)
            for indic in self.indicators:
                val = indic.value[i]
                states.append(val)
                if indic == self.withdraw_sig:
                    if val == -1:
                        new_cost = kline['low']
                    elif val == 1:
                        new_cost = kline['high']
                    else:
                        new_cost = kline['close']
                else:
                    new_cost = kline['close']
                feedback = indic.forward(i, kline['close'], new_cost)
                states.extend(feedback)
            self._states.append(states)

    def reset(self):
        self.i = BacktestData.SKIPPED

    def forward(self):
        self.i += 1

    @property
    def indicators(self):
        return (
            self.emas_trend,
            self.emas_support,
            self.qianlon_sign,
            self.qianlon_vel_sign,
            self.boll_sig,
            self.period_sig,
            self.rsi_sig,
            self.rsv_trend,
            self.rsv_sig,
            self.withdraw_sig
        )

    @property
    def states_dim(self):
        return len(self.indicators)*2

    @property
    def states(self):
        return self._states[self.i]

    @property
    def terminated(self):
        return self.i == (self.COUNT - 1)

    @property
    def klines(self):
        return self._klines[BacktestData.SKIPPED:]


if __name__ == '__main__':
    from datetime import datetime
    bd = BacktestData()
    asyncio.run(bd.sync())
    for i in range(BacktestData.COUNT):
        x = bd._states[i][0]
        print(f'{datetime.fromtimestamp(x["id"], tz=None)}, {x["close"]}', end='\t')

        x = bd._states[i][1:]
        print(f'e_trend {x[0]}, {x[1]:.2f}, {x[2]:.2f}', end='\t')
        print(f'e_support {x[3]}, {x[4]:.2f}, {x[5]:.2f}', end='\n')

        # print(f'q_sign {x[6]}, {x[7]:.2f}, {x[8]:.2f}', end='\t')
        # print(f'q_vel_sign {x[9]}, {x[10]:.2f}, {x[11]:.2f}', end='\n')

        # print(f'boll_sig {x[12]}, {x[13]:.2f}, {x[14]:.2f}', end='\t')
        # print(f'period_sig {x[15]}, {x[16]:.2f}, {x[17]:.2f}', end='\t')
        # print(f'rsi_sig {x[18]}, {x[19]:.2f}, {x[20]:.2f}', end='\n')

        # print(f'rsv_trend {x[21]}, {x[21]:.2f}, {x[22]:.2f}', end='\t')
        # print(f'rsv_sig {x[24]}, {x[25]:.2f}, {x[26]:.2f}', end='\n')

        # print(f'withdraw {x[27]}, {x[28]:.2f}, {x[29]:.2f}', end='\n')
