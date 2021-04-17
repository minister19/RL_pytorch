import math
from datetime import datetime, timedelta
from .history import History
from .frequencydata import FrequencyData
from .helper import Helper


class Frequency:
    def __init__(self, symb, freq):
        self.symb = symb
        self.freq = freq
        self.data = FrequencyData()        # original data
        self.i_data = FrequencyData()      # interpolated data

    def fetch_history(self, is_force=False, days=7):
        if (History.is_history_exist(self.symb, self.freq) and not is_force):
            return
        start_time = datetime.now() + timedelta(days=-1*days)
        end_time = datetime.now()
        History.fetch_history(self.symb, self.freq, start_time, end_time)

    def load_history(self):
        bars = History.load_history(self.symb, self.freq)
        bars_len = len(bars)
        batch_len = math.ceil(bars_len/100)
        print('{0} {1}: '.format(self.symb, self.freq), end='', flush=True)
        for i in range(bars_len):
            self.on_bar(bars[i])
            if i % batch_len == 0:
                print('.', end='', flush=True)
        print('\n')

    def on_bar(self, bar):
        self.save_data(bar)
        self.interpolate_data(bar)

    def save_data(self, bar):
        self.data.data_len += 1
        self.data.open.append(bar['open'])
        self.data.close.append(bar['close'])
        self.data.high.append(bar['high'])
        self.data.low.append(bar['low'])
        self.data.amount.append(bar['amount'])
        self.data.volume.append(bar['volume'])
        self.data.position.append(bar['position'])
        self.data.bob.append(bar['bob'])
        self.data.eob.append(bar['eob'])
        pre_close = self.data.close[-2] if len(self.data.close) >= 2 else bar['close']
        close_gap = bar['close'] - pre_close
        self.data.close_gap.append(close_gap)
        self.data.abs_close_gap.append(abs(close_gap))
        self.data.is_up = bar['close'] > bar['open']
        self.data.is_increase = close_gap > 0
        self.data.on_freq()

    def interpolate_data(self, bar):
        i_window = 10
        slot = 1
        if len(self.data.abs_close_gap) >= i_window:
            avg = Helper.MA(self.data.abs_close_gap, i_window)
            slot = math.floor(self.data.abs_close_gap[-1] / avg)
            slot = max(slot, 1)
        pre_close = self.data.close[-2] if len(self.data.close) >= 2 else bar['close']
        close_gap = (bar['close'] - pre_close)/slot
        for i in range(slot):
            i_bob = bar['bob'] + (bar['eob'] - bar['bob'])/slot*i
            i_eob = bar['bob'] + (bar['eob'] - bar['bob'])/slot*(i+1)
            i_open = pre_close + close_gap*(i)
            i_close = pre_close + close_gap*(i+1)
            i_amount = bar['amount']/slot
            i_volume = bar['volume']/slot
            self.i_data.data_len += 1
            self.i_data.open.append(i_open)
            self.i_data.close.append(i_close)
            self.i_data.high.append(bar['high'])
            self.i_data.low.append(bar['low'])
            self.i_data.amount.append(i_amount)
            self.i_data.volume.append(i_volume)
            self.i_data.position.append(bar['position'])
            self.i_data.bob.append(i_bob)
            self.i_data.eob.append(i_eob)
            self.i_data.close_gap.append(close_gap)
            self.i_data.abs_close_gap.append(abs(close_gap))
            self.i_data.is_up = i_close > i_open
            self.i_data.is_increase = close_gap > 0
            self.i_data.on_freq()
