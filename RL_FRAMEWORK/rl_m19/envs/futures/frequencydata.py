from datetime import datetime, timedelta
from .indicator.adx import ADX
from .indicator.bbands import BBands
from .indicator.period import Period
from .indicator.qianlon import Qianlon
from .indicator.rsi import RSI
from .indicator.rsv import RSV

'''
参数名      类型    说明
symbol      str	    标的代码
frequency	str	    频率
open	    float	开盘价
close	    float	收盘价
high	    float	最高价
low	        float	最低价
amount	    float	成交额
volume	    float	成交量
position    long	持仓量
bob	        datetime.datetime	bar开始时间
eob	        datetime.datetime	bar结束时间
'''


class FrequencyData:
    def __init__(self):
        self.ignore_len = 26  # bbands' n parameter is largest among.
        self.data_len = 0
        self.open = []
        self.close = []
        self.high = []
        self.low = []
        self.amount = []
        self.volume = []
        self.position = []
        self.bob = []
        self.eob = []

        self.close_gap = []
        self.abs_close_gap = []
        self.is_up = True
        self.is_increase = True

        self.bbands = BBands(self, 26, 2, 5, 26)
        self.__adx_14 = ADX(self, 14, 6)
        self.__rsi_7 = RSI(self, 7, 5, 5, 14)
        self.__rsv_14 = RSV(self, 14)
        self.period = Period(self, 65)
        self.period.adx_14 = self.__adx_14
        self.period.rsi_7 = self.__rsi_7
        self.period.rsv_14 = self.__rsv_14
        self.qianlon = Qianlon(self, 10, 20, 5, 20)
        self.rsi = RSI(self, 50, 5, 5, 100)

    def on_freq(self):
        self.bbands.on_freq()
        self.period.on_freq()
        self.qianlon.on_freq()
        self.rsi.on_freq()
