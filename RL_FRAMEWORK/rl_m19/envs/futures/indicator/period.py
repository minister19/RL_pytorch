from ..helper import Helper


class Period:
    def __init__(self, freq_data, p_thre=65):
        self.freq_data = freq_data
        self.p_thre = p_thre
        self.down_thre = p_thre
        self.up_thre = 300 - p_thre
        self.adx_14 = None
        self.rsi_7 = None
        self.rsv_14 = None
        self.trend = []
        self.period = []

    def on_freq(self):
        self.adx_14.on_freq()
        self.rsi_7.on_freq()
        self.rsv_14.on_freq()
        self.__period()

    def __period(self):
        trend = self.adx_14.adx[-1] + self.rsi_7.rsi[-1] + self.rsv_14.rsv[-1]
        self.trend.append(trend)

        if len(self.period) <= 0:
            self.period.append(0)

        elif trend < self.down_thre:
            self.period.append(-1)  # 空
        elif Helper.UpCross(self.trend, self.down_thre):
            self.period.append(-2)  # 空转多

        elif trend > self.up_thre:
            self.period.append(1)  # 多
        elif Helper.DownCross(self.trend, self.up_thre):
            self.period.append(2)  # 多转空

        elif self.period[-1] in [-2, -3]:
            self.period.append(-3)  # 空转多结束
        elif self.period[-1] in [2, 3]:
            self.period.append(3)  # 多转空结束
        else:
            self.period.append(0)
