from ..helper import Helper


class RSV:
    def __init__(self, freq_data, p_n=14):
        self.freq_data = freq_data
        self.p_n = p_n
        self.rsv = []

    def on_freq(self):
        self.__rsv()

    def __rsv(self):
        d = self.freq_data
        hhv = Helper.HHV(d.high, self.p_n)
        llv = Helper.LLV(d.low, self.p_n)
        if hhv-llv == 0:
            rsv = 0
        else:
            rsv = 100*(d.close[-1]-llv)/(hhv-llv)-50
        self.rsv.append(rsv)
