from ..helper import Helper


class BBands:
    def __init__(self, freq_data, p_n=26, p_p=2, p_ma=5, p_wr_window=26):
        self.freq_data = freq_data
        self.p_n = p_n
        self.p_p = p_p
        self.p_ma = p_ma
        self.p_ma_ = [1./p_ma, 1.-1./p_ma]
        self.p_wr_window = p_wr_window
        self.mid = []
        self.tmp = []
        self.top = []
        self.bottom = []
        self.offset = []
        self.offset_sma = []
        self.tmp_wr = []
        self.location = []

    def on_freq(self):
        self.__bbands()
        self.__offset()
        self.__tmp_wr()
        self.__location()

    def __bbands(self):
        mid = Helper.MA(self.freq_data.close, self.p_n)
        self.mid.append(mid)
        tmp = Helper.STD(self.freq_data.close, self.p_n)
        self.tmp.append(tmp)
        self.top.append(mid+self.p_p*tmp)
        self.bottom.append(mid-self.p_p*tmp)

    def __offset(self):
        mid = self.mid[-1]
        tmp = self.tmp[-1]

        ratio_1 = 100*tmp/mid
        peak_1 = self.freq_data.high[-1] if self.freq_data.is_up else self.freq_data.low[-1]
        width = self.p_p*tmp
        if width == 0:
            ratio_2 = 0
        else:
            ratio_2 = 100*(peak_1 - mid) / width
        offset = ratio_1*ratio_2
        self.offset.append(offset)

        if len(self.offset_sma) < 1:
            offset_sma = offset * self.p_ma_[0]
        else:
            offset_sma = offset * self.p_ma_[0] + self.offset_sma[-1] * self.p_ma_[1]
        self.offset_sma.append(offset_sma)

    def __tmp_wr(self):
        tmp = self.tmp[-1]
        hhv = Helper.HHV(self.tmp, self.p_wr_window)
        llv = Helper.LLV(self.tmp, self.p_wr_window)
        if hhv-llv == 0:
            tmp_wr = 0
        else:
            tmp_wr = 100*(tmp-llv)/(hhv-llv)-50
        self.tmp_wr.append(tmp_wr)

    def __location(self):
        gap = self.freq_data.close[-1] - self.mid[-1]
        width = self.p_p*self.tmp[-1]
        if gap > width:
            self.location.append(4)
        elif gap > 2/3*width and gap <= 3/3*width:
            self.location.append(3)
        elif gap > 1/3*width and gap <= 2/3*width:
            self.location.append(2)
        elif gap > 0/3*width and gap <= 1/3*width:
            self.location.append(1)
        elif gap == 0:
            self.location.append(0)
        elif gap > -1/3*width and gap <= -0/3*width:
            self.location.append(-1)
        elif gap > -2/3*width and gap <= -1/3*width:
            self.location.append(-2)
        elif gap > -3/3*width and gap <= -2/3*width:
            self.location.append(-3)
        elif gap <= -3/3*width:
            self.location.append(-4)
