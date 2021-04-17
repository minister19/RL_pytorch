from ..helper import Helper


class RSI:
    def __init__(self, freq_data, p_n=7, p_ma=5, p_smh_ma=5, p_wr=14):
        self.freq_data = freq_data
        self.p_n = p_n
        self.p_n_ = [1./self.p_n, 1.-1./self.p_n]
        self.p_ma = p_ma
        self.p_smh_ma = p_smh_ma
        self.p_wr = p_wr

        self.__p1 = []
        self.__p2 = []
        self.rsi = []
        self.rsi_ma = []

        self.__p3 = []
        self.__p4 = []
        self.rsi_smh = []
        # TODO: region
        self.rsi_smh_region = []
        self.rsi_smh_ma = []

        self.rsi_smh_wr = []
        self.rsi_smh_wr_region = []
        self.rsi_smh_wr_ma = []

    def on_freq(self):
        self.__rsi()
        self.__rsi_smh()
        self.__rsi_smh_wr()

    def __rsi_mixin(self, gap, __p1, __p2, __rsi, __rsi_ma):
        if self.freq_data.data_len < 2:
            temp1 = 0
            temp2 = 0
        else:
            temp1 = max(gap, 0)
            temp2 = abs(gap)

        if len(__p1) < 1:
            p1 = temp1 * self.p_n_[0]
            p2 = temp2 * self.p_n_[0]
        else:
            p1 = temp1 * self.p_n_[0] + __p1[-1] * self.p_n_[1]
            p2 = temp2 * self.p_n_[0] + __p2[-1] * self.p_n_[1]
        __p1.append(p1)
        __p2.append(p2)
        rsi = 50 if p2 == 0 else p1/p2
        __rsi.append(rsi)
        rsi_ma = Helper.MA(__rsi, self.p_ma)
        __rsi_ma.append(rsi_ma)

    def __rsi(self):
        close_gap = self.freq_data.close_gap[-1]
        self.__rsi_mixin(close_gap, self.__p1, self.__p2, self.rsi, self.rsi_ma)

    def __rsi_smh(self):
        avg_gap = Helper.MA(self.freq_data.close_gap, self.p_smh_ma)
        self.__rsi_mixin(avg_gap, self.__p3, self.__p4, self.rsi_smh, self.rsi_smh_ma)

    def __rsi_smh_wr(self):
        rsi_smh = self.rsi_smh[-1]
        hhv = Helper.HHV(self.rsi_smh, self.p_wr)
        llv = Helper.LLV(self.rsi_smh, self.p_wr)
        if hhv-llv == 0:
            rsi_smh_wr = 0
        else:
            rsi_smh_wr = 100*(rsi_smh-llv)/(hhv-llv)-50
        self.rsi_smh_wr.append(rsi_smh_wr)

        rsi_smh_wr_ma = Helper.MA(self.rsi_smh_wr, self.p_n)
        self.rsi_smh_wr_ma.append(rsi_smh_wr_ma)
