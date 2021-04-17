from ..helper import Helper


class ADX:
    def __init__(self, freq_data, p_n=14, p_n2=6):
        self.freq_data = freq_data
        self.p_n = p_n
        self.p_n2 = p_n2
        self.tr_data = []
        self.dmp_data = []
        self.dmm_data = []
        self.adx_data = []
        self.adx = []

    def on_freq(self):
        self.__adx()

    def __adx(self):
        d = self.freq_data
        if d.data_len < 2:
            max_gap = 0
        else:
            max_gap = max(d.high[-1]-d.low[-1], abs(d.high[-1]-d.close[-2]), abs(d.low[-1]-d.close[-2]))
        self.tr_data.append(max_gap)
        tr = Helper.SUM(self.tr_data, self.p_n)
        hd = d.high[-1] - Helper.RefN(d.high, 1)
        ld = Helper.RefN(d.low, 1) - d.low[-1]
        dmp_data = hd if hd > 0 and hd > ld else 0
        self.dmp_data.append(dmp_data)
        dmm_data = ld if ld > 0 and ld > hd else 0
        self.dmm_data.append(dmm_data)
        dmp = Helper.SUM(self.dmp_data, self.p_n)
        dmm = Helper.SUM(self.dmm_data, self.p_n)
        if tr == 0:
            pdi = 0
            mdi = 0
        else:
            pdi = dmp*100 / tr
            mdi = dmm*100 / tr
        if (mdi + pdi) == 0:
            adx_data = 0
        else:
            adx_data = abs(mdi - pdi) / (mdi + pdi)*100
        self.adx_data.append(adx_data)
        adx = Helper.MA(self.adx_data, self.p_n2)
        self.adx.append(adx)
