from ..helper import Helper


class Qianlon:
    def __init__(self, freq_data, p_diff=10, p_dea=20, p_ma=5, p_wr_window=20):
        self.freq_data = freq_data
        self.p_diff = p_diff
        self.p_diff_ = [1./self.p_diff, 1.-1./self.p_diff]
        self.p_dea = p_dea
        self.p_dea_ = [1./self.p_dea, 1.-1./self.p_dea]
        self.p_ma = p_ma
        self.p_ma_ = [1./self.p_ma, 1.-1./self.p_ma]
        self.p_wr_window = p_wr_window
        self.__rc = []
        self.__rc_sum = 0
        self.__diff = []
        self.__dea = []

        self.lon = []
        self.lon_ma = []
        self.lon_sma = []
        self.lon_vel = []
        self.lon_wr = []

        self.region = []
        self.isup = []
        self.ma_region = []
        self.ma_isup = []
        self.wr_tran = []

    def on_freq(self):
        self.__qianlon()
        self.__lon_vel()
        self.__lon_wr()
        self.__lon_region()

    def __qianlon(self):
        if self.freq_data.data_len < 2:
            vid = 0
            rc = 0
        else:
            volume_sum = Helper.SUM(self.freq_data.volume, 2)
            price_gap = Helper.HHV(self.freq_data.high, 2) - Helper.LLV(self.freq_data.low, 2)
            vid = 0 if price_gap == 0 else volume_sum/(price_gap*100)
            rc = self.freq_data.close_gap[-1] * vid
        self.__rc.append(rc)
        self.__rc_sum += rc

        if len(self.__diff) < 1:
            diff = self.__rc_sum * self.p_diff_[0]
            dea = self.__rc_sum * self.p_dea_[0]
        else:
            diff = self.__rc_sum * self.p_diff_[0] + self.__diff[-1] * self.p_diff_[1]
            dea = self.__rc_sum * self.p_dea_[0] + self.__dea[-1] * self.p_dea_[1]
        self.__diff.append(diff)
        self.__dea.append(dea)

        lon = diff - dea
        self.lon.append(lon)

    def __lon_vel(self):
        lon = self.lon[-1]
        lon_ma = Helper.MA(self.lon, self.p_ma)
        self.lon_ma.append(lon_ma)

        if len(self.lon_sma) < 1:
            lon_sma = lon * self.p_ma_[0]
        else:
            lon_sma = lon * self.p_ma_[0] + self.lon_sma[-1] * self.p_ma_[1]
        self.lon_sma.append(lon_sma)

        lon_vel = lon - lon_sma
        self.lon_vel.append(lon_vel)

    def __lon_wr(self):
        lon = self.lon[-1]
        hhv = Helper.HHV(self.lon, self.p_wr_window)
        llv = Helper.LLV(self.lon, self.p_wr_window)
        if hhv-llv == 0:
            lon_wr = 0
        else:
            lon_wr = 100*(lon-llv)/(hhv-llv)-50
        self.lon_wr.append(lon_wr)

    def __lon_region(self):
        if Helper.Region(self.lon):
            self.region.append(1)
        else:
            self.region.append(-1)

        if len(self.lon) < 2:
            self.isup.append(0)
        elif Helper.IsUp(self.lon):
            self.isup.append(1)
        else:
            self.isup.append(-1)

        if Helper.Region(self.lon_ma):
            self.ma_region.append(1)
        else:
            self.ma_region.append(-1)

        if len(self.lon_ma) < 2:
            self.ma_isup.append(0)
        elif Helper.IsUp(self.lon_ma):
            self.ma_isup.append(1)
        else:
            self.ma_isup.append(-1)

        if len(self.lon_wr) < 2:
            self.wr_tran.append(0)
        elif (Helper.UpCross(self.lon_wr, 0.5) or
              Helper.UpCross(self.lon_wr, 49.5)):
            self.wr_tran.append(1)
        elif (Helper.DownCross(self.lon_wr, 0.5) or
              Helper.DownCross(self.lon_wr, 49.5)):
            self.wr_tran.append(-1)
        else:
            self.wr_tran.append(self.wr_tran[-1])
