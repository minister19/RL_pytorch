import pandas as pd
from pandas import ExcelWriter, ExcelFile


class Exceller(object):
    def __init__(self):
        pass

    def export_excel(self, filepath):
        df = pd.DataFrame({
            'eob': self.eob_str,
            'bbands_offset': self.bbands.offset_sma_,
            'bbands_tmp_wr': self.bbands.tmp_wr_,
            'donchian_offset': self.donchian.offset_sma_,
            'lon': self.qianlon.lon,
            'lon_vel': self.qianlon.lon_vel_,
            'lon_wr': self.qianlon.lon_wr_,
            'sma_wr': self.sma_wr.sma_wr_,
            'team_climb': self.eams_climb.cross_,
            'team_cross': self.eams_cross_offset.cross_,
            'team_offset': self.eams_cross_offset.offset_
        })
        writer = ExcelWriter(filepath)
        df.to_excel(writer, 'Sheet1', index=False)
        writer.save()

    def import_excel(self, filepath):
        df = pd.read_excel(filepath, sheet_name='Sheet1')
        print(df)
