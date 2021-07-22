from websockets import client
from .indicators import Indicators


class History:
    def __init__(self) -> None:
        self.wsclient = None
        self.indic_15min = Indicators('indic_15min')
        self.indic_60min = Indicators('indic_60min')
        self.get_data()

    def get_data(self):
        # connect
        # send indic_15min_sync, indic_60min_sync
        # parse to local data
        # interpolate indic_60min such that length equals indic_15min
        pass
