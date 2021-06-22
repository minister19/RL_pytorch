from websockets import client
from .indicators import Indicators


class History:
    def __init__(self) -> None:
        wsclient = None
        indic_15min = Indicators('indic_15min')
        indic_60min = Indicators('indic_60min')

    def get_data(self):
        # connect
        # send indic_15min_sync, indic_60min_sync
        # parse to local data
        # interpolate indic_60min such that length equals indic_15min
        pass
