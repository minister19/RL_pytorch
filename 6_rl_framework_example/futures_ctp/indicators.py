class Indicators:
    def __init__(self, collection: str) -> None:
        self.collection = collection
        self.emas_sign = []
        self.emas_support = []
        self.qianlon_lon = []
        self.qianlon_vel = []
        self.boll_sig = []
        self.period_sig = []
        self.rsi_sig = []
