class Action:
    def __init__(self, posi: str, vol: float) -> None:
        self.posi = posi
        self.vol = vol


ActionTable = {
    0: Action('L', 0.5),
    1: Action('L', 1.0),
    2: Action('S', 0.5),
    3: Action('S', 1.0),
    4: Action('N', 0),
}
