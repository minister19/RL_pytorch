class Action:
    def __init__(self, a: str, p: str, v: float) -> None:
        self.a = a
        self.p = p
        self.v = v


ActionTable = {
    0: Action('O', 'L', 0.5),
    1: Action('O', 'L', 1.0),
    2: Action('O', 'S', 0.5),
    3: Action('O', 'S', 1.0),

    4: Action('C', 'L', 0.5),
    5: Action('C', 'L', 1.0),
    6: Action('C', 'S', 0.5),
    7: Action('C', 'S', 1.0),

    8: Action('S'),
}
