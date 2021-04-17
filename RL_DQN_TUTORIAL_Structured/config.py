class Config:
    def __init__(self, BATCH_SIZE=128,
                 GAMMA=0.999,
                 EPS_START=0.9,
                 EPS_END=0.05,
                 EPS_DECAY=200,
                 TARGET_UPDATE=10):
        # Hyperparameters and utilities
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_actions = None
        self.policy_net = None
        self.target_net = None
