class Config:
    def __init__(self, config_dict=None):
        self.episode_lifespan = 10**3
        self.episodes = 1000
        self.BATCH_SIZE = 64
        self.GAMMA = 0.999
        self.EPS_fn = None
        self.LR = 0.001  # LEARNING_RATE
        self.MC = 1000  # MEMORY_CAPACITY
        self.TUF = 10  # TARGET_UPDATE_FREQUENCY

        self.plotter = None
        self.device = None
        self.env = None
        self.states_dim = None
        self.actions_dim = None

        self.memory_fn = None
        self.policy_net_fn = None
        self.target_net_fn = None
        self.optimizer_fn = None
        self.loss_fn = None

        if isinstance(config_dict, dict):
            for key in config_dict.keys():
                setattr(self, key, config_dict[key])
