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

        self.logger = None
        self.plotter = None
        self.device = None
        self.env = None
        self.test_env = None
        self.state_dim = None
        self.action_dim = None

        # CNN
        self.cnn_image_width = 0
        self.cnn_image_height = 0

        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.loss_fn = None

        if isinstance(config_dict, dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
