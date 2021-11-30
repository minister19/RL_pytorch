from rl_m19.agent import DQNAgent


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.fund_totals = []

    def on_episode_done(self):
        self.config.plotter.plot_multiple({
            'id': 'loss',
            'title': 'loss',
            'xlabel': 'step',
            'ylabel': ['train_loss', 'test_loss'],
            'x_data': [range(len(self.train_losses)), range(len(self.test_losses))],
            'y_data': [self.train_losses, self.test_losses],
            'color': ['blue', 'red'],
        })

        self.fund_totals.append(self.config.env.account.fund_totals[-1])
        self.config.plotter.plot_single_with_mean({
            'id': 'fund_totals',
            'title': 'fund_totals',
            'xlabel': 'time',
            'ylabel': 'value',
            'x_data': range(len(self.fund_totals)),
            'y_data': self.fund_totals,
            'm': 100
        })

        self.config.env.render()

    def episodes_learn(self):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode, False)
            if i_episode % self.config.target_update_freq == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # self.episode_test(i_episode, False)

            self.on_episode_done()
