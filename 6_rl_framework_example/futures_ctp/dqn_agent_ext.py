from itertools import count
from rl_m19.agent import DQNAgent

'''
- Find the relationships:
  - eps decay - memory capacity
  - batch size - memory capacity
  - eps decay - data trained
  - batch size - data trained

- episode train loss (mean) vs test loss
'''


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.train_loss = []
        self.test_loss = []
        self.fund_totals = []

    def episode_learn(self, i_episode):
        state = self.config.env.reset()

        for t in count():
            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            self.memory.push(state, action, reward, next_state)

            if len(self.memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch()

                # gradient descent
                loss = self.gradient_descent(q_eval, q_target)
                self.train_loss.append(loss)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
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
                break
            else:
                # update state
                state = next_state

    def episodes_learn(self):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode)
            if i_episode % self.config.TUF == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # plot train and test loss
            # time = range(len(self.train_loss))
            # self.config.plotter.plot_multiple({
            #     'id': 'loss',
            #     'title': 'loss',
            #     'xlabel': 'step',
            #     'ylabel': ['train_loss'],
            #     'x_data': [time],
            #     'y_data': [self.train_loss],
            # })
