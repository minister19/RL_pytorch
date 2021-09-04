from itertools import count
from rl_m19.agent import DQNAgent


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
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
