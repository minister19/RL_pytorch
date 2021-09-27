from itertools import count
from rl_m19.agent import DQNAgent

'''
- accumulated_reward algorithm
  - only when action transits, assign accumulated reward to each action, and push to memory
  - 随 action 切换添加 memory，而不是随 each action 添加 memory
  
- early_done algorithm
  - if action transits quota exhausted, episode is done
  - if action transits quota exhausted, the last action's state is calculated by its next kline
  - action u 和 early_done 配合训练，action u 给予 small margin

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

    # q_eval, q_target: torch.tensor
    def gradient_descent(self, q_eval, q_target):
        loss = self.loss_fn(q_eval, q_target)  # compute loss
        self.optimizer.zero_grad()
        loss.backward()
        # 2020-08-13 Shawn: While, no clamp is better sometimes.
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

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
            policy_net_bak = self.policy_net.state_dict()

            self.episode_learn(i_episode)
            if len(self.fund_totals) >= 2 and self.fund_totals[-1] < self.fund_totals[-2]:
                self.policy_net.load_state_dict(policy_net_bak)
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
