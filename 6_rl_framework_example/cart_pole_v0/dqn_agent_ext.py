from itertools import count
from rl_m19.agent import DQNAgent


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.train_loss = []

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
            self.config.env.render()

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
                self.config.plotter.plot_single_with_mean({
                    'id': 1,
                    'title': 'episode_t',
                    'xlabel': 'iteration',
                    'ylabel': 'lifespan',
                    'x_data': range(len(self.episode_t)),
                    'y_data': self.episode_t,
                    'm': 100
                })
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
            time = range(len(self.train_loss))
            self.config.plotter.plot_multiple({
                'id': 'loss',
                'title': 'loss',
                'xlabel': 'step',
                'ylabel': ['train_loss'],
                'x_data': [time],
                'y_data': [self.train_loss],
            })
