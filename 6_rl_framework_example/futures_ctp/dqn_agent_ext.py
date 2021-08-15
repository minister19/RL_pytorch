from itertools import count
from rl_m19.agent import DQNAgent


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)

    # q_eval, q_target: torch.tensor
    def gradient_descent(self, q_eval, q_target):
        loss = self.loss_fn(q_eval, q_target)  # compute loss
        self.optimizer.zero_grad()
        loss.backward()
        # 2020-08-13 Shawn: While, no clamp is better for CartPole
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def episode_learn(self, i_episode):
        state = self.config.env.reset()

        for t in count():
            self.config.env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            # 2020-08-18 Shawn: 仅当 reward 较大时保存 memory.
            self.memory.push(state, action, reward, next_state)

            if len(self.memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch()

                # gradient descent
                self.gradient_descent(q_eval, q_target)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                self.config.plotter.plot_list_ndarray(self.episode_t)
                self.config.env.render()
                break
            else:
                # update state
                state = next_state
