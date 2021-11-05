import torch
import random
from collections import namedtuple
from itertools import count
from numpy import average
from torch.distributions import Categorical
from rl_m19.network import ReplayMemory
from .base_agent import BaseAgent, AgentUtils

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCriticAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.train_memory = ReplayMemory(config.MC)
        self.train_loss = []
        self.train_losses = []

        self.test_memory = ReplayMemory(config.MC)
        self.test_loss = []
        self.test_losses = []

        self.select_action_counter = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy_net(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.policy_net.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def sample_minibatch(self, memory: ReplayMemory):

        # q_eval, q_target: torch.tensor

    def gradient_descent(self, q_eval, q_target):

    def episode_learn(self, i_episode, step_render=True):
        state = self.config.env.reset()

        for t in count():
            if step_render:
                self.config.env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.env.step(action.item())

            # store transition
            self.train_memory.push(state, action, reward, next_state)

            if len(self.train_memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch(self.train_memory)

                # gradient descent
                loss = self.gradient_descent(q_eval, q_target)
                self.train_loss.append(loss)

            if done or t >= self.config.episode_lifespan:
                self.episode_t.append(t)
                avg_loss = average(self.train_loss) if len(self.train_loss) > 0 else None
                self.train_losses.append(avg_loss)
                self.train_loss.clear()
                break
            else:
                # update state
                state = next_state

    def episode_test(self, i_episode, step_render=True):
        state = self.config.test_env.reset()

        for t in count():
            if step_render:
                self.config.test_env.render()

            # choose action
            action = self.select_action(state)

            # take action and observe
            next_state, reward, done, info = self.config.test_env.step(action.item())

            # store transition
            self.test_memory.push(state, action, reward, next_state)

            if len(self.test_memory) >= self.config.BATCH_SIZE:
                # sample minibatch
                q_eval, q_target = self.sample_minibatch(self.test_memory)

                # compute loss
                loss_t = self.loss_fn(q_eval, q_target)
                loss = loss_t.item()
                self.test_loss.append(loss)

            if done or t >= self.config.episode_lifespan:
                self.test_memory.clear()
                avg_loss = average(self.test_loss) if len(self.test_loss) > 0 else None
                self.test_losses.append(avg_loss)
                self.test_loss.clear()
                break
            else:
                # update state
                state = next_state

    def on_episode_done(self):
        self.config.plotter.plot_single_with_mean({
            'id': 1,
            'title': 'episode_t',
            'xlabel': 'iteration',
            'ylabel': 'lifespan',
            'x_data': range(len(self.episode_t)),
            'y_data': self.episode_t,
            'm': 100
        })

        self.config.plotter.plot_multiple({
            'id': 'loss',
            'title': 'loss',
            'xlabel': 'step',
            'ylabel': ['train_loss', 'test_loss'],
            'x_data': [range(len(self.train_losses)), range(len(self.test_losses))],
            'y_data': [self.train_losses, self.test_losses],
            'color': ['blue', 'red'],
        })

    def episodes_learn(self):
        for i_episode in range(self.config.episodes):
            self.episode_learn(i_episode)
            if i_episode % self.config.TUF == 0:
                # update memory
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.episode_test(i_episode)

            self.on_episode_done()
