import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from itertools import count
from replay import Transition, ReplayMemory
from config import Config


class Agent:
    def __init__(self, env, env_w, device, config: Config):
        self.env = env
        self.env_w = env_w
        self.device = device
        self.cfg = config
        self.n_actions = config.n_actions
        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        self.steps_done += 1
        sample = random.random()
        eps_threshold = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END) * \
            math.exp(-1. * self.steps_done / self.cfg.EPS_DECAY)
        if sample < eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # action = self.policy_net(state).max(1)[1]
                action = self.policy_net(state).argmax() % self.n_actions
        else:
            action = random.randrange(self.n_actions)
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.cfg.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.cfg.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.cfg.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def step(self, i_episode):
        # Initialize the environment and state
        self.env.reset()
        last_screen = self.env_w.get_screen()
        current_screen = self.env_w.get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = self.select_action(state)
            obs, reward, done, obs_ = self.env.step(action.item())
            # reward = torch.tensor([reward], device=self.device)
            reward = torch.tensor([-abs(obs[2])], device=self.device, dtype=torch.float)

            # Observe new state
            last_screen = current_screen
            current_screen = self.env_w.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            self.optimize_model()
            if done:
                self.episode_durations.append(t + 1)
                self.env_w.plot_durations(self.episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % self.cfg.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
