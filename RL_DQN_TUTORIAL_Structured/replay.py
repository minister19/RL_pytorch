# 1. Replay Memory
# For this, we’re going to need two classses:
# - Transition - a named tuple representing a single transition in our environment.
# It essentially maps (state, action) pairs to their (next_state, reward) result,
# with the state being the screen difference image as described later on.
# - ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently.
# It also implements a .sample() method for selecting a random batch of transitions for training.

import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
