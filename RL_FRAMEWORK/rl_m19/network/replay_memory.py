import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # *args: torch.tensor.
    def push(self, *args):
        if len(self.memory) < self.capacity:
            # 2020-08-11 Shawn: only valid memory inside
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample2Transitions(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return transitions

    def sample2Batch(self, batch_size):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
