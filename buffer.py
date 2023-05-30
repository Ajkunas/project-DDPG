import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'trunc'))

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque()
        self.max_size = max_size

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):

        if (len(self) >= self.max_size):
            self.buffer.popleft() 

        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)