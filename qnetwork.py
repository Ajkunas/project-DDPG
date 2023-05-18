import random
import torch
import torch.nn as nn
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
 

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
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, states, action):
        
        x = torch.cat([states, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
def update(batch, critic, criterion, agent, optimizer, gamma):
    # Get tensors from the batch
    state_batch = torch.FloatTensor(batch.state)
    action_batch = torch.FloatTensor(batch.action)
    done_batch = torch.FloatTensor(batch.done)
    reward_batch = torch.FloatTensor(batch.reward)

    next_state_batch = batch.next_state

    next_action_batch = []

    for next_state in next_state_batch:
        next_state = next_state.tolist()
        next_action = agent.compute_action(next_state)
        next_action_batch.append(next_action.tolist())

    next_state_batch = torch.FloatTensor(batch.next_state)
    next_action_batch = torch.FloatTensor(next_action_batch)
    
    reward_batch = reward_batch.unsqueeze(1)
    done_batch = done_batch.unsqueeze(1)

    q_val = critic.forward(state_batch, action_batch)
    q_next = critic.forward(next_state_batch, next_action_batch)

    with torch.no_grad():
        targets = reward_batch + (1.0 - done_batch) * gamma * q_next
    
    critic_loss = criterion(q_val, targets)

    # critic update
    optimizer.zero_grad()
    critic_loss.backward() 
    optimizer.step()

    return critic_loss.item()
    