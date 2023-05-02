import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, trunc):
        transition = (state, action, reward, next_state, trunc)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            # not sure about the behaviour when buffer is overloaded.
            self.buffer[self.idx] = transition
            self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, truncs = zip(*batch)
        return states, actions, rewards, next_states, truncs
    
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def update(self, optimizer, transitions, gamma):
        states, actions, rewards, next_states, truncs = transitions
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        truncs = torch.FloatTensor(truncs)

        # Compute the TD target
        with torch.no_grad():
            next_actions = policy.select_action(next_states)
            q_next = self.forward(torch.cat([next_states, next_actions], dim=1))
            target = rewards + gamma * q_next * (1 - truncs)

        # Compute the TD error and update the Q network
        q_values = self.forward(torch.cat([states, actions], dim=1))
        loss = F.mse_loss(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()