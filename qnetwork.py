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
    def __init__(self, agent, norm_env):
        super(QNetwork, self).__init__()
        self.agent = agent
        self.norm_env = norm_env
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def update(self, transition, gamma = 0.001):
        state = transition[:, :3]
        action = transition[:, 3]
        # Compute the TD target
        with torch.no_grad():
            targets = []
            for s, a in zip(state, action):
                next_state, reward, terminated, truncated, info = self.norm_env.step(a.numpy()) 
                next_actions = self.agent.compute_action(next_state)
                
                next_state, next_actions  = torch.Tensor(next_state).view(1, -1), torch.Tensor(next_actions).view(1, -1)
                q_next = self.forward(torch.cat([next_state, next_actions], dim=1))
                target = reward + gamma * q_next * (1 - truncated)
                targets.append(target[0])  
            targets = torch.Tensor(targets)
            
        q_values = self.forward(transition)
        
        loss = F.mse_loss(q_values.view(-1), targets)
        return loss
    