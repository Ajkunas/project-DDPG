import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque()
        self.max_size = max_size

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, trunc):
        transition = (state, action, np.array([reward]), next_state, trunc)
        
        if (len(self) >= self.max_size):
            self.buffer.popleft() 
         
        self.buffer.append(transition)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        trunc_batch = []
        
        batch = random.sample(self.buffer, batch_size)
        
        for transition in batch: 
            state, action, reward, next_state, trunc = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            trunc_batch.append(trunc)
            
        return state_batch, action_batch, reward_batch, next_state_batch, trunc_batch
    
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, agent):
        super(QNetwork, self).__init__()
        self.critic_criterion  = nn.MSELoss()
        self.agent = agent
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, states, action):
        #print("states :", states)
        #print("action :", action)
        x = torch.cat([states, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def update(self, optimizer, transition, trunc, gamma):
        
        # Compute the TD target
        with torch.no_grad():
            states, actions, rewards, next_states, _ = transition

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            next_actions = []

            #print("states :", states)
            #print("actions :", actions)
            #print("rewars :", rewards)
            #print("next_states :", next_states)

            for next_state in next_states: 
                next_action = self.agent.compute_action(next_state)
                next_actions.append(next_action.tolist())
                
                #print("next_actions: ", next_actions)
            
            next_actions = torch.FloatTensor(next_actions)

            q_values = self.forward(states, actions)
            #print("q_values :", q_values)
            q_next = self.forward(next_states, next_actions)
            #print("q_next :", q_next)
            #print("trunc :", trunc)
            targets = rewards + gamma * q_next * (1 - trunc)
            #print("target :", targets)

            targets = torch.FloatTensor(targets)

            loss = self.critic_criterion(q_values, targets)
            #print("loss :", loss)

            optimizer.zero_grad()
            #loss.backward() 
            optimizer.step()
        return loss
