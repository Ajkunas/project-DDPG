import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch

from networks import *
from buffer import *

class DDPGAgent:
    def __init__(self, device, env, learning_rate, buffer_size, gamma, tau):
        
        self.device = device
        
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.hidden_size = 32
        
        self.gamma = gamma
        self.tau = tau
        
        #initialize the networks
        self.actor = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.actor_target = PolicyNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        
        self.critic = QNetwork(self.state_size + self.action_size, self.hidden_size, self.action_size).to(self.device)
        self.critic_target = QNetwork(self.state_size + self.action_size, self.hidden_size, self.action_size).to(self.device)
        
        self.buffer = ReplayBuffer(buffer_size)
        
        self.critic_criterion = nn.MSELoss()
        
        # define optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        #initialize the targets
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
    def compute_action(self, state, noise, deterministic=True): # deterministic regulates whether to add a random noise to the action or not

        state = state.to(self.device)

        self.actor.eval()
        action = self.actor.forward(state)
        self.actor.train()
        action = action.data
        
        if (deterministic):
            action = noise.get_noisy_action(action)
        return action
    
    def update(self, batch):
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        trunc_batch = torch.cat(batch.trunc).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        
        reward_batch = reward_batch.unsqueeze(1)
        trunc_batch = trunc_batch.unsqueeze(1)
        
        with torch.no_grad():
            next_action_batch = self.actor_target.forward(next_state_batch)
            q_next = self.critic_target.forward(next_state_batch, next_action_batch)
            targets = reward_batch + (1.0 - trunc_batch) * self.gamma * q_next
            
        
        # update critic
        self.critic_optimizer.zero_grad()
        q_val = self.critic.forward(state_batch, action_batch)
        critic_loss = self.critic_criterion(q_val, targets)
        critic_loss.backward() 
        self.critic_optimizer.step()
        
         # update actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update the target networks
        update_target_params(self.actor_target, self.actor, self.tau)
        update_target_params(self.critic_target, self.critic, self.tau)

        return policy_loss.item(), critic_loss.item()
        

def update_target_params(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
        