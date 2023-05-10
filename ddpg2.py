import numpy as np
from helpers import NormalizedEnv, RandomAgent
from qnetwork2 import ReplayBuffer
from noise import OUActionNoise
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# Actor network implementation
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__() # refers to fact that it is a subclass of nn.Module and is inheriting all methods
        
        self.layer1 = nn.Linear(3, 32) # accepts the vector states as input (size 3)
        self.layer2 = nn.Linear(32, 32) 
        self.layer3 = nn.Linear(32, 1) # outputs the action (a scalar)
        
    def forward(self, states):
        x = self.layer1(states)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = torch.tanh(x) # output action between -1 and 1 
        
        return x
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, states, action):
        #print("states :", type(states))
        #print("action :", type(action))
        if  (type(action) is np.ndarray):
            action = torch.from_numpy(action).float()
            
        x = torch.cat([states, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    
# Implementation of a minimal ddpg agent
class DDPGAgent:
    def __init__(self, env, actor_learning_rate, critic_learning_rate, gamma, buffer_size, sigma, theta, tau, hidden_size=256):
        
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        
        self.gamma = gamma
        self.tau = tau
        
        #set up actor critic
        self.actor = PolicyNetwork(self.state_size, hidden_size, self.action_size)
        self.critic = QNetwork(self.state_size + self.action_size, hidden_size, self.action_size)
        
        # buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # gaussian noise
        self.noise = OUActionNoise(sigma, theta)
        
        # critic criterion
        self.critic_criterion  = nn.MSELoss()
        
        # initialize optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        # the target networks 
        self.actor_target = PolicyNetwork(self.state_size, hidden_size, self.action_size, actor_learning_rate)
        self.critic_target = QNetwork(self.state_size + self.action_size, hidden_size, self.action_size)
        
        # initiliaze target networks as copies of original networks
        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(param.data)
        
    def compute_action(self, state, deterministic=True): # deterministic regulates whether to add a random noise to the action or not
        
        if  (type(state) is np.ndarray):
            state = torch.from_numpy(state).float()
            
        action = self.actor_target.forward(state)
        action = action.detach().numpy()
        
        if (deterministic):
            action = self.noise.get_noisy_action(action)
        return action
    
    def update(self, transition):
        state, action, reward, next_state, trunc = transition
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        
        # critic loss
        q_val = self.critic.forward(state, action)
        
        # actor loss 
        next_action = self.compute_action(next_state)
        
        q_next = self.critic_target.forward(next_state, next_action)
        
        qprime = reward + self.gamma * q_next 
        
        critic_loss = self.critic_criterion(q_val, qprime)
        
        actionprime = self.actor.forward(state) # not sure though
        
        policy_loss = -self.critic.forward(state, actionprime).mean() # average output of the Q network
        
        # update the networks 
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        # updates the target network 
        self.update_target_params()
        
        return policy_loss, critic_loss
    
    def update_target_params(): 
        # soft update of target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        