import numpy as np
import torch

class GaussianActionNoise: 
    def __init__(self, sigma):
        self.sigma = sigma
        
    def get_noisy_action(self, action):
        noisy_action = action + self.sigma*torch.randn_like(action)
        noisy_action = torch.clamp(noisy_action, -1, 1)
        return noisy_action # or max(-1, min(noisy_action, 1)), action need to be between -1 and 1


class OUActionNoise: 
    def __init__(self, device, action_space, sigma, theta):
        self.device = device
        self.sigma = sigma
        self.theta = theta
        self.action_dim = action_space.shape[0]
        
    def reset(self): 
        self.state = torch.zeros(self.action_dim).to(self.device)
    
    def evolve_state(self, action): 
        x = self.state
        self.state = (1.0 - self.theta)*x + self.sigma*torch.randn_like(action)
        return self.state
    
    def get_noisy_action(self, action):
        ou_noise = self.evolve_state(action)
        noisy_action = action + ou_noise
        noisy_action = torch.clamp(noisy_action, -1, 1)
        return noisy_action

    
    