import numpy as np

class GaussianActionNoise: 
    def __init__(self, sigma):
        self.sigma = sigma
        
    def get_noisy_action(self, action):
        noisy_action = action + self.sigma*np.random.randn()
        # noisy_action = noisy_action.detach().numpy()
        return np.clip(noisy_action, -1, 1) # or max(-1, min(noisy_action, 1)), action need to be between -1 and 1


class OUActionNoise: 
    def __init__(self, sigma, theta):
        self.sigma = sigma
        self.theta = theta
        
    def reset(self): 
        self.state = 0
    
    def evolve_state(self): 
        x = self.state
        self.state = (1 - self.theta)*x + self.sigma*np.random.randn()
        return self.state
    
    def get_noisy_action(self, action):
        ou_noise = self.evolve_state()
        noisy_action = action + ou_noise
        return np.clip(noisy_action, -1, 1)
        
     
    
    
    