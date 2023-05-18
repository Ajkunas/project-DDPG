import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size) # accepts the vector states as input (size 3)
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, output_size) # outputs the action (a scalar)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.layer1(state)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = torch.tanh(x) # output action between -1 and 1 
        
        return x
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x