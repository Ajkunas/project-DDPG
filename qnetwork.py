import torch
import torch.nn as nn

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
    trunc_batch = torch.FloatTensor(batch.trunc)
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
    trunc_batch = trunc_batch.unsqueeze(1)

    q_next = critic.forward(next_state_batch, next_action_batch)

    with torch.no_grad():
        targets = reward_batch + (1.0 - trunc_batch) * gamma * q_next

    # critic update
    optimizer.zero_grad()
    q_val = critic.forward(state_batch, action_batch)
    critic_loss = criterion(q_val, targets)
    critic_loss.backward() 
    optimizer.step()

    return critic_loss.item()
    