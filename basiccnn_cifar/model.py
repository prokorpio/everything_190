# contains neural networks used by the RL agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr=1e-4):
        super().__init__()
        
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,self.action_size)
        
        self.Adamizer = optim.Adam(self.parameters(),lr=lr)
    
    def forward(self,input_state):
        x = F.relu(self.fc1(input_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = x.view((self.action_size, 2))# sum over columns will be 1
        x = torch.softmax(x, dim = 0)
        return x
