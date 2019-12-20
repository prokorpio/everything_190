import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr=1e-3):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,action_size)
        
        self.Adamizer = optim.Adam(self.parameters(),lr=lr)
    
    def forward(self,input_state):
        x = F.relu(self.fc1(input_state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1) # sum over columns will be 1
        
        return x
