# contains neural networks used by the RL agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr=1e-4):
        super().__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512,action_size)
        
        self.Adamizer = optim.SGD(self.parameters(),lr=lr)
    
    def forward(self,input_state):
        x = F.relu(self.fc1(input_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        ###One prob for keep one for prune
        #x = x.view((self.action_size, 2))
        
        x = torch.softmax(x, dim = 0)
        # print("xshape", x.shape)
        return x
