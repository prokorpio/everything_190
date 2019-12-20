# Import libraries
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self):
    
        filters = 64
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, filters,3,stride = 2)
        self.conv2 = nn.Conv2d(filters,filters*2,3,stride = 2)
        self.conv3 = nn.Conv2d(filters*2,filters*4,3,stride = 2)
        self.dropout = nn.Dropout(p = 0.3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*3*3, 10)
        # self.fc2 = nn.Linear(1024, 1024).cuda()
        # self.fc3 = nn.Linear(1024, 10).cuda()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        #x = x.to(device)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = x.view(-1,256*3*3)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)

        return x






