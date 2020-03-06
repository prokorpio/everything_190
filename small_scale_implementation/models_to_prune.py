# contains the neural networks to be pruned

# Import libraries
import logging
logging.basicConfig(level=logging.INFO,
                    format=('%(levelname)s:'+
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self):
    
        filters = 64
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, filters,3,stride=2)
        self.conv2 = nn.Conv2d(filters,filters*2,3)
        self.conv3 = nn.Conv2d(filters*2,filters*4,3,stride=2)
        self.conv4 = nn.Conv2d(filters*4, filters*8, 3)
        self.dropout = nn.Dropout(p = 0.3)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters*2)
        self.bn3 = nn.BatchNorm2d(filters*4)
        self.bn4= nn.BatchNorm2d(filters*8)
        self.fc1 = nn.Linear((filters*8)*4*4, 10)
        # self.fc2 = nn.Linear(1024, 1024).cuda()
        # self.fc3 = nn.Linear(1024, 10).cuda()
        #self.sigmoid = nn.Sigmoid()
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
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.bn4(x)

        #print(x.shape,"Shape of X")
        x = x.view(-1,512*4*4)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)

        return x

class RandBasicCNN(nn.Module):
    '''Rand-init equiv of BasicCNN
    '''
    def __init__(self,filter_counts):
    
        L1_filters, L2_filters, L3_filters, L4_filters = filter_counts
        # number of filters per layer depends on how much was pruned
        logging.info('Rand-Subnet Filters: {}'.format(filter_counts))
        super(RandBasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, L1_filters,3,stride=2)
        self.conv2 = nn.Conv2d(L1_filters,L2_filters,3)
        self.conv3 = nn.Conv2d(L2_filters,L3_filters,3,stride=2)
        self.conv4 = nn.Conv2d(L3_filters, L4_filters, 3)
        self.dropout = nn.Dropout(p = 0.3)
        self.bn1 = nn.BatchNorm2d(L1_filters)
        self.bn2 = nn.BatchNorm2d(L2_filters)
        self.bn3 = nn.BatchNorm2d(L3_filters)
        self.bn4 = nn.BatchNorm2d(L4_filters)
        self.fc1 = nn.Linear(L4_filters*4*4, 10)
        self.linear_in = L4_filters
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
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.bn4(x)
        x = x.view(-1,self.linear_in*4*4)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)

        return x

class TestCNN(nn.Module):
    def __init__(self):
    
        filters = 16
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, filters, 3, stride = 2)
        self.conv2 = nn.Conv2d(filters, filters*2, 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters*2)
        self.fc1 = nn.Linear((filters*2)*3*3, 10)
        # self.fc2 = nn.Linear(1024, 1024).cuda()
        # self.fc3 = nn.Linear(1024, 10).cuda()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        #x = x.to(device)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x.view(-1,128*3*3)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)

        return x





