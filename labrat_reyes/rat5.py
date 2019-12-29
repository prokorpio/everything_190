import os

import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as f

import torch.optim as optim
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train = torchvision.datasets.CIFAR10(root = os.getcwd(), train = True, download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True, num_workers = 0)

test = torchvision.datasets.CIFAR10(root = os.getcwd(), train = False, download = True, transform = transform)

testloader = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = False, num_workers = 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
        
        
        
inputsize = 32*32*3
###Three layer cnn
class Mylenet(nn.Module):
    def __init__(self):
    
        filters = 64
        super(Mylenet, self).__init__()
        self.conv1 = nn.Conv2d(3, filters,3,stride = 2).cuda()
        self.conv2 = nn.Conv2d(filters,filters*2,3,stride = 2).cuda()
        self.conv3 = nn.Conv2d(filters*2,filters*4,3,stride = 2).cuda()
        self.dropout = nn.Dropout(p = 0.3).cuda()
        self.bn1 = nn.BatchNorm2d(64).cuda()
        self.bn2 = nn.BatchNorm2d(128).cuda()
        self.bn3 = nn.BatchNorm2d(256).cuda()
        self.fc1 = nn.Linear(256*3*3, 10).cuda()#why is it 262144
        # self.fc2 = nn.Linear(1024, 1024).cuda()
        # self.fc3 = nn.Linear(1024, 10).cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        self.softmax = nn.Softmax(1).cuda()

    def forward(self, x):
        x = x.to(device)
        x = f.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.bn1(x)
        x = f.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = f.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = x.view(-1,256*3*3)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)
        return x
##Deprecated function    
def simpleprune(layer, amount_to_prune, net, device):
    print("STARTING PRUNING---------------------------------------")
    to_prune_amount = 700 #for now keep it within 0 to 1024
    pruned_count = 0
    layer_neuron_dict = []
    #i represents the layer. In thise case I only care about i = 0,2
    for i, param in enumerate(net.parameters()):
        if i != 0:
            continue
        a = param.size() #if len(a) == 1 thats the bias layer ignore it for now
        
        #j represents the neuron IF the layer is a weights layer
        for j, neuron_array in enumerate(param.data):
            # print(neuron_array.size(),"neuron_array",j,i)
            
            #
            if len(a) == 1:  #Bias
                continue
                if abs(neuron_array.cpu().numpy()) < 0.01:
                    param.data[j] = torch.tensor(0.0, requires_grad = True, device = device)
                    pruned_count = pruned_count + 1
            else:
                #If the weights for a certain neuron have a mean of x:
                if torch.mean(abs(neuron_array.cpu())) < 0.2 and pruned_count < to_prune_amount:
                    # print("Pruned one neuron")\
                    print(param.data[j], "pre")
                    
                    #Zero out the weights connected to that neuron.
                    #I.e. Chop it off
                    param.data[j] = torch.zeros(3072)
                    print(param.data[j], "post")
                    
                    pruned_count = pruned_count + 1
def maskbuildbias(indices,sizes):
    mask0 = torch.zeros(0).to(device)
    mask1 = torch.ones(1).to(device)
    
    for i, val in enumerate(indices):
        if i == 0:
            if val == 0:
                finalmask = mask0
            else:
                finalmask = mask1
        else:
            if val == 0:
                finalmask = torch.cat((finalmask, mask0),0)
            else:
                finalmask = torch.cat((finalmask, mask1),0)
    
    return finalmask
        
def filterprune(layer_number, amount_to_prune, net, device):
    iter = 0
    
    #iterate through all the parameters of the network
    for layer in net.children():
        # print(type(layer))
        # print("Layer number:", iter)
        
        #If convolutional layer
        if type(layer) == nn.Conv2d:
            
            #If not the layer to be pruned, skip the below
            if iter != layer_number:
                iter = iter + 1
                continue
                
            #enumerate through all the contents of the layer.
            #for a conv layer thats: 1. weights 2. biases
            for i, param in enumerate(layer.parameters()):
            
                #use the param size to determine if weight or bias
                a = param.size()
                
                #If bias
                if (len(a) == 1):
                    # print("A",a[0])
                    
                    #Multiply param.data with a mask of zeros up to the desired index, all else are filled with ones
                    mask = torch.cat((torch.zeros(amount_to_prune).to(device),torch.ones(a[0]-amount_to_prune).to(device)),0)
                    # print(mask, "MASK")
                    # print(param.data)
                    param.data = torch.mul(param.data,mask)
                    # print(param.data, "AFTER")
                    
                    #Iterate the cnn layer counter
                    iter = iter + 1
                #If weights
                else:
                    print(a,"a")
                    print(a[1],"Num filter", a[2],"KernelSize")
                    #mask per channel
                    mask = torch.cat((torch.zeros((amount_to_prune,a[2],a[2])).to(device),torch.ones((a[0]-amount_to_prune,a[2],a[2])).to(device)),0)
                    masktuple = ((mask),)*a[1]
                    finalmask = torch.stack((masktuple),1)
                    # finalmask = torch.cat((finalmask,mask),2)
                    # print(param.data,"BEFORE")
                    # print(finalmask.size())
                    # print(param.data.size())
                    param.data = torch.mul(param.data,finalmask)
                    # print(param.data,"AFTER")
                    
                    #sanity check for mask
                    # try:
                        # print(param.data[254],"Should be all zeros")
                    # except:
                        # pass
                    #iterate the cnn layer counter
                    iter = iter + 1
            # print(param[1,:], "param  1") #This refers to the 1st neuron's weights for the layer                  
                # print(neuron)

#Initialize network
mylenet = Mylenet()
mylenet.to(device)

print("Skipping Training")

for epoch in range(0):
    pass
#Load a pretrained model
PATH = os.getcwd() + '\cifar_net_3layercnn.pth'
# This has a performance of 51%
mylenet.load_state_dict(torch.load(PATH))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.001)

print('Finished Training')

filterprune(0, 21, mylenet, device)
filterprune(1, 42, mylenet, device)
filterprune(2, 85, mylenet, device)
print("postPrune")
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(-1,3,32,32)

        outputs = mylenet(inputs)
        _, predicted  = torch.max(outputs.data, 1)
        total += labels.size(0)
        # print(predicted, "predicted")
        # print(labels,"labels")
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f' % (100 * correct / total))   
print('Unpruned network has 64%ish accuracy')    

print("maskbuilding bias")
print(maskbuildbias([0,0,1,1,0,0], 0))