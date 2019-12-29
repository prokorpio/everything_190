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

from pfec import prune_step


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
    
        filters = 3
        super(Mylenet, self).__init__()
        self.conv1 = nn.Conv2d(3, filters,3,stride = 2).cuda()
        self.conv2 = nn.Conv2d(filters,filters,3,stride = 2).cuda()
        self.conv3 = nn.Conv2d(filters,filters,3,stride = 2).cuda()
        self.dropout = nn.Dropout(p = 0.3).cuda()
        self.bn1 = nn.BatchNorm2d(3).cuda()
        self.bn2 = nn.BatchNorm2d(3).cuda()
        self.bn3 = nn.BatchNorm2d(3).cuda()
        self.fc1 = nn.Linear(3*3*3, 10).cuda()#why is it 262144
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
        x = x.view(-1,3*3*3)
        x = (self.fc1(x))#Dense
        x = self.softmax(x)
        return x
                
def maskbuildbias(indices):
    mask0 = torch.zeros(1).to(device)
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

#This thing is stacked too
def maskbuildweight(indices, kernelsize):
    mask0 = torch.zeros((1,kernelsize, kernelsize))
    mask1 = torch.ones((1,kernelsize, kernelsize))
    
    for i, val in enumerate(indices):
        #initialize the mask
        if i == 0:
            if val == 0:
                finalmask = mask0
            else:
                finalmask = mask1
        #concatenate the masks
        else:
            if val == 0:
                finalmask = torch.cat((finalmask, mask0),0)
            else:
                finalmask = torch.cat((finalmask, mask1),0)
        # print("finalmaskshape", finalmask.shape)
    # print(finalmask)
    return finalmask
    
#for next layer
#you build the mask based on the 
#previous layer's indices but stack it according to this layer's indices
def maskbuildweight2(prev_indices, kernel1, kernel2):
    mask0 = torch.zeros((1,kernel1, kernel2))
    mask1 = torch.ones((1,kernel1, kernel2))
    
    #build on a per channel basis
    for i, val in enumerate(prev_indices):
        #initialize the mask
        if i == 0:
            if val == 0:
                finalmask = mask0
            else:
                finalmask = mask1
        #concatenate the masks
        else:
            if val == 0:
                finalmask = torch.cat((finalmask, mask0),0)
            else:
                finalmask = torch.cat((finalmask, mask1),0)
        # print("finalmaskshape", finalmask.shape)
        
        
    #stack on a per filter basis
    #meaning change torch.stack dimension to 0
    #but masktuple is still multiplied by number of filters since it is about hte current layers filters
    
    return finalmask

def filterpruneindice(layer_number, indices, net, device, amount_to_prune):
    iter = 0
    iterbn = 0
    
    #iterate through all the parameters of the network
    for layer in net.children():
        #hardcode to find the last conv layer
        #this is not needed for now as long as you set the last batchnorm layer to 0
        #proven empirically on the 3 layer 3*3 network
            
        #If convolutional layer
        if type(layer) == nn.Conv2d:
            
            #If not the layer to be pruned, skip the below
            if iter != layer_number and iter != layer_number + 1:
                iter = iter + 1
                continue
                
            #enumerate through all the contents of the layer.
            #use a different mask depending on whether this is current or next
            #there should be no bias change if this is for the next
            #for a conv layer thats: 1. weights 2. biases
            for i, param in enumerate(layer.parameters()):
            
                #use the param size to determine if weight or bias
                a = param.size()
                
                #If bias 
                #then make the mask for current only
                if (len(a) == 1):

                    #it has to be stacked conditions so that it doesn't go to the "else"
                    if iter == layer_number:
                        # print("A",a[0])
                        
                        #Multiply param.data with a mask of zeros up to the desired index, all else are filled with ones
                        
                        mask = maskbuildbias(indices)
                        param.data = torch.mul(param.data,mask)

                        
                        #Iterate the cnn layer counter
                #If weights
                else:
                    print(a,"a")
                    print(a[1],"Num filter", a[2],"KernelSize")
                    #mask per channel
                    if iter == layer_number:
                        mask = maskbuildweight(indices, a[2])
                        # print("MASK SHAPE", mask.shape)
                        masktuple = ((mask),)*a[1]
                        finalmask = torch.stack((masktuple),1)
                        # print("FINAL MASK SHAPE", finalmask.shape)
                    elif iter == layer_number+1:
                        print("THIS HAPPENED FOR LAYER NUMBER",layer_number)
                        mask = maskbuildweight2(indices, a[2], a[3])
                        # print("MASK SHAPE", mask.shape)
                        masktuple = ((mask),)*a[1]
                        finalmask = torch.stack((masktuple),0)
                        # print("FINAL MASK SHAPE", finalmask.shape)
                        
                    # finalmask = torch.cat((finalmask,mask),2)
                    # print(param.data,"BEFORE")
                    # print(finalmask.size())
                    # print(param.data.size())
                    param.data = torch.mul(param.data,finalmask.to(device))
                    # print(param.data,"AFTER")
            iter = iter + 1    
        if type(layer) == nn.BatchNorm2d:
            for i , param in enumerate(layer.parameters()):
                if iterbn == layer_number:
                    # print("A",a[0])
                    
                    #Multiply param.data with a mask of zeros up to the desired index, all else are filled with ones
                    mask = maskbuildbias(indices)
                    print(mask, "MASK")
                    # print(param.data)
                    param.data = torch.mul(param.data,mask)
            iterbn = iterbn + 1
            

#Initialize network
mylenet = Mylenet()
mylenet.to(device)

print("Skipping Training")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.001)
for epoch in range(0):
    running_loss = 0.0
    print("NEW EPOCH")
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        
        optimizer.zero_grad()
        outputs = mylenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            print(torch.cuda.memory_allocated(device=device), "Device Memory")
            running_loss = 0.0

print('Finished Training')
#Load a pretrained model
PATH = os.getcwd() + '\cifar_net_rat7_2.pth'
# This has a performance of 51%
# torch.save(mylenet.state_dict(), PATH)
mylenet.load_state_dict(torch.load(PATH))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.001)

print('Finished Training')
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

print('Accuracy of the nonindices network on the 10000 test images: %f' % (100 * correct / total))   
print('Unpruned network has 64%ish accuracy')   

# for layer in mylenet.children():
    # print(layer)
# print("ENDOFLAYER")

amount_to_prune = 1
indices = torch.cat((torch.zeros(amount_to_prune).to(device),torch.ones(3-amount_to_prune).to(device)),0)
filterpruneindice(0, indices, mylenet, device,1)

amount_to_prune = 2
indices = torch.cat((torch.zeros(amount_to_prune).to(device),torch.ones(3-amount_to_prune).to(device)),0)
filterpruneindice(2, indices, mylenet, device,1)

print("INDICEPARAMETERS")
print(mylenet.state_dict())


correct = 0
total = 0
layernum = 0

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

print('Accuracy of the nonindices1 network on the 10000 test images: %f' % (100 * correct / total))   
print('Unpruned network has 64%ish accuracy')   



