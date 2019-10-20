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

trainloader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True, num_workers = 0)

test = torchvision.datasets.CIFAR10(root = os.getcwd(), train = False, download = True, transform = transform)

testloader = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = False, num_workers = 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
        
        
        
inputsize = 32*32*3

class Mylenet(nn.Module):
    def __init__(self):
        super(Mylenet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024).cuda()
        self.fc2 = nn.Linear(1024, 1024).cuda()
        self.fc3 = nn.Linear(1024, 10).cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        self.softmax = nn.Softmax(1).cuda()

    def forward(self, x):
        x = x.to(device)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = (self.fc3(x))#Dense
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x
    
def simpleprune(amount_to_prune, net, device):
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
                
               
                
            # print(param[1,:], "param1") #This refers to the 1st neuron's weights for the layer                  
                # print(neuron)
    
mylenet = Mylenet()

#This has a performance of 51%
PATH = os.getcwd() + '\cifar_net_51_1st.pth'
mylenet.load_state_dict(torch.load(PATH))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.0001)

print('Finished Training')

simpleprune(10, mylenet, device)

print("postPrune")

# for i,param in enumerate(mylenet.parameters()):
    # print(param, i)

correct = 0
total = 0

print(mylenet)
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(-1,32*32*3)

        outputs = mylenet(inputs)
        _, predicted  = torch.max(outputs.data, 1)
        total += labels.size(0)
        # print(predicted, "predicted")
        # print(labels,"labels")
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))   
print('Unpruned network has 51% accuracy')    