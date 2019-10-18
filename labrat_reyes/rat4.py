import os

import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as f

import torch.optim as optim

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
#slf_size is the indices containing the nth layers amount of neurons
slf_list = [20,20,10]
fclist = []

def createmodel(slf_list):
    #idea from amc
    for i, size in enumerate(slf_list):
        #First layer
        if i == 0:
            fclist.append(nn.Linear(inputsize, size))
        else:
            fclist.append(nn.Linear(slf_list[i-1], size))

    net = nn.Sequential(*fclist).to(device)
    print(net)
    return net
    
def simpleprune(amount_to_prune, net, device):
    pruned_count = 0
    
    for i, param in enumerate(net.parameters()):
        a = param.size()
        # print(a, "a")
        for j, neuron_array in enumerate(param.data):
            # print(neuron_array.size(), "neuron_array",j,i)
            if len(a) == 1:  #Bias
                if abs(neuron_array.cpu().numpy()) < 0.01:
                    param.data[j] = torch.tensor(0.0, requires_grad = True, device = device)
                    pruned_count = pruned_count + 1
            else:
                for k, neuron, in enumerate(neuron_array):
                    if abs(neuron.cpu().numpy()) < 0.01:
                        param.data[j] = torch.tensor(0.0, requires_grad = True, device = device)
                        pruned_count = pruned_count + 1
                # print(neuron)
    
mylenet = createmodel(slf_list)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.0001)

print("Starting Training")
for epoch in range(0):
    
    running_loss = 0.0
    print("NEW EPOCH")
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(-1,32*32*3)
        
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
for param in mylenet.parameters():
    print(param)

simpleprune(10, mylenet, device)

for param in mylenet.parameters():
    print(param)
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