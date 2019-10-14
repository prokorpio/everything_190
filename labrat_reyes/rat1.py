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

mylenet = Mylenet()
mylenet.to(device)

print(mylenet.parameters())
print("Parameters")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mylenet.parameters(), lr = 0.0001)

print("Starting Training")
for epoch in range(17):
    
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

PATH = os.getcwd() + '\cifar_net.pth'
print("Saving to path", PATH)
torch.save(mylenet.state_dict(), PATH)

dataiter = iter(testloader)
images,labels = dataiter.next()[0].to(device),dataiter.next()[1].to(device)

images = images.view(-1,32*32*3)
outputs = mylenet(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))

correct = 0
total = 0
mylenet = Mylenet()
mylenet.load_state_dict(torch.load(PATH))

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