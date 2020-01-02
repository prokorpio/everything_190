#!/usr/bin/env python
# coding: utf-8


# Import Libraries
from models_to_prune import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import torchvision.transforms as tf
import torchvision.datasets as ds
import torch.utils.data as data

import os
import time
import torch

from torch.utils.data import TensorDataset, DataLoader

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# See layer shapes and create dataset for autoenc
'''
cnn = BasicCNN()
for name, layer in cnn.named_modules():
    if 'conv' in name:
        filters = layer.weight.data.clone()
        print(name,':',filters.size())
        # reduce last dim of 3x3 to 1x1 then squeeze
        pooled_filter = torch.squeeze(F.avg_pool2d(filters,
                                                   filters.size()[-1])) 
        pooled_filter = pooled_filter*1000 # scaling up the magnitudes 
        print("pooled :",pooled_filter.size())

# interpet 4d tensors as set of 3d blocks.  
#print(pooled_filter[0])
print(pooled_filter.cpu().mean())


# Create "dataset" of pooled layers
# Convert set of 3d blocks to set of flat 2d maps. 

# Create pad-tensor container, same size as biggest layer
filter_repeats = 10000 # each filter layer will be repeated this many times
feat_size = 16  # size of 2d maps
magnitude_scaler = 100
state_rep = torch.zeros([filter_repeats, 512, feat_size,feat_size]) # set of N padded [512,16,16] 
                                                        # tensors for each of the 4 layers  
for i in range(filter_repeats):
    cnn = BasicCNN()
    for name, layer in cnn.named_modules():
        if 'conv' in name:
            filters = layer.weight.data.clone()
            pooled_filter = torch.squeeze(F.avg_pool2d(filters,
                                                       filters.size()[-1]))
            pooled_filter = pooled_filter*magnitude_scaler # scaling up the magnitudes 
            conv_layer_num = int(name[-1])
            size = pooled_filter.size()
            #if conv_layer_num == 1:
            #    pads = (feat_size//2) - size[-1]//2
            #    state_rep[i, :size[0], feat_size//2, pads-1 :-pads] = pooled_filter  # copy in center
            #elif conv_layer_num == 2:
            #    pads = (feat_size//2) - 4
            #    state_rep[i+filter_repeats, :size[0], pads:-pads, pads:-pads] = pooled_filter.view(size[0],8,8)
            #elif conv_layer_num == 3:
            #    pads_r = (feat_size//2) - 4
            #    pads_c = (feat_size//2) - 8
            #    state_rep[i+filter_repeats*2, :size[0], :8, :16] = pooled_filter.view(size[0],8,16)
            if conv_layer_num == 4:
                state_rep[i] = pooled_filter.view(size[0],16,16) # same size as init state_rep
                #state_rep[i+filter_repeats*3] = pooled_filter.view(size[0],16,16) # same size as init state_rep
                #print(state_rep[i+filter_repeats*3][0])

val_rep = filter_repeats//10
validation = torch.zeros([val_rep*4, 512, feat_size,feat_size]) # set of N padded [512,16,16] 
                                                        # tensors for each of the 4 layers  
for i in range(val_rep):
    cnn = BasicCNN()
    for name, layer in cnn.named_modules():
        if 'conv' in name:
            filters = layer.weight.data.clone()
            pooled_filter = torch.squeeze(F.avg_pool2d(filters,
                                                       filters.size()[-1]))
            pooled_filter = pooled_filter*magnitude_scaler # scaling up the magnitudes 
            conv_layer_num = int(name[-1])
            size = pooled_filter.size()
            if conv_layer_num == 1:
                pads = (feat_size//2) - size[-1]//2
                validation[i, :size[0], feat_size//2, pads-1 :-pads] = pooled_filter  # copy in center
            elif conv_layer_num == 2:
                pads = (feat_size//2) - 4
                validation[i+val_rep, :size[0], pads:-pads, pads:-pads] = pooled_filter.view(size[0],8,8)
            elif conv_layer_num == 3:
                pads_r = (feat_size//2) - 4
                pads_c = (feat_size//2) - 8
                validation[i+val_rep*2, :size[0], :8, :16] = pooled_filter.view(size[0],8,16)
            elif conv_layer_num == 4:
                validation[i+val_rep*3] = pooled_filter.view(size[0],16,16) # same size as init state_rep

'''

# Build Autoencoder Class

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__(encoding_dim = 64)
        self.encoding_dim = encoding_dim 
        self.encoder = nn.Sequential( # input size is [512,16,16]
            nn.Conv2d(512, 256, 3),  # b, 256, 14, 14
            nn.Sigmoid(),
            nn.MaxPool2d(4, stride=1),  # b, 256, 11, 11
            nn.Conv2d(256, 128, 3),  # b, 128, 9, 9
            
            nn.Sigmoid(),
            nn.MaxPool2d(3, stride=1),  # b, 128, 7, 7
            nn.Conv2d(128, 64, 3),  # b, 64, 5, 5
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=1),# b, 64, 4, 4

            nn.Flatten(), #from dim=1 to -1
            nn.Linear(64*4*4,self.encoding_dim)
        )
        
        self.latent_to_map = nn.Linear(self.encoding_dim, 64*4*4)
        self.decoder = nn.Sequential(    
            nn.ConvTranspose2d(64, 128, 4, stride=1),  # b, 64,7,7
            nn.Sigmoid(),
            nn.ConvTranspose2d(128, 256, 5, stride=1),  # b, 256, 11, 11
            nn.Sigmoid(),
            nn.ConvTranspose2d(256, 512, 6),  # b, 512, 16, 16
          #  nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_to_map(x).view(-1,64,4,4) 
        x = self.decoder(x)
        return x

#model = autoencoder().cuda()
#print(model)

# Training Loop
'''
num_epochs = 10
batch_size = 16
learning_rate = 1e-3

state_rep = state_rep.to(device)
train_dl = DataLoader(state_rep, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(validation, batch_size=batch_size, shuffle=True)

model = autoencoder().cuda()
MSE_criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
                                                # removed weight_decay=1e-5
#sample = iter(train_dl).next()
#print(sample.cpu().mean())
#raise KeyboardInterrupt


for epoch in range(num_epochs):
    for i, data in enumerate(train_dl):
        model.train()
        data = Variable(data).cuda()
        # ===================forward=====================
        output = model(data)
        #loss = torch.sum(torch.log(torch.cosh(data-output)))
        loss = torch.mean(torch.abs(data-output)) # MAE criterion
        rand_loss = torch.mean(torch.abs(data-torch.rand_like(output))) # sanity check
        #loss = MSE_criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         # ===================log========================
        if i % 100 == 0:
            model.eval()
            val_loss = 0
            num_batches = 0
            ave_val_loss = 0
            with torch.no_grad():
                for val in valid_dl:
                    val = Variable(val).cuda()
                    val_output = model(val)
                    #val_loss += torch.sum(torch.log(torch.cosh(val-val_output)))
                    val_loss += torch.mean(torch.abs(val - val_output))
                    #val_loss += MSE_criterion(val_output,val)
                    num_batches += 1
                ave_val_loss = val_loss/num_batches


            print('epoch [{}/{}], loss:{}, val_loss:{}, rand_loss:{}'
                .format(epoch+1, num_epochs, loss.item(), ave_val_loss, rand_loss.item()))
            torch.save(model.state_dict(), './conv_autoencoder.pth')

'''

# Test Model
'''                
loaded_model = autoencoder().cuda()
checkpoint_dict = torch.load('conv_autoencoder.pth')
loaded_model.load_state_dict(checkpoint_dict)

encoded_states = loaded_model.encoder(data)
print(encoded_states.size())
#print(encoded_states[1])
output_ = loaded_model(data)
MAE_loss = torch.mean(torch.abs(data-output))
MSE_loss = MSE_criterion(output, data)
print(MAE_loss.item())
print(MSE_loss.item())
print(output[0][0])
'''
