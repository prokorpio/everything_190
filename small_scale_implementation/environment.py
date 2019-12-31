#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os

import torch
import torchvision.transforms as tf
import torchvision.datasets as ds
import torch.utils.data as data

import models_to_prune 

import torch.nn as nn
import torch.optim as optim
import time

import numpy as np


class PruningEnv:
    # NOTE: will prune a single layer for now

    def __init__(self, dataset='cifar10', model_type='basic'):

        # assign dataset
        self.dataset = dataset
        self.train_dl, self.test_dl = self.get_dataloaders()
        
        # build chosen model to prune
        self.model_type = model_type
        self.model = self.build_model_to_prune()
        # may pre-train the model here for N epochs

        # state
        self.layer_to_process = None # Layer to process, 
                                     # str name is usr-identified 
        self.state_size = 64 # TODO: Ask kuya Lejan what to use

        # per episode
        self.expis = 5    # num of experience before backprop for agent       
        self.xp_count = 0 # count xp for now, stopping is controlled 
                          # by the reward value not chaning any more

    def get_dataloaders(self):
        ''' imports the chosen dataset '''

        if self.dataset.lower() == 'cifar10':
            # copied from marcus' rat5, lines 17:29
            cifar10_trans = tf.Compose([tf.ToTensor(),
                                        tf.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5))])

            train = ds.CIFAR10(root = os.getcwd(),
                               train = True,
                               download = True,
                               transform = cifar10_trans)

            train_loader = data.DataLoader(train,
                                           batch_size = 32,
                                           shuffle = True,
                                           num_workers = 0)
            
            test = ds.CIFAR10(root = os.getcwd(),
                              train = False,
                              download = True,
                              transform = cifar10_trans)
            
            test_loader = data.DataLoader(test,
                                          batch_size = 64, # testing use less 
                                                           # memory, can afford 
                                                           # larger batch_size
                                          shuffle = False,
                                          num_workers = 0)

            return train_loader, test_loader
                              
        # TODO: use proper exception handling
        print('dataset not available') 
        return -1

    def build_model_to_prune(self): 
        ''' Builds the model to compress '''

        if self.model_type.lower() == 'basic' :

            return models_to_prune.BasicCNN()

        else:
            print('model not available') #TODO: use proper handling
            return -1

    def _get_state(self): # class private method, 
                           #access via: _PruningEnv__get_state()
        ''' Helper tool for step(), 
            gets the layer/state '''
        
        # get conv layer 
        # may keep an external pth file for original model
        for name, module in self.model.named_modules():
            if self.layer_to_process in name:
                #conv_layer_name = name
                conv_layer = module
                break

        #print(conv_layer_name)
        print('weight:',type(conv_layer.weight))
        print('bias:', type(conv_layer.bias))

        #TODO: process layer with kuya Lejan's state-rep

        # return processed state
        return conv_layer

   def _train_model(self, num_epochs=10): 
        ''' Helper tool for _calculate_reward(),
            trains the model being pruned '''
        
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(0.001))
        
        self.model.train()
        print('Training CNN model')
        for epoch in range(num_epochs):
            train_acc = []
            start_time = time.time()
            for idx, train_data in enumerate(self.train_dl):
                inputs, labels = train_data
                # TODO: transfer to device?
                
                optimizer.zero_grad()
                
                # forward
                preds = self.model(inputs) # forward pass
                loss = loss_func(preds,labels) # compute loss
                
                # backward
                loss.backward()  # compute grads
                optimizer.step() # update params w/ Adam update rule

                # print accuracy
                _, prediction = torch.max(preds, dim=1) # idx w/ max val is
                                                        # most confident class
                train_acc.append((prediction==labels).type(torch.double).mean())

                #if (idx+1) % 1500 == 0:
            elapsed_time = time.time() - start_time
            str_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print('Epoch [{}/{}] Step [{}/{}] | Loss: {:.4f} Acc: {:.4f} Time: {}'
                    .format(epoch+1, num_epochs, idx+1, len(self.train_dl), 
                            loss.item(), train_acc[-1], str_time))
        print('Training Done')

    def _evaluate_model(self):
        ''' Helper tool for _calculate_reward(),
            evaluates the model being pruned'''

        self.model.eval()
        print('Evaluating CNN model''')
        total = 0 # total number of labels
        correct = 0 # total correct preds
                
        with torch.no_grad():
            for test_data in self.test_dl:
                inputs, labels = test_data
                # TODO: transfer to device?

                preds = self.model(inputs) #forward pass
                
                _, prediction = torch.max(preds,dim=1)
                total += labels.size(0) # number of rows = num of samples
                correct += (prediction == labels).sum().item() 

        val_acc = correct/total
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
        
        return val_acc
                
    def _estimate_layer_flops(self):
        ''' Helper tool for _calculate_reward(),
            estimate conv layer flops,
            same as in AMC implementation '''

        for name, module in self.model.named_modules():
            if self.layer_to_process in name:
                #conv_layer_name = name
                conv_layer = module
                break

        for inputs, _ in self.train_dl:
            break # just to get input size

        input_h = inputs.size()[2]
        input_w = inputs.size()[3]
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]
        pad_h = conv_layer.padding[0]
        pad_w = conv_layer.padding[1]
        stride_h = conv_layer.stride[0]
        stride_w = conv_layer.stride[1]
        C_in = conv_layer.in_channels
        C_out = conv_layer.out_channels
        groups = conv_layer.groups

        filter_steps_h = (input_h + 2*pad_h - kernel_h)/stride_h + 1  
        filter_steps_w = (input_w + 2*pad_w - kernel_w)/stride_w + 1  

        layer_flops = C_out * (C_in/groups) * kernel_h*kernel_w \
                            * filter_steps_h * filter_steps_w
       
        return layer_flops

    def _calculate_reward(self): 
        ''' Helper tool for step(), 
            performs the ops to get reward'''

        # train for M epochs
        self._train_model(num_epochs=2)

        # test
        acc = self._evaluate_model() # acc is in {0,1}

        # get flops 
        flops = self._estimate_layer_flops()

        # get reward as func of acc and flops
        reward = -(1-acc)*np.log(flops)

        return reward

    def step(self, action):
        ''' Run one timestep '''

        # prune_layer(action) # perform action via MARCUS' function

        reward = self._calculate_reward()
        new_state = self._get_state()
        
        if self.xp_count == self.expis:
            done = True
            self.xp_count = 0
        else:
            done = False
            self.xp_count +=1

        return new_state, reward, done

    
    def reset(self):
        ''' gives starting state '''
        
        
    ''' added filter pruning function '''
    ''' layer_number = the layer to be pruned starts from 0 to 4 '''
    ''' indices = tensor of indices to be pruned i.e. [0,0,0,0,1,1,1,0,1,1,1,0...]'''
    ''' net = network to be pruned '''
    ''' device = cuda if cuda '''
    ''' amount_to_prune = deprecated value. previously used for generating indices '''
    def filterpruneindice(layer_number, indices, net, device, amount_to_prune):
    iter = 0
    iterbn = 0
    print("NUM_indices", indices.shape)
    #iterate through all the parameters of the network
    for layer in net.children():
        #hardcode to find the last conv layer
        #this is not needed for now as long as you set the last batchnorm layer to 0
        #proven empirically on the 3 layer 3*3 network
            
        #If convolutional layer
        if type(layer) == nn.Conv2d:
            print("Am at layer", iter)
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
                        masktuple = ((mask),)*a[0]
                        finalmask = torch.stack((masktuple),0)
                        # print("FINAL MASK SHAPE", finalmask.shape)
                        
                    # finalmask = torch.cat((finalmask,mask),2)
                    # print(param.data,"BEFORE")
                    # print(finalmask.size())
                    # print(param.data.size())
                    print(param.data.shape,"SIZE")
                    print(finalmask.shape,"SIZE")
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
            

