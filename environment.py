#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os

import torch
import torchvision.transforms as tf
import torchvision.datasets as ds
import torch.utils.data as data

import models_to_prune 


class PruningEnv:
    # NOTE: will prune a single layer for now

    def __init__(self, dataset, model_type):

        # assign dataset
        self.dataset = dataset
        self.train_dl, self.test_dl = self.get_dataloaders()
        
        # build chosen model to prune
        self.model_type = model_type
        self.model = self.build_model_to_prune()

        # state
        self.layer_to_process = None # Layer to process, name is usr-identified 
        self.state_size = 64 # TODO: What to use?

    

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


    def get_state(self): 
        ''' Helper tool for step(), 
            gets the layer/state'''
        
        # get conv layer 
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


    def calculate_reward(self, model): 
        ''' Helper tool for step(), 
            performs the ops to get reward'''


    def step(self, action):
        ''' Perform one timestep '''



    def reset(self):
        ''' gives starting state '''
