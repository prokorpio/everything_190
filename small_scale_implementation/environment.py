#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import copy

import torch
import torchvision.transforms as tf
import torchvision.datasets as ds
import torch.utils.data as data

import models_to_prune 
from temp_files.state_rep_autoencoder import autoencoder

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import numpy as np

import logging
logging.basicConfig(level=logging.INFO, 
                    format=('%(levelname)s:' +
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

class PruningEnv:

    def __init__(self, dataset='cifar10', 
                 model_type='basic',
                 state_size = 1024):

        # assign dataset
        self.dataset = dataset
        self.train_dl, self.test_dl = self.get_dataloaders()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build chosen model to prune
        self.model_type = model_type
        self.model = self._build_model_to_prune().to(self.device)
        #print("Starting Pre-Training")
        #self._train_model(num_epochs=0)
        #self.init_full_weights = copy.deepcopy(self.model.state_dict()) 
                                    # initially, model to be pruned has full-params
                                    # used in reset_to_k()

        # set training parameters
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(0.0008))

        # state
        self.layer_to_process = None # Layer to process, 
                                     # str name is usr-identified 
        self.state_size = state_size 
        #autoenc = autoencoder(self.state_size)
        #pretrained_autoenc_dict = torch.load('conv_autoencoder.pth',
        #                                     map_location=self.device)
        #autoenc.load_state_dict(pretrained_autoenc_dict)
        #autoenc.eval() # dont store grads
        #self.state_encoder = autoenc.encoder

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

    def _build_model_to_prune(self): 
        ''' Builds the model to compress '''

        if self.model_type.lower() == 'basic' :

            return models_to_prune.BasicCNN()

        else:
            print('model not available') #TODO: use proper handling
            return -1

    def get_state(self): 
        ''' Gets the layer/state '''
        
        # get conv layer 
        # may keep an external pth file for original model
        for name, module in self.model.named_modules(): # this model changes
            if self.layer_to_process in name:
                conv_layer = module
                #logging.info(module)
                break

        filter_weights = conv_layer.weight.data.clone() # copy params
        #TODO: what about the bias tensor ?
        pooled_filter = torch.squeeze(F.avg_pool2d(filter_weights,
                                                   filter_weights.size()[-1]))
        pooled_filter = pooled_filter*1000 # scale up magnitudes for encoder
        padded_state_rep = torch.zeros([1,512,16,16]) # largest pooled filter
                                                      # size is [512,256].
                                                      # think of 1 as batchsize
        state_rep = pooled_filter.mean(axis = 1)
        #print(state_rep.shape,"SHAPE")
        state_rep_padded = torch.zeros([512])
        state_rep_padded[0:state_rep.shape[0]] = state_rep.cpu()



        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(0.0008))
        

        #Fake train. To obtain gradients
        for idx, train_data in enumerate(self.train_dl):
                inputs, labels = train_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # TODO: transfer to device?
                
                optimizer.zero_grad()
                # forward
                preds = self.model(inputs) # forward pass
                loss = loss_func(preds,labels) # compute loss
                
                # backward
                loss.backward()  # compute grads
                for key, var in self.model.named_parameters():

                    if key == self.layer_to_process + '.weight':
                        pooled_grad = torch.squeeze(F.avg_pool2d(var.grad,
                                                           var.grad.size()[-1]))
                        pooled_grad = pooled_grad*1000
                        grad_rep = pooled_grad.mean(axis = 1)

                        break
                break
        
        grad_rep_padded = torch.zeros([512])
        grad_rep_padded[0:grad_rep.shape[0]] = grad_rep
        state_rep_final = torch.cat((state_rep_padded,grad_rep_padded),0)



        #temp for a verying state
        


        # return processed state
        return state_rep_final

    def _train_model(self, num_epochs=10): 
        ''' Helper tool for _calculate_reward(),
            trains the model being pruned '''
        
        self.model.train()
        logging.info('Training CNN model')
        for epoch in range(num_epochs):
            train_acc = []
            start_time = time.time()
            for idx, train_data in enumerate(self.train_dl):
                inputs, labels = train_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                # forward
                preds = self.model(inputs) # forward pass
                loss = self.loss_func(preds,labels) # compute loss
                
                # backward
                loss.backward()  # compute grads
                
                self.optimizer.step() # update params w/ Adam update rule

                # print accuracy
                _, prediction = torch.max(preds, dim=1) # idx w/ max val is
                                                        # most confident class
                train_acc.append((prediction==labels).type(torch.double).mean())

                if (idx+1) % 2 == 0:
                    elapsed_time = time.time() - start_time
                    str_time = time.strftime("%H:%M:%S", 
                                             time.gmtime(elapsed_time))
                    # print(('Epoch [{}/{}] Step [{}/{}] | ' + 
                           # 'Loss: {:.4f} Acc: {:.4f} Time: {}')
                           # .format(epoch+1, num_epochs, idx+1, 
                                   # len(self.train_dl), 
                                   # loss.item(), train_acc[-1], 
                                   # str_time))
        logging.info('Training Done')

    def _evaluate_model(self):
        ''' Helper tool for _calculate_reward(),
            evaluates the model being pruned'''

        self.model.eval()
        logging.info('Evaluating CNN model''')
        total = 0 # total number of labels
        correct = 0 # total correct preds

        with torch.no_grad():
            for test_data in self.test_dl:
                inputs, labels = test_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)


                preds = self.model(inputs) #forward pass
                
                _, prediction = torch.max(preds,dim=1)
                total += labels.size(0) # number of rows = num of samples
                correct += (prediction == labels).sum().item() 

        val_acc = correct/total
        
        return val_acc
                
    def _estimate_layer_flops(self, amount_pruned, pruned_prev_layer):
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
        C_in = conv_layer.in_channels - pruned_prev_layer
        C_out = conv_layer.out_channels - amount_pruned
        groups = conv_layer.groups

        filter_steps_h = (input_h + 2*pad_h - kernel_h)/stride_h + 1  
        filter_steps_w = (input_w + 2*pad_w - kernel_w)/stride_w + 1  

        layer_flops = C_out * (C_in/groups) * kernel_h*kernel_w \
                            * filter_steps_h * filter_steps_w
       
        return layer_flops

    def _calculate_reward(self, amount_pruned, pruned_prev_layer): 
        ''' Performs the ops to get reward 
            of action of current layer'''

        # train for M epochs
        self._train_model(num_epochs=1)
        #logging.info("Training skipped")

        # test
        acc = self._evaluate_model() # acc is in {0,1}
        logging.info('Validation Accuracy: {:.2f}%'.format(acc*100))

        # get flops 
        flops_orig = self._estimate_layer_flops(0, pruned_prev_layer)
        flops_remain = self._estimate_layer_flops(amount_pruned,\
                                                    pruned_prev_layer)

        flops_ratio = float(float(flops_remain) / float(flops_orig))
        # get reward as func of acc and flops
        #amount_pruned = amount_pruned.type(torch.float)
        #total_filters = torch.tensor(total_filters, dtype = torch.float)
        #reward = -(1-acc)*(flops_ratio) 
        reward = -(1-acc)*np.log(flops_remain)#*(total_filters/amount_pruned)#np.log(flops)
        logging.info("%Flops: {}".format(flops_ratio))
        logging.info("Reward: {}".format(reward))

        return reward, acc, flops_orig, flops_ratio

    def maskbuildbias(self, indices, num_filters):
        ''' Builds a mask for the bias of the layer to be pruned. 
            Sub function of prune_layer.

            Args:
                indices = list of indices to be pruned. 
                          i.e. [0,1,1,0,0,1,1,0,1,0...]
        '''
        bias_mask = copy.copy(indices[0, :num_filters])

        #mask0 = torch.zeros(1).to(self.device)
        #mask1 = torch.ones(1).to(self.device)
        #indices = indices[0]
        #indices = indices[:num_filters]
        #for i, val in enumerate(indices):
        #    if i == 0:
        #        if val == 0:
        #            finalmask = mask0
        #        else:
        #            finalmask = mask1
        #    else:
        #        if val == 0:
        #            finalmask = torch.cat((finalmask, mask0),0)
        #        else:
        #            finalmask = torch.cat((finalmask, mask1),0)
        #logging.info("B mask size: {}".format(finalmask.size()))
        #logging.info("B mask diff: {}".format((finalmask-bias_mask).sum()))
        #return finalmask
        return bias_mask

    def maskbuildweight(self, indices, kernel_size, num_filters):
        '''
            Builds a mask for the weights of the layer to be pruned. 
            Sub function of prune_layer

            Args:
                indices = list of indices to be pruned. 
                          i.e. [0,1,1,0,0,1,1,0,1,0...]
                kernel_size = size of the kernel of the filters, 
                             assumed to be n*n (i.e. square filters)
        '''
        weight_mask = copy.copy(indices[0,:num_filters]).view(-1,1,1)
        weight_mask = weight_mask.expand(-1,kernel_size,kernel_size)

        #mask0 = torch.zeros((1,kernel_size, kernel_size))
        #mask1 = torch.ones((1,kernel_size, kernel_size))
        #
        ##Workaround for indices [[]]
        #indices = indices[0]
        #indices = indices[:num_filters]
        #for i, val in enumerate(indices):
        #    #initialize the mask
        #    if i == 0:
        #        if val == 0:
        #            finalmask = mask0
        #        else:
        #            finalmask = mask1
        #    #concatenate the masks
        #    else:
        #        if val == 0:
        #            finalmask = torch.cat((finalmask, mask0),0)
        #        else:
        #            finalmask = torch.cat((finalmask, mask1),0)

        #logging.info("W mask size: {}".format(finalmask.size()))
        #logging.info("W mask diff: {}".format((weight_mask-finalmask).sum()))
        #return finalmask
        return weight_mask
    
    def maskbuildweight2(self, prev_indices, kernel1, kernel2, num_filters_prev):
        ''' Builds a mask for the weights of the next layer. 
            Sub function of prune_layer
            Necessity from the previous layer's having less output feature maps
            Args:
                indices = list of indices to be pruned. 
                          i.e. [0,1,1,0,0,1,1,0,1,0...]
                kernel1, kernel2 = n*m kernel size.
        '''
        next_weight_mask = copy.copy(prev_indices[0,:num_filters_prev])
        next_weight_mask = next_weight_mask.view(-1,1,1)
        next_weight_mask = next_weight_mask.expand(-1,kernel1,kernel2)

        #you build the mask based on the 
        #previous layer's indices but stack it according to this layer's indices
        #mask0 = torch.zeros((1,kernel1, kernel2))
        #mask1 = torch.ones((1,kernel1, kernel2))
       
        ##Workaround for indices [[]]
        #prev_indices = prev_indices[0]
        #prev_indices = prev_indices[:num_filters_prev]

        ##build on a per channel basis
        #for i, val in enumerate(prev_indices):
        #    #initialize the mask
        #    if i == 0:
        #        if val == 0:
        #            finalmask = mask0
        #        else:
        #            finalmask = mask1
        #    #concatenate the masks
        #    else:
        #        if val == 0:
        #            finalmask = torch.cat((finalmask, mask0),0)
        #        else:
        #            finalmask = torch.cat((finalmask, mask1),0)

        ## stack on a per filter basis
        ## meaning change torch.stack dimension to 0
        ## but masktuple is still multiplied by number of filters since it 
        ## is about hte current layers filters
        #logging.info("W2 mask size: {}".format(finalmask.size()))
        #logging.info("W2 mask diff: {}".format((next_weight_mask-finalmask).sum()))
        #return finalmask
        return next_weight_mask

    def prune_layer(self, indices):
        ''' Added filter pruning function 
            Args: 
                layer_number = the layer to be pruned starts from 0 to 4 
                indices = tensor of indices to be pruned 
                          i.e. [0,0,0,0,1,1,1,0,1,1,1,0...]
                self.model = network to be pruned 
        '''

        iter_ = 0
        iterbn = 0
        amt_pruned = 0 # to be assigned in mask_per_channel condition
        
        named_children = self.model.named_children()
        for idx, module in enumerate(named_children): 
            if self.layer_to_process in module[0]:
                layer_number = idx
                conv_layer = module
                _, next_conv_layer = next(named_children)
                break

        #iterate through all the parameters of the network
        for layer in self.model.children():
            #hardcode to find the last conv layer
            #this is not needed for now as long as you set the 
            #last batchnorm layer to 0
            #proven empirically on the 3 layer 3*3 network

            #if convolutional layer
            if type(layer) == nn.Conv2d:
                #if not the layer to be pruned, skip the below
                if iter_ != layer_number and iter_ != layer_number + 1:
                    iter_ = iter_ + 1
                    continue

                #enumerate through all the contents of the layer.
                #use a different mask depending on whether this is current or next
                #there should be no bias change if this is for the next
                #for a conv layer thats: 1. weights 2. biases
                for i, param in enumerate(layer.parameters()):
                    #use the param size to determine if weight or bias

                    size = param.size()
                    #if bias, then make the mask for current only
                    if (len(size) == 1):

                        #it has to be stacked conditions so that it doesn't go to the "else"
                        if iter_ == layer_number:
                            # print("a",a[0])

                            #multiply param.data with a mask of zeros up to the 
                            #desired index, all else are filled with ones
                            #logging.info('Build bias mask')
                            mask = self.maskbuildbias(indices, size[0])
                            param.data = torch.mul(param.data,mask)

                            #print("Built bias mask")
                            #iterate the cnn layer counter
                    #if weights
                    else:
                        #mask per channel
                        if iter_ == layer_number:
                            #size[2] == kernel size size[0] == num filters
                            #logging.info('Build filter mask')
                            mask = self.maskbuildweight(indices, size[2], size[0])
                            masktuple = ((mask),)*size[1]
                            finalmask = torch.stack((masktuple),1)
                            # get prune amount to return to caller
                            amt_pruned = indices[0,:size[0]].sum()
                            
                        elif iter_ == layer_number+1:
                            #size[2]&[3] == kernel_size size[1] = prev_num_filters
                            #logging.info('Build next filter mask')
                            mask = self.maskbuildweight2(indices, size[2], size[3], size[1])
                            masktuple = ((mask),)*size[0]
                            finalmask = torch.stack((masktuple),0)

                        #print(param.data.shape, finalmask.shape)
                        param.data = torch.mul(param.data,
                                               finalmask.to(self.device))
                        # print(param.data,"after")
                iter_ = iter_ + 1    
            if type(layer) == nn.BatchNorm2d:
                for i , param in enumerate(layer.parameters()):
                    if iterbn == layer_number:
                        size = param.size()

                        #multiply param.data with a mask of zeros up to 
                        #the desired index, all else are filled with ones
                        #logging.info('Build batchnorm mask')
                        mask = self.maskbuildbias(indices, size[0])

                        # print(param.data)
                        param.data = torch.mul(param.data,mask)
                iterbn = iterbn + 1

        return amt_pruned

    def reset_to_k(self):
        ''' resets CNN to partially trained net w/ full params'''
        #self.model.load_state_dict(self.init_full_weights)

        self.model = copy.deepcopy(torch.load(os.getcwd() + \
                                                '/partially_trained_3.pt',
                                                map_location = self.device))

    #def load_trained(self):
    #    '''loads a trained model'''
    #    ###Alternate way of loading a state dict.
    #    ###Dependent on how it was saved.
    #    self.model = self.trained_weights
    
        
    #def step(self, action):
        #''' Run one timestep '''

        # prune_layer(action) # perform action via MARCUS' function

        #reward = self._calculate_reward()
        #new_state = self._get_state()
        
        #if self.xp_count == self.expis:
        #    done = True
        #    self.xp_count = 0
        #else:
        #    done = False
        #    self.xp_count +=1

        #return new_state, reward#, done


