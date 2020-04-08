#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import copy

import torch
import torchvision.transforms as transforms
import torchvision.datasets as ds
import torch.utils.data as data
from collections import OrderedDict

import models_to_prune 
#from temp_files.state_rep_autoencoder import autoencoder
#from utilities import estimate_layer_flops

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
                 state_size = 960+960+960):

        # assign dataset
        self.dataset = dataset
        self.train_dl, self.test_dl = self.get_dataloaders()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.info("Device {}".format(self.device))

        # build chosen model to prune
        self.model_type = model_type
        self.model = self._build_model_to_prune().to(self.device)

        #logging.info("Starting Pre-Training")
        # set training parameters
        self.loss_func = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0008)
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4)

        #self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1
        #self._train_model(num_epochs=1)
        #self.init_full_weights = copy.deepcopy(self.model.state_dict()) 
                                    # initially, self.model has full-params
                                    # used in reset_to_k()

        # state
        #self.layers_to_prune = [name for name,_ in self.model.named_modules() 
        #                        if 'conv' in name]
        self.layers_to_prune = ['conv1','conv2','conv3','conv4']
        logging.info(self.layers_to_prune)
        self.layer = None # Layer to process, 
                                     # str name, is usr-identified 
        self.layer_prune_amounts = OrderedDict()
        self.layer_flops = OrderedDict()
        self.full_model_flops = 0 # will be calculated in reset_to_k()
        #self.amount_pruned_dict = {} # {layer_name : amount it was pruned}
        #self.amount_pruned = 0  # On layer_to_process, updated in prune_layer()
        #self.prev_amount_pruned = 0 # previous layer's amt pruned
        self.prev_out_feat = [0,0]  # List of [h,w] of prev layer's featmap 

        self.state_size = state_size 
        self.max_layer_idx = 4 #TODO: can be derived from self.model

    def get_dataloaders(self):
        ''' imports the chosen dataset '''

        if self.dataset.lower() == 'cifar10':
            # copied from marcus' rat5, lines 17:29
            cifar10_trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5))])
            train_transform = transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
            
            test_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
            
            train = ds.CIFAR10(root = os.getcwd(),
                               train = True,
                               download = True,
                               transform = cifar10_trans)

            train_loader = data.DataLoader(train,
                                           batch_size = 256,
                                           shuffle = False,
                                           num_workers = 0, pin_memory = True)
            
            test = ds.CIFAR10(root = os.getcwd(),
                              train = False,
                              download = True,
                              transform = cifar10_trans)
            
            test_loader = data.DataLoader(test,
                                          batch_size = 256, # testing use less 
                                                           # memory, can afford 
                                                           # larger batch_size
                                          shuffle = False,
                                          num_workers = 0, pin_memory = True)

            return train_loader, test_loader
        elif self.dataset.lower() == 'mnist':
            print("Using mnist")
            mnist_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                         ])
            train = ds.MNIST(root = os.getcwd(),
                               train = True,
                               download = True,
                               transform = mnist_transform)

            train_loader = data.DataLoader(train,
                                           batch_size = 256,
                                           shuffle = False,
                                           num_workers = 0, pin_memory = True)
            
            test = ds.MNIST(root = os.getcwd(),
                              train = False,
                              download = True,
                              transform = mnist_transform)
            
            test_loader = data.DataLoader(test,
                                          batch_size = 256, # testing use less 
                                                           # memory, can afford 
                                                           # larger batch_size
                                          shuffle = False,
                                          num_workers = 0, pin_memory = True)
        # TODO: use proper exception handling
            return train_loader, test_loader
        print('dataset not available') 
        return -1

    def _build_model_to_prune(self): 
        ''' Builds the model to compress '''

        if self.model_type.lower() == 'basic' :

            return models_to_prune.BasicCNN()

        else:
            print('model not available') #TODO: use proper handling
            return -1
    
    def _estimate_layer_flops(self):
        ''' Helper tool for calculate_network_flops
            and _calculate_reward(),
            estimates single conv layer flops

            Important: Assumes calculation is always done in order,
                        from first to last conv layer'''

        for name, module in self.model.named_modules():
            if self.layer in name:
                conv_layer = module
                #logging.info('conv name: {}'.format(name))
                break

        if '1' in self.layer: # first conv layer
            for inputs, _ in self.train_dl:
                self.prev_out_feat = inputs.size()[2:] # input is data
                prev_amount_pruned = 0 # no prev layer was pruned
                break 
        else: # get previous
            idx = list(self.layer_prune_amounts.keys()).index(self.layer) - 1 
            prev_amount_pruned = list(self.layer_prune_amounts.values())[idx]

        amount_pruned = self.layer_prune_amounts[self.layer]

        input_h = self.prev_out_feat[0]
        input_w = self.prev_out_feat[1]
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]
        pad_h = conv_layer.padding[0]
        pad_w = conv_layer.padding[1]
        stride_h = conv_layer.stride[0]
        stride_w = conv_layer.stride[1]
        C_in = conv_layer.in_channels - prev_amount_pruned
        C_out = conv_layer.out_channels 
        groups = conv_layer.groups

        out_h = (input_h + 2*pad_h - kernel_h)//stride_h + 1  
        out_w = (input_w + 2*pad_w - kernel_w)//stride_w + 1  

        # ff assumes that flops estimation is always done in order
        self.prev_out_feat = [out_h, out_w]

        original_layer_flops = C_out * (C_in/groups) * kernel_h*kernel_w \
                            * out_h * out_w
        pruned_layer_flops = (C_out - amount_pruned) * (C_in/groups) \
                            * kernel_h*kernel_w * out_h * out_w

        return original_layer_flops, pruned_layer_flops

    def _calculate_network_flops(self):
        ''' Helper function for get_state '''
        
        #total_network_flops = sum(layer_flops.values())
        layer_idx = list(self.layer_flops.keys()).index(self.layer)
        #logging.info('layer_idx: {}'.format(layer_idx))
        reduced_layer_flops = sum(list(self.layer_flops.values())[:layer_idx])
        current_layer_flops = self.layer_flops[self.layer]
        rest_layer_flops = sum(list(self.layer_flops.values())[layer_idx+1:])

        return reduced_layer_flops, current_layer_flops, rest_layer_flops

    def get_state(self, include_grads=False,
                        include_flops=False): 
        ''' Gets the layer/state '''
        
        # get conv layer 
        # may keep an external pth file for original model
        for name, module in self.model.named_modules(): # this model changes
            if self.layer in name:
                conv_layer = module
                layer_idx = int(name[-1]) # to be used in state_rep
                break

        # State element 1
        layer_idx = torch.tensor([layer_idx/self.max_layer_idx]) # (0,1]
        #logging.info('layer_index: {}'.format(layer_idx))

        # State element 2
        # copy params
        filter_weights = torch.abs(conv_layer.weight.data.clone())
        #TODO: what about the bias tensor ?
        pooled_weights = torch.squeeze(F.avg_pool2d(filter_weights,
                                                   filter_weights.size()[-1]))
        pooled_weights = pooled_weights*1000 # scale up magnitudes for encoder
        
        pooled_weights_mean = pooled_weights.mean(axis = 1)
        padded_weights = torch.zeros([512])
        padded_weights[0:pooled_weights_mean.shape[0]] = pooled_weights_mean
        padded_weights -= padded_weights.min()
        padded_weights /= padded_weights.max() # squish to [0,1]

        # Concat two core elements
        state_rep = torch.cat((layer_idx,padded_weights),0)
                    # addn'ls to be concat thru ff conditions

        # State element 3
        if include_grads: 
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4)

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
                        if key == self.layer + '.weight':
                            grads = torch.abs(var.grad.clone())
                            pooled_grad = torch.squeeze(F.avg_pool2d(var.grad,
                                                        var.grad.size()[-1]))
                            pooled_grad = pooled_grad*1000
                            grad_rep = pooled_grad.mean(axis = 1)
                            grad_rep -= grad_rep.min()
                            grad_rep /= grad_rep.max()
                            break
                    break

            grad_rep_padded = torch.zeros([512])
            grad_rep_padded[0:grad_rep.shape[0]] = grad_rep
            state_rep = torch.cat((state_rep,grad_rep_padded),0)

        # State element 4
        if include_flops:
            reduced, current, rest = self._calculate_network_flops()
            # normalized wrt original full flops
            reduced = torch.tensor([reduced/self.full_model_flops])
            current = torch.tensor([current/self.full_model_flops])
            rest = torch.tensor([rest/self.full_model_flops])
            #logging.info('red {} cur {} res {}'.format(reduced, 
            #                            current,rest))
            
            state_rep = torch.cat((state_rep,reduced,current,rest),0)


        return state_rep
        
    def get_global_state_rep(self):
        #Go to all the layers
        for layer in self.layers_to_prune:
        
            #Get the weights of each layer
            for name, param in self.model.named_parameters():
                if layer in name and 'weight' in name:
                    # State element 2
                    # copy params
                    filter_weights = torch.abs(param.data.clone())
                    #TODO: what about the bias tensor ?
                    pooled_weights = torch.squeeze(F.avg_pool2d(filter_weights,
                                                               filter_weights.size()[-1]))
                    try:  #all non first layers have to have their axis specified for mnist
                        pooled_weights_mean = pooled_weights.mean(axis = 1)
                    except: #except when first layer
                        pooled_weights_mean = pooled_weights
                    pooled_weights_mean -= pooled_weights_mean.min()
                    pooled_weights_mean /= pooled_weights_mean.max() # squish to [0,1]

                    # Concat two core elements
                    try:
                        state_rep = torch.cat((state_rep,pooled_weights_mean),0)

                    except:
                        state_rep = pooled_weights_mean
                                # addn'ls to be concat thru ff conditions
        grad_rep = self.get_grads()
        state_rep = torch.cat((state_rep, grad_rep),0)
        bn_rep = self.get_BNs()
        bn_rep = torch.abs(bn_rep)
        state_rep = torch.cat((state_rep, bn_rep),0)
        print(state_rep.shape)
        return state_rep
    def get_BNs(self):
        
        for key,var in self.model.named_parameters():
            if 'bn' in key and 'weight' in key:
                try: #concatenate
                    bn_rep = torch.cat((bn_rep, var),0)
                except: #initialize if not initialized yet
                    bn_rep = var
        
        bn_rep -= bn_rep.min()
        bn_rep /= bn_rep.max()
        return bn_rep
        
    def get_grads(self):
        loss_func = nn.CrossEntropyLoss()
        
        #Fake train. To obtain gradients
        for idx, train_data in enumerate(self.test_dl):
            inputs, labels = train_data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # TODO: transfer to device?
            
            self.optimizer.zero_grad()

            # forward
            preds = self.model(inputs) # forward pass
            loss = loss_func(preds,labels) # compute loss
            # backward
            loss.backward()  # compute grads
            for layer in self.layers_to_prune:
                for key, var in self.model.named_parameters():
                    if key == layer + '.weight':
                        abs_grad = torch.abs(var.grad.clone())
                        pooled_grad = torch.squeeze(F.avg_pool2d(abs_grad,
                                                    abs_grad.size()[-1]))
                        try:
                        
                            grad_rep = pooled_grad.mean(axis = 1)
                        except:
                            grad_rep = pooled_grad
                            
                        grad_rep -= grad_rep.min()
                        grad_rep /= grad_rep.max()
                        
                        try:
                            total_grad_rep = torch.cat((total_grad_rep,grad_rep),0)

                        except:
                            total_grad_rep = grad_rep
                        break #break the named_parameters loop only if matched
            break #break the training loop
            

        # grad_rep_padded = torch.zeros([512])
        # grad_rep_padded[0:grad_rep.shape[0]] = grad_rep
        # state_rep = torch.cat((state_rep,grad_rep_padded),0)
        return total_grad_rep
        
    def updateBN(self): # jeff: default model is the global model
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1
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
                # print("Loss", loss)
                # backward
                loss.backward()  # compute grads
                self.updateBN()
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

        val_acc = float(correct/total)
        val_acc = torch.tensor(val_acc, requires_grad = True)
        return val_acc#, correct, total
                
    def _calculate_reward(self, total_filters, amount_pruned): 
        ''' Performs the ops to get reward 
            of action of current layer'''

        # train for M epochs
        #self._train_model(num_epochs=1)
        logging.info("Training skipped")

        # test
        acc = self._evaluate_model() # acc is in {0,1}
        logging.info('Validation Accuracy: {:.2f}%'.format(acc*100))

        # get flops 
        flops_orig, flops_remain = self._estimate_layer_flops()
        self.layer_flops[self.layer] = flops_remain
        
        #temporary delete after single layer experiments
        flops_remain = 1
        flops_orig = 1
        
        
        flops_ratio = float(float(flops_remain) / float(flops_orig))
        # get reward as func of acc and flops
        reward = acc #+ float(amount_pruned)/float(total_filters)
        print(reward, "reward")
        print(amount_pruned, "pruned")
        print(total_filters, "filters")
        # logging.info("%Layer Flops: {}".format(flops_ratio))
        logging.info("Reward: {}".format(reward))
        # logging.info("Flops Remain: {}".format(np.log(flops_remain)))
        logging.info("Amount_pruned: {}".format(amount_pruned))

        return reward, acc, flops_orig, flops_ratio
    
    def param_to_mask(self, ratio, method):
        if method == 'weight_norm':
            for name, module in self.model.named_modules(): # this model changes
                if self.layer in name:
                    conv_layer = module
                    layer_idx = int(name[-1]) # to be used in state_rep
                    break
            filter_weights = torch.abs(conv_layer.weight.data.clone())
            num_filters = filter_weights.shape[0]
            pooled_weights = torch.squeeze(F.avg_pool2d(filter_weights,
                                                       filter_weights.size()[-1]))
            # print(pooled_weights.shape)        
            try:
                pooled_weights_mean = pooled_weights.mean(axis = 1)
            except:
                pooled_weights_mean = pooled_weights
            # print("Pooled weights\n",pooled_weights_mean)
            mask = torch.ones(pooled_weights_mean.shape[0])
            mag_rank = torch.topk(pooled_weights_mean,int(num_filters*ratio),largest = False)
            mask[mag_rank[1]] = 0
            padded_weights = torch.zeros([512])
            # print(mask)
        elif method == 'weight_norm_and_var':
            for name, module in self.model.named_modules(): # this model changes
                if self.layer in name:
                    conv_layer = module
                    layer_idx = int(name[-1]) # to be used in state_rep
                    break
            filter_weights = torch.abs(conv_layer.weight.data.clone())
            num_filters = filter_weights.shape[0]
            pooled_weights = torch.squeeze(F.avg_pool2d(filter_weights,
                                                       filter_weights.size()[-1]))
            # print(pooled_weights.shape)        
            pooled_weights_mean = pooled_weights.mean(axis = 1)
            # print("Pooled weights\n",pooled_weights_mean)
            mask = torch.ones(pooled_weights_mean.shape[0])
            mag_rank = torch.topk(pooled_weights_mean,int(num_filters*ratio),largest = False)
            
            #Variance across the channels/kernels?, and not the individual kernel elements
            pooled_weights_var = torch.var(pooled_weights, dim = 1)
            
            mask[mag_rank[1]] = 0
            padded_weights = torch.zeros([512])
        elif method == 'grad_norm':
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
                        if key == self.layer + '.weight':
                            pooled_grad = torch.squeeze(F.avg_pool2d(torch.abs(var.grad),
                                                        var.grad.size()[-1]))
                            grad_rep = pooled_grad.mean(axis = 1)
                            num_filters = grad_rep.shape[0]
                            print("NUM", num_filters)
                            break
                    break
            mask = torch.ones(grad_rep.shape[0])
            grad_rank = torch.topk(grad_rep, int(num_filters*ratio), largest = False)
            mask[grad_rank[1]] = 0
        else:
            print("No mask")
        return mask
    def maskbuildbias(self, indices, num_filters):
        ''' Builds a mask for the bias of the layer to be pruned. 
            Sub function of prune_layer.

            Args:
                indices = list of indices to be pruned. 
                          i.e. [0,1,1,0,0,1,1,0,1,0...]
        '''
        bias_mask = copy.copy(indices[0, :num_filters])
        bias_mask = bias_mask.type(torch.FloatTensor)

        return bias_mask.to(self.device)

    def maskbuildweight(self, indices, kernel1, kernel2, num_filters):
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
        weight_mask = weight_mask.expand(-1,kernel1,kernel2)
        weight_mask = weight_mask.type(torch.FloatTensor)

        return weight_mask.to(self.device)
    
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
        next_weight_mask = next_weight_mask.type(torch.FloatTensor)

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
        return next_weight_mask.to(self.device)

    def prune_layer(self, indices):
        ''' Added filter pruning function 
            Args: 
                indices = tensor of indices to be pruned 
                          i.e. [0,0,0,0,1,1,1,0,1,1,1,0...]
        '''

        iter_ = 0
        iterbn = 0
        amt_pruned = 0 # to be assigned in mask_per_channel condition
        total_filters = 0 # same as above,filter # before pruning the layer
        
        named_children = self.model.named_children()
        for idx, module in enumerate(named_children): 
            if self.layer in module[0]:
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
                            mask = self.maskbuildweight(indices,size[2],\
                                                            size[3],size[0])
                            masktuple = ((mask),)*size[1]
                            finalmask = torch.stack((masktuple),1)
                            # get prune amount to return to caller
                            amt_pruned = (size[0] -
                                          indices[0,:size[0]].sum()).item()
                            
                            total_filters = size[0]

                            # update env pruning record
                            #self.prev_amount_pruned = self.amount_pruned
                            self.layer_prune_amounts[self.layer] = amt_pruned
                            
                        elif iter_ == layer_number+1:
                            #size[2]&[3] == kernel_size size[1] = prev_num_filters
                            #logging.info('Build next filter mask')
                            mask = self.maskbuildweight2(indices, size[2],\
                                                            size[3], size[1])
                            masktuple = ((mask),)*size[0]
                            finalmask = torch.stack((masktuple),0)

                        param.data = torch.mul(param.data,finalmask)
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

        return total_filters, amt_pruned 



    def reset_to_k(self):
        ''' resets CNN to partially trained net w/ full params'''
        #self.model.load_state_dict(self.init_full_weights)

        #self.model = copy.deepcopy(torch.load(os.getcwd() + \
        #                                        '/partially_trained_3.pt',
        #                                        map_location = self.device))
        self.model.load_state_dict(torch.load(os.getcwd() + \
                                                '/april_experiments_withBN_epoch_5.pth')['state_dict'])
        self.optimizer.load_state_dict(torch.load(os.getcwd() + \
                                    '/april_experiments_withBN_epoch_5.pth')['optim'])
        # initialize starting layer to process
        self.layer = self.layers_to_prune[0]
        # initialize prune amounts to zer
        self.layer_prune_amounts = OrderedDict(zip(self.layers_to_prune,\
                                                [0]*len(self.layers_to_prune)))
        # get layer_flops dict 
        layer_to_process = self.layer # preserve
        for name in self.layers_to_prune:
                self.layer = name
                orig_flops, flops_remain = self._estimate_layer_flops() 
                                #TODO: might be better to explicitly pass layer
                                # name to estimate_flops()
                self.layer_flops[self.layer] = flops_remain
        self.layer = layer_to_process
        # save total network flops
        self.full_model_flops = sum(self.layer_flops.values())
        
    def load_trained(self):
        '''loads a trained model'''
        ###Alternate way of loading a state dict.
        ###Dependent on how it was saved.
        self.model = copy.deepcopy(torch.load(os.getcwd() + \
                                                '/best_snapshot_78.pt',
                                                map_location = self.device))
        # initialize starting layer to process
        self.layer = self.layers_to_prune[0]
        # initialize prune amounts to zer
        self.layer_prune_amounts = OrderedDict(zip(self.layers_to_prune,\
                                                [0]*len(self.layers_to_prune)))
        # get layer_flops dict 
        layer_to_process = self.layer # preserve
        for name in self.layers_to_prune:
                self.layer = name
                orig_flops, flops_remain = self._estimate_layer_flops() 
                                #TODO: might be better to explicitly pass layer
                                # name to estimate_flops()
                self.layer_flops[self.layer] = flops_remain
        self.layer = layer_to_process
        # save total network flops
        self.full_model_flops = sum(self.layer_flops.values())
        
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


