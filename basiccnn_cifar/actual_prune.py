import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import os
import copy
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
from models_to_prune import *
from utilities import *
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
###Argument parsing
parser = argparse.ArgumentParser(description='Arguments for masker')
parser.add_argument('--criterion', type=str, default='mag',
                    help='criterion to use')
parser.add_argument('--foldername', type=str, default = 'trash',
                    help='folder to store masked networks in')
parser.add_argument('--ratio_prune', type=float, default = 0.5,
                    help='amount to prune')
parser.add_argument('--inv_flag', action = 'store_true', default = False,
                    help='invert criterion if True')

args = parser.parse_args()
def actual_prune(path_from, path_to):
    env = PruningEnv()
    env.model.load_state_dict(torch.load(os.getcwd() + '/' + path_from)['state_dict'])
    env.optimizer.load_state_dict(torch.load(os.getcwd() + '/' + path_from)['optim']) 
    print("accuracy is ", env._evaluate_model())
    layer_mask = [] #list
    num_per_layer = []
    for module in env.model.modules():
        #for conv2d obtain the filters to be kept.
        if isinstance(module, nn.BatchNorm2d):
            weight_copy = module.weight.data.clone()
            filter_mask = weight_copy.gt(0.0).float()
            #print(filter_mask)
            #print(weight_copy)
            layer_mask.append(filter_mask)



    for i, item in enumerate(layer_mask):
        ###Have to use.item for singular element tensors to extract the element
        ###Have to use int()
        num_per_layer.append(int(item.sum().item()))
        
    print(num_per_layer)
    newmodel = RandSubnet(filter_counts = num_per_layer)
    newmodel.build()

    ####Pruning the parameters
    conv_iter = 0
    bn_iter = 0
    for [moduleold, modulenew] in zip(env.model.modules() , newmodel.model.modules()):

        if isinstance(moduleold, nn.Dropout):
            print("Skipping dropout")
            continue
        
        #Replace the Conv2d's of the layer
        if isinstance(moduleold, nn.Conv2d):
            indices = (np.argwhere(layer_mask[conv_iter].cpu())).squeeze(0)
            print(indices.numel())
            print(indices, "indices at" ,conv_iter)
            if conv_iter > 0:
                indices_prev = (np.argwhere(layer_mask[conv_iter-1].cpu())).squeeze(0)
            
            
            weight_copy = moduleold.weight.data.clone()
            
            #Squeeze the dimention you give the indices too.

            pruned_weight = weight_copy[indices,:,:,:]
            if conv_iter > 0:
                pruned_weight = pruned_weight[:,indices_prev,:,:]
            

            modulenew.weight.data = pruned_weight
            ##If the replacement is not the same size the value does not change
            ##BE CAREFUL
            

            #Biases seem to transfer correctly
            bias_copy = moduleold.bias.data.clone()
            pruned_bias = bias_copy[indices]
            if indices.numel() > 1:
                pruned_bias = pruned_bias.squeeze(0)
            print(pruned_bias.shape,"pbshape")
            modulenew.bias.data = pruned_bias

            conv_iter += 1
            print(iter)
            if (modulenew.weight == pruned_weight).all():
                print("IDENTICAL WEIGHTS")
            if (modulenew.bias == pruned_bias).all():
                print("IDENTICAL BIAS")
                
        #Replace Batchnorms
        if isinstance(moduleold, nn.BatchNorm2d):
            indices = (np.argwhere(layer_mask[bn_iter].cpu())).squeeze(0)
            if conv_iter > 0:
                indices_prev = (np.argwhere(layer_mask[bn_iter-1].cpu())).squeeze(0)
            
            modulenew.weight.data = moduleold.weight.data[indices].clone()
            modulenew.bias.data = moduleold.bias.data[indices].clone()
            modulenew.running_mean.data = moduleold.running_mean.data[indices].clone()
            modulenew.running_var.data = moduleold.running_var.data[indices].clone()
            bn_iter += 1
         
        if isinstance(moduleold, nn.Linear):
            indices = (np.argwhere(layer_mask[-1].cpu())).squeeze(0)
            print(moduleold.weight.data.shape)
            print(moduleold.bias.data.shape)
            #Chunks is the feature map size
            chunks = 16
            newindices = torch.zeros(moduleold.weight.data.shape[1])
            for i in range(chunks):
                newindices[indices.numpy()*chunks + i] = 1
            
            newindices = (np.argwhere(newindices)).squeeze(0)
            print(newindices)
            print(modulenew.weight.data.shape)
            print(moduleold.weight.data.shape)
            print(modulenew.bias.data.shape)
            print(moduleold.bias.data.shape)
            modulenew.weight.data = moduleold.weight.data[:,newindices]
            modulenew.bias.data = moduleold.bias.data
    model_dicts = {'state_dict': newmodel.model.state_dict(),
                    'filters_per_layer':num_per_layer}
    PATH = os.getcwd() + '/' + path_to
    torch.save(model_dicts, PATH)
    print(newmodel.evaluate(env.test_dl))
    print(env._evaluate_model())
    print("successfulsave")

actual_prune('masked_may_12/SA0.7_20_rand.pth', 'pruned/SA0.7_20_rand_pruned.pth')
exit()


crit_list = ['bn_abs', 'grad', 'mag','rand']
inv_flag = args.inv_flag
folder = args.foldername
for criterion in crit_list:    
    if inv_flag == True:
        actual_prune(str(folder) + '/masked_inv/' + str(criterion) + '_inv_masked_2.pth', str(folder) + '/pruned_inv/' + str(criterion) + '_inv_pruned_2.pth')
    
    else:
        actual_prune(str(folder) + '/masked/' + str(criterion) + '_masked_2.pth', str(folder) + '/pruned/' + str(criterion) + '_pruned_2.pth')




exit()
