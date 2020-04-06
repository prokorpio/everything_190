#main_retrain_masked

import torch
import time
import os
import copy
import math
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
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


# Basemark is 5381
trialnum = 2

env = PruningEnv()
env.reset_to_k()

#### Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        # print(name)
        # print(param.shape[0])
        size_of_layer.append(param.shape[0])
        
ratio_prune = args.ratio_prune
limit_rem = 0.02
criterion = args.criterion
inv_flag = args.inv_flag
#Setting the criterion
if criterion == 'mag':
    random_mask = env.get_global_state_rep()
    random_mask = random_mask[0:960]
    print("Using mag")
elif criterion == 'bn':
    random_mask = env.get_BNs()
    print("Using bn")
elif criterion == 'grad':
    random_mask = env.get_grads()
    print("Using grad")
elif criterion == 'rand':
    random_mask = torch.rand((960))
    print("Using rand")
elif criterion == 'bn_abs':
    random_mask = env.get_BNs()
    random_mask = torch.abs(random_mask)
    print("Using bn_abs")
else:
    print("No criterion was used")
    exit()

idx = 0
total_pruned = 0
for i in range(len(size_of_layer)):
    
    #choose the layer
    env.layer = env.layers_to_prune[i]
    #Get current action for the layer as well as the mask
    #if local
    layer_tempmask = random_mask[idx:idx+size_of_layer[i]].clone()
    if inv_flag == True:
        layer_rank = torch.topk(layer_tempmask, int(size_of_layer[i]*ratio_prune), largest = True)
        # print("Using inverted criterion")
    else:
        layer_rank = torch.topk(layer_tempmask, int(size_of_layer[i]*ratio_prune), largest = False)
        # print("Using non inverted criterion")
    layer_mask = torch.ones(size_of_layer[i])
    layer_mask[layer_rank[1]] = 0
    
    
    #if global
    # layer_action  = global_randmask[idx:idx+size_of_layer[i]].clone()
    # layer_values = random_mask[idx:idx+size_of_layer[i]].clone()
    # layer_pruned = global_randmask[idx:idx+size_of_layer[i]].sum()
    
    
    #Renaming local to fit with global scheme convention
    layer_action = layer_mask
    layer_values = layer_tempmask
    layer_pruned = layer_action.sum()

    # print("Proposed_layer_kept",layer_pruned)
    amt_unprune = math.ceil(limit_rem*float(size_of_layer[i]))
    if layer_pruned < int(amt_unprune):
        if inv_flag == True:
            unprune = torch.topk(layer_values,int(amt_unprune),largest = False)
        else:
            unprune = torch.topk(layer_values,int(amt_unprune),largest = True)
            
        layer_action[unprune[1]] = True
        #global_randmask[idx:idx+size_of_layer[i]] = layer_action
        print("Unpruned some", amt_unprune)

    else:
        pass
        

    # print("Amount kept is", layer_action.sum())
    total_pruned += size_of_layer[i] - layer_mask.sum()
    layer_action = torch.unsqueeze(layer_action,0)
    #layer_action = torch.unsqueeze(global_randmask[idx:idx+size_of_layer[i]],0)    
    filters_counted, pruned_counted = env.prune_layer(layer_action)           
    idx += size_of_layer[i]
    
    try: #concat
        total_action = torch.cat((total_action,layer_action),0)
    except: #else initialize
        total_action = layer_action

val_acc = env._evaluate_model()
print(val_acc)


if inv_flag == True:
    PATH = os.getcwd() + '/' + str(args.foldername) + '/masked_inv/' + str(criterion) + '_inv_masked_'+str(trialnum)+'.pth'

else:
    PATH = os.getcwd() + '/' + str(args.foldername) + '/masked/' + str(criterion) + '_masked_'+str(trialnum)+'.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict(),
        'kept_indices' : torch.where(layer_action == 1)}
torch.save(model_dicts, PATH)
exit()


