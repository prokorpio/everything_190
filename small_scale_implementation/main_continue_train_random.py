#main_retrain_masked

import torch
import time
import os
import copy
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter



# Basemark is 5381
trialnum = 2
writer = SummaryWriter(('runs_march17_continue/rand_march_17_global61_'+str(trialnum)))

env = PruningEnv()
env.reset_to_k()

#### Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        print(name)
        print(param.shape[0])
        size_of_layer.append(param.shape[0])
        
ratio_prune = 0.6
limit_rem = 0.05
random_mask = torch.rand((960))
global_randmask = torch.ones((960))
# global_tempmask = action
# global_tempmask /= global_tempmask
global_mag_rank = torch.topk(random_mask,int(random_mask.shape[0]*ratio_prune),largest = False)
global_randmask[global_mag_rank[1]] = 0

print(global_randmask.sum())


##inspect each layer's mask and unprune if needed
idx = 0

print(env._evaluate_model())
for i in range(len(size_of_layer)):
    
    #choose the layer
    env.layer = env.layers_to_prune[i]
    #Get current action for the layer as well as the mask
    layer_action  = global_randmask[idx:idx+size_of_layer[i]].clone()
    layer_values = random_mask[idx:idx+size_of_layer[i]].clone()
    layer_pruned = global_randmask[idx:idx+size_of_layer[i]].sum()
    
    

    print("Proposed_layer_kept",layer_pruned)
    if layer_pruned < int(limit_rem * size_of_layer[i]):
        unprune = torch.topk(layer_values,int(limit_rem*size_of_layer[1]),largest = True)
        layer_action[unprune[1]] = True
        global_randmask[idx:idx+size_of_layer[i]] = layer_action

    else:
        pass
        

    print("Amount kept is", global_randmask[idx:idx+size_of_layer[i]].sum())
    layer_action = torch.unsqueeze(global_randmask[idx:idx+size_of_layer[i]],0)    
    filters_counted, pruned_counted = env.prune_layer(layer_action)           
    idx += size_of_layer[i]

val_acc = env._evaluate_model()
print(val_acc)

for n_iter in range(85):
    if n_iter in ([25,55]):
        for param_group in env.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",n_iter)
    env._train_model(num_epochs = 1)
    val_acc = env._evaluate_model()
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, n_iter)



writer.close()
val_acc = env._evaluate_model()
print(val_acc)

##Train the final to compare with the unpruned model
PATH = os.getcwd() + '/rand_march_17_global_trained_61_'+str(trialnum)+'.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)
