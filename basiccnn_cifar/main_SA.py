#main_SA

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

from utilities import *

import argparse

from collections import deque

xp_num_ = 9
trialnum = xp_num_
writer = SummaryWriter(('runs_april_SA/experiment_' + str(xp_num_)))
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



env = PruningEnv()
env.reset_to_init_1()

#### Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        size_of_layer.append(param.shape[0])


mask_list = 0
mask = torch.cat((torch.ones((3)), torch.zeros((3))),0)
mask = torch.rand((total_filters_count))
print(mask)
mask[mask >0.3] = 1
mask[mask<=0.3] = 0
print(mask)
mask_list = copy.deepcopy(mask)


#SA Loop



mem_size = 50



##HAM DIST VARS
ham_dist = int(mask.sum()/2)
ham_dist_decay = 0.99
prev_ham_dist = ham_dist

###TEMP variables

iter_per_temp =  1# allows multiple decisions per given temp value
iter_multiplier = 1.005 # increase iters for every temp decrease
max_iter_per_temp = 50 # ceiling on iterations per temp value'


ave_acc = 5
accs = [ave_acc]
acc_temp = 0.075
acc_temp_decay = 0.995


stop_flag = True #Remove this later with always true. It is merely a place holder for now to stop after one iteration

z = 0
current_mask = mask

closed_q = deque(maxlen=mem_size)
closed_q.append(current_mask)

print(closed_q, "Closed_Q")
new_mask_flag = 0 # 1 if new mask found
print("Current_Mask")
while (stop_flag == True):
    print("TRIAL START------------------")
    #Find a new TEST mask
    for i in range(int(iter_per_temp)):
        # z = z + 1
        new_mask_flag = 0
        while new_mask_flag == 0:
            print("Generating New Mask")
            new_mask = step_from(current_mask, ham_dist)
            if is_in_list(new_mask, closed_q):
                print("Duplicate")
                # print(new_mask)
            else:
                print("Non duplicate")
                # print("Final_mask", current_mask)
                
                closed_q.append(new_mask)
                new_mask_flag = 1
        
        #Tentatively implement the mask
        idx = 0
        total_pruned = 0
        env.reset_to_init_1()
        for i in range(len(size_of_layer)):
            env.layer = env.layers_to_prune[i]
            layer_mask = new_mask[idx:idx+size_of_layer[i]].clone()
            layer_mask = torch.unsqueeze(layer_mask,0)
            total_pruned += size_of_layer[i] - layer_mask.sum()
            
            filters_counted, pruned_counted = env.prune_layer(layer_mask)
            idx += size_of_layer[i]
        
        amount_pruned = total_pruned
        idx = 0
        
        #Check if keep or discard
        # _, new_acc, _, _  = env._calculate_reward(total_filters_count, amount_pruned)
        new_acc = env.forward_pass(1)
        # new_acc = new_acc * 100
        # print("ACC IS ", new_acc)
        
        ave_acc = sum(accs)/len(accs)
        
        acc_delta = (ave_acc - new_acc)/100
        #print(acc_delta, ave_acc, new_acc, "del ave new")
        
        if acc_delta < 0:
            print("Accepted the mask")
            current_mask = new_mask
            accs.append(new_acc) 
            
            #Update the averate
            ave_acc = sum(accs)/len(accs)
        else:
            print("Deciding...")
            allow_uphill = True
            if allow_uphill:
                q = torch.rand(1).item()
                acc_prob = math.exp(-1*acc_delta/acc_temp)
                print(acc_prob, "Acc_prob")
                if q < acc_prob:
                    print("Accepted")
                    current_mask = new_mask
                    accs.append(new_acc)
                    
                    #Update the average before resetting
                    ave_acc = sum(accs)/len(accs)
                else:
                    print("Rejected")
                    pass
        
    accs = [ave_acc]
    print("AVE ACC", accs)
    
    writer.add_scalar('ave_acc', ave_acc, z)
    writer.add_scalar('iter_per_temp', int(iter_per_temp),z)
    writer.add_scalar('ham_dist', ham_dist, z)
    writer.add_scalar('amount_pruned' , amount_pruned, z)
    #Scheduler of Neighbor Size
    ham_dist = max(1, int(ham_dist*ham_dist_decay))
    
    if ham_dist < prev_ham_dist*0.5:
        ham_dist = int(prev_ham_dist*0.75)
        prev_ham_dist = ham_dist
    
    #Scheduler of TEMP
    acc_temp = acc_temp * acc_temp_decay

    
    #Scheduler of iter_per_temp
    iter_per_temp = min(max_iter_per_temp, iter_per_temp*iter_multiplier)
    
    
    

    z += 1
    if z == 10000:
        stop_flag = False
        
PATH = os.getcwd() + '/pruned_may_10' + str(xp_num_) + '.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)
writer.close()
# print(closed_q)


###TEST WITH UPHILL AND NO UPHILL
