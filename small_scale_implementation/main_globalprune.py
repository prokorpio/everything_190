# Import libraries
import time
import os
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
from utilities import RandSubnet

# Logging utils
logging.basicConfig(level=logging.INFO, 
                    format=('%(levelname)s:' +
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

get_log = True 
xp_num_ =12


if get_log:
    print ("Initializing Experiment", xp_num_, "Writer")
    writer = SummaryWriter(('runs_march16_global/experiment_' + str(xp_num_)))

# Define Agent, Training Env, & HyperParams
env = PruningEnv()
# Obtain layers of the neural network
total_filters_count = 0
size_of_layer = []
for name, param in env.model.named_parameters():
    if 'conv' in name and 'weight' in name:
        total_filters_count += param.shape[0]
        print(name)
        print(param.shape[0])
        size_of_layer.append(param.shape[0])

#make an agent with said action_size
agent = REINFORCE_agent(env.state_size, action_size=total_filters_count)

print("Neural network has a total of" , total_filters_count)
print(size_of_layer, len(size_of_layer))
action = torch.rand(total_filters_count)
print(action.shape,"shape")
ratio_prune = 0.5
limit_rem = 0.05
action_per_layer = []
idx = 0
# for i in range(len(size_of_layer)):
    # print(idx,idx+size_of_layer[i],"indices")
    
    # current_action = action[idx:idx+size_of_layer[i]]
    # idx += size_of_layer[i]
    # print(current_action.shape, "curr")
    # tempmask = torch.ones(current_action.shape[0])
    # mag_rank = torch.topk(current_action,int(current_action.shape[0]*ratio_prune),largest = False)
    # tempmask[mag_rank[1]] = 0
    # action_per_layer.append(tempmask)
    # print(tempmask.sum())
# print(action_per_layer)


global_tempmask = torch.ones(action.shape[0])
global_mag_rank = torch.topk(action,int(action.shape[0]*ratio_prune),largest = False)
global_tempmask[global_mag_rank[1]] = 0

##inspect each layer's mask and unprune if needed
idx = 0
for i in range(len(size_of_layer)):

    #Get current action for the layer as well as the mask
    layer_action  = action[idx:idx+size_of_layer[i]]
    layer_pruned = global_tempmask[idx:idx+size_of_layer[i]].sum()
    
    

    
    if layer_pruned < int(limit_rem * size_of_layer[i]):
        unprune = torch.topk(layer_action,int(limit_rem*size_of_layer[1]),largest = True)
        layer_action[unprune[1]] = True
        global_tempmask[idx:idx+size_of_layer[i]] = layer_action
        
    idx += size_of_layer[i]

print(global_tempmask)
print(global_tempmask.shape)
print(action.shape[0])






M =250# no reason, number of training episodes

# Define RandSubnet, for benchmarking

start_time = time.time()
# Training Loop

for episode in range(M):
    print("\n=========== Episode", episode,"============")
    env.reset_to_k() # reset CNN to full-params
    #TODO: include condition to calc layer info ^^ 
    #env._evaluate_model()
    action_reward_buffer = [] # list of (action,reward) tuples per episode
    #pruned_prev_layer = 0 # how much was pruned in a previous layer
    flops_ratio_accumulated = 0
    amount_pruned_accum = 0
    total_reward = 0
    
    
    state_rep = env.get_global_state_rep()
    state_rep = state_rep.cpu()
    action = agent.get_action(state_rep)
    print(action.shape, "actionshape")
    global_tempmask = torch.ones(action.shape[0])
    # global_tempmask = action
    # global_tempmask /= global_tempmask
    global_mag_rank = torch.topk(action,int(action.shape[0]*ratio_prune),largest = False)
    global_tempmask[global_mag_rank[1]] = 0
    
    ##inspect each layer's mask and unprune if needed
    idx = 0
    for i in range(len(size_of_layer)):

        #Get current action for the layer as well as the mask
        layer_action  = action[idx:idx+size_of_layer[i]]
        layer_pruned = global_tempmask[idx:idx+size_of_layer[i]].sum()
        
        

        print("layer_pruned",layer_pruned)
        if layer_pruned < int(limit_rem * size_of_layer[i]):
            unprune = torch.topk(layer_action,int(limit_rem*size_of_layer[1]),largest = True)
            layer_action[unprune[1]] = True
            global_tempmask[idx:idx+size_of_layer[i]] = layer_action
            print("unpruned" ,size_of_layer[i])
        else:
            print("kept as pruned", size_of_layer[i])
            
        idx += size_of_layer[i]
    
    
    idx = 0
    print(env._evaluate_model())
    for i in range(len(size_of_layer)):
        env.layer = env.layers_to_prune[i]
        print("Pruning", env.layers_to_prune[i])
        layer_action = torch.unsqueeze(global_tempmask[idx:idx+size_of_layer[i]],0)
        env.prune_layer(layer_action)

    
    print(env._evaluate_model())
    amount_pruned = global_tempmask.sum()
    reward,acc,flop,flops_ratio = env._calculate_reward(total_filters_count, amount_pruned)
    action_reward_buffer.append((action, reward))
        
    # Log info's
    if get_log:
        flops_ratio_accumulated += flops_ratio #remaining per layer
        total_reward += reward 
        amount_pruned_accum += amount_pruned
        if (True):
            total_flops_ratio = flops_ratio_accumulated/4
            #total_xps = episode*len(layers_to_prune) + xp_num
            writer.add_scalar('Accuracy_vs_Episode', acc, episode)
            # writer.add_scalar('Pruned_Flops_Ratio_vs_Episode', 
                              # total_flops_ratio, episode)
            writer.add_scalar('Amount_Pruned_vs_Episode',
                              amount_pruned_accum, episode)
            writer.add_scalar('Reward_vs_Episode', 
                              total_reward, episode)
    #pruned_prev_layer = amount_pruned #next layer's previous is this layer

    
    # calc cumulative reward, agent learns 
    logging.info('RL Agent learning')
    actions, rewards = zip(*action_reward_buffer) # both var are tuple wrapped
    # actions: tuple->tensor
    actions = torch.squeeze(torch.stack(actions)).type(torch.float)
    agent.update_policy(rewards, actions) 
    print("amount_pruned", amount_pruned)
    print("acc", acc)

if get_log:
    writer.close()
    elapsed_time = time.time() - start_time
    logging.info("Total elapsed time: {}".format( 
                 time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    

##Train the final to compare with the unpruned model
PATH = os.getcwd() + '/pruned_march_15_' + str(xp_num_) + '.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)
