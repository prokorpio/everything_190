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
xp_num_ = 108



M = 100# no reason, number of training episodes

if get_log:
    print ("Initializing Experiment", xp_num_, "Writer")
    writer = SummaryWriter(('runs_april_RL/experiment_' + str(xp_num_)))

# Define Agent, Training Env, & HyperParams
env = PruningEnv()

print(env.get_global_state_rep().shape)


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
#5e-3 shows promise 103 evaporated upon multiple trials
agent = REINFORCE_agent(env.state_size, action_size=total_filters_count, lr = 1e-7)

print("Neural network has a total of" , total_filters_count)
print(size_of_layer, len(size_of_layer))



ratio_prune = 0.5
limit_rem = 0.05
# Define RandSubnet, for benchmarking

start_time = time.time()
# Training Loop

env.reset_to_k()
for episode in range(M):
    print("\n=========== Episode", episode,"============")
    env.reset_to_k() # reset CNN to full-params
    
    print(env._evaluate_model())
   #TODO: include condition to calc layer info ^^ 
    #env._evaluate_model()
    action_reward_buffer = [] # list of (action,reward) tuples per episode
    #pruned_prev_layer = 0 # how much was pruned in a previous layer
    flops_ratio_accumulated = 0
    amount_pruned_accum = 0
    total_reward = 0
    
    if episode == 60:
        for param_group in agent.policy.Adamizer.param_groups:
            param_group['lr'] *= 1
    if episode == 120:
        for param_group in agent.policy.Adamizer.param_groups:
            param_group['lr'] *= 1      
    state_rep = env.get_global_state_rep()
    
    state_rep = state_rep.cpu()
    action, action_log_prob = agent.get_action(state_rep)
    print(action.shape)
    print(action_log_prob.shape)
    global_tempmask = torch.ones(action.shape[0])
    # global_tempmask = action
    # global_tempmask /= global_tempmask
    global_mag_rank = torch.topk(action,int(action.shape[0]*ratio_prune),largest = False)
    global_tempmask[global_mag_rank[1]] = 0
    
    
    ###Replace with ones and zeros from updated sampling
    #global_tempmask = action
    ##inspect each layer's mask and unprune if needed
    idx = 0
    total_pruned = 0
    for i in range(len(size_of_layer)):
        
        #choose the layer
        env.layer = env.layers_to_prune[i]
        #Get current action for the layer as well as the mask
        layer_action  = global_tempmask[idx:idx+size_of_layer[i]].clone()
        layer_values = action[idx:idx+size_of_layer[i]].clone()
        layer_pruned = global_tempmask[idx:idx+size_of_layer[i]].sum()
        
        #if local
        layer_tempmask = action[idx:idx + size_of_layer[i]].clone()
        layer_rank = torch.topk(layer_tempmask, int(size_of_layer[i]*ratio_prune), largest = False)
        layer_mask = torch.ones(size_of_layer[i])
        layer_mask[layer_rank[1]] = 0
    
        
         #Renaming local to fit with global scheme convention
        layer_action = layer_mask
        layer_values = layer_tempmask
        layer_pruned = layer_action.sum()


        print("Proposed_layer_kept",layer_pruned)
        if layer_pruned < int(limit_rem * size_of_layer[i]):
            unprune = torch.topk(layer_values,int(limit_rem*size_of_layer[1]),largest = True)
            layer_action[unprune[1]] = True

        else:
            pass
            

        print("Amount kept is", layer_action.sum())
        layer_action = torch.unsqueeze(layer_action,0)  
        total_pruned += size_of_layer[i] - layer_mask.sum()
        ###Actual_Pruning
        filters_counted, pruned_counted = env.prune_layer(layer_action)           
        idx += size_of_layer[i]
    
    
    idx = 0
    # for i in range(len(size_of_layer)):
        # env.layer = env.layers_to_prune[i]
        # print("Pruning", env.layers_to_prune[i])
        # print("Amount pruned is", global_tempmask[idx:idx+size_of_layer[i]].sum())
        # layer_action = torch.unsqueeze(global_tempmask[idx:idx+size_of_layer[i]],0)


    
    amount_pruned = 960-total_pruned
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

    agent.update_policy(rewards, actions, action_log_prob) 
    print("amount_pruned", amount_pruned)
    print("acc", acc)

if get_log:
    writer.close()
    elapsed_time = time.time() - start_time
    logging.info("Total elapsed time: {}".format( 
                 time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    

###Obtain Final Network and it's inverse
#Reset the network
env.reset_to_k()

#Get the state_rep and the corresponding action
state_rep = env.get_global_state_rep()
    
state_rep = state_rep.cpu()
action, action_log_prob = agent.get_action(state_rep)

idx = 0
total_pruned = 0
inv_flaglist = [True, False]
for inv_flag in inv_flaglist:
    for i in range(len(size_of_layer)):
        
        #choose the layer
        env.layer = env.layers_to_prune[i]
        
        #if local
        layer_tempmask = action[idx:idx + size_of_layer[i]].clone()
        
        if inv_flag == True:
            layer_rank = torch.topk(layer_tempmask, int(size_of_layer[i]*ratio_prune), largest = True)
        else:
            layer_rank = torch.topk(layer_tempmask, int(size_of_layer[i]*ratio_prune), largest = False)
        layer_mask = torch.ones(size_of_layer[i])
        layer_mask[layer_rank[1]] = 0

        
        #Renaming local to fit with global scheme convention
        layer_action = layer_mask
        layer_values = layer_tempmask
        layer_pruned = layer_action.sum()


            

        print("Amount kept is", layer_action.sum())
        layer_action = torch.unsqueeze(layer_action,0)  
        total_pruned += size_of_layer[i] - layer_mask.sum()
        ###Actual_Pruning
        filters_counted, pruned_counted = env.prune_layer(layer_action)           
        idx += size_of_layer[i]


    idx = 0
    ##Train the final to compare with the unpruned model
    
    if inv_flag == True:
        PATH = os.getcwd() + '/april_RL_inv' + str(xp_num_) + '.pth'
    else:
        
        PATH = os.getcwd() + '/april_RL' + str(xp_num_) + '.pth'
    model_dicts = {'state_dict': env.model.state_dict(),
            'optim': env.optimizer.state_dict()}
    torch.save(model_dicts, PATH)

