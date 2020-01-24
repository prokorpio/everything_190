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
xp_num = 1
if get_log:
    print ("Initializing Experiment", xp_num, "Writer")
    writer = SummaryWriter(('runs/experiment_' + str(xp_num)))

# Define Agent, Training Env, & HyperParams
env = PruningEnv()
agent = REINFORCE_agent(env.state_size, action_size=512)
M = 50# no reason, number of training episodes

#layers_to_prune = [] # will be list of string names
#for layer_name, _ in env.model.named_modules():
#    if "conv" in layer_name:
#        layers_to_prune.append(layer_name)

# Define RandSubnet, for benchmarking
rand_compare = False 
if rand_compare:
    rand_subnet = RandSubnet(env.model_type)


# Training Loop
for episode in range(M):
    print("\n=========== Episode", episode,"============")
    env.reset_to_k() # reset CNN to full-params
    #TODO: include condition to calc layer info ^^ 
    #env._evaluate_model()
    action_reward_buffer = [] # list of (action,reward) tuples per episode
    #pruned_prev_layer = 0 # how much was pruned in a previous layer

    # single rollout, layer-by-layer CNN scan
    for xp_num, layer_name in enumerate(env.layers_to_prune):
    #for xp_num, layer_name in enumerate(layers_to_prune):
        env.layer = layer_name
        print("\n===== Working on", layer_name, "layer =====")

        # get state from orig model (or should we get from pruned model?)
        state = env.get_state()
        
        # get action
        action = agent.get_action(state)
        action = torch.unsqueeze(action, 0)
        action_to_index = (action > 0.5).type(torch.int)

        # perform action 
        total_filters, amount_pruned = env.prune_layer(action_to_index)
        if rand_compare:
            remaining_filters = (total_filters-amount_pruned).item()
            rand_subnet.filter_counts.append(remaining_filters)

        # get reward
        logging.info("Calculating reward")
        reward,acc,flop,flops_ratio = env._calculate_reward()
        action_reward_buffer.append((action, reward))
        
        # Log info's
        if get_log:
            max_layer_num = len(env.layer_prune_amounts.keys())
            total_xps = episode*max_layer_num + xp_num
            #total_xps = episode*len(layers_to_prune) + xp_num
            writer.add_scalar('Accuracy_vs_Experience', acc, total_xps)
            writer.add_scalar('Percent_Flops_Remaining_vs_Experience', 
                              flops_ratio, total_xps)
            writer.add_scalar('Reward_vs_Experience', reward, total_xps)
        #pruned_prev_layer = amount_pruned #next layer's previous is this layer
    
    # get equivalent rand-init pruned network
    if rand_compare:
        logging.info('Building and Evaluating Equiv RandSubnet')
        rand_subnet.build()
        # may train subnet here for n epochs
        rand_acc = rand_subnet.evaluate(env.test_dl)
        logging.info('Rand Validation Accuracy: {:.2f}%'.format(rand_acc*100))
        if get_log:
            writer.add_scalar('RandAccuracy_vs_Episode', rand_acc, total_xps)

    # calc cumulative reward, agent learns 
    logging.info('RL Agent learning')
    actions, rewards = zip(*action_reward_buffer) # both var are tuple wrapped
    # actions: tuple->tensor
    actions = torch.squeeze(torch.stack(actions)).type(torch.float)
    agent.update_policy(rewards, actions) 

if get_log:
    writer.close()
    

###Train the final to compare with the unpruned model
#PATH = os.getcwd() + 'pruned_2.pth'
#torch.save(env.model.state_dict(), PATH)
