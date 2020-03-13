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
xp_num_ = 6
#1 march 13 is with amount pruned to be increased (works as intended)
#2 is with amount pruned to be decreased
#3 is still amount to be decreased  but wt threshold set to 0.5
#4 threshold at 0.5 and -amount_pruned as reward same with [5,

#2 is with 200 episodes basline
#3 is with accuracy only
#4 is with 485
#5 is with 485 and 1 epoch of training
#6 is idk
#7 is with inverted reward and 0.49
#8 is with (3+acc) reward
#9 is with (3+acc)*np.log(flops_remain) and 0.49
#10 is same as 9 but without the grads in state rep
#11 is (3+acc)*(total/amount pruned) without grads in state rep
#12 is with the inverse of 11 since 11 forces amount prund to go up
if get_log:
    print ("Initializing Experiment", xp_num_, "Writer")
    writer = SummaryWriter(('runs_march/experiment_march13_' + str(xp_num_)))

# Define Agent, Training Env, & HyperParams
env = PruningEnv()
agent = REINFORCE_agent(env.state_size, action_size=512)
M = 500# no reason, number of training episodes

# Define RandSubnet, for benchmarking
rand_compare = False 
if rand_compare:
    rand_subnet = RandSubnet(env.model_type)

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

    # single rollout, layer-by-layer CNN scan
    for xp_num, layer_name in enumerate(env.layers_to_prune):
    #for xp_num, layer_name in enumerate(layers_to_prune):
        env.layer = layer_name
        print("\n===== Working on", layer_name, "layer =====")

        # get state from orig model (or should we get from pruned model?)
        state = env.get_state()
        #logging.info('state: {}'.format(state[-3:]))
        logging.info('Sum %flops: {}'.format(sum(state[-3:]))) 
                                        #network remaining %flops

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
        reward,acc,flop,flops_ratio = env._calculate_reward(total_filters, amount_pruned)
        action_reward_buffer.append((action, reward))
        
        # Log info's
        if get_log:
            flops_ratio_accumulated += flops_ratio #remaining per layer
            total_reward += reward 
            amount_pruned_accum += amount_pruned
            if (xp_num == 1):
                total_flops_ratio = flops_ratio_accumulated/4
                #total_xps = episode*len(layers_to_prune) + xp_num
                writer.add_scalar('Accuracy_vs_Episode', acc, episode)
                writer.add_scalar('Pruned_Flops_Ratio_vs_Episode', 
                                  total_flops_ratio, episode)
                writer.add_scalar('Amount_Pruned_vs_Episode',
                                  amount_pruned_accum, episode)
                writer.add_scalar('Reward_vs_Episode', 
                                  total_reward, episode)
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
    elapsed_time = time.time() - start_time
    logging.info("Total elapsed time: {}".format( 
                 time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    

##Train the final to compare with the unpruned model
PATH = os.getcwd() + '/pruned_march_11_' + str(xp_num_) + '.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)
