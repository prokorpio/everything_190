# Import libraries
import torch
import time
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Logging utils
logging.basicConfig(level=logging.INFO, 
                    format=('%(levelname)s:' +
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))
get_log = True
if get_log:
    writer = SummaryWriter('runs/experiment_1')

# Define Agent, Training Env, & HyperParams
env = PruningEnv()
agent = REINFORCE_agent(env.state_size, 512)
M = 1# no reason, number of training episodes
layers_to_prune = [] # will be list of string names
for layer_name, _ in env.model.named_modules():
    if "conv" in layer_name:
        layers_to_prune.append(layer_name)

# Training Loop
for episode in range(M):
    print("\n=========== Episode", episode,"============")
    env.reset_to_k() # reset CNN to full-params
    #env._evaluate_model()
    action_reward_buffer = [] # list of (action,reward) tuples per episode

    # single rollout, layer-by-layer CNN scan
    for xp_num, layer_name in enumerate(layers_to_prune):
        env.layer_to_process = layer_name
        print("\n===== Working on", layer_name, "layer =====")

        # get state from orig model (or should we get from pruned model?)
        state = env.get_state()
        
        # get action
        action = agent.get_action(state)
        action = torch.unsqueeze(action, 0)
        action_to_index = (action > 0.5).type(torch.int)

        # perform action
        amount_pruned = env.prune_layer(action_to_index)

        # get reward
        logging.info("Calculating reward")
        reward,acc,flop,flops_ratio = env._calculate_reward(amount_pruned)
        action_reward_buffer.append((action, reward))
        
        # Log info's
        if get_log:
            total_xps = episode*len(layers_to_prune) + xp_num
            writer.add_scalar('Accuracy_vs_Experience', acc, total_xps)
            writer.add_scalar('Percent_Flops_Remaining_vs_Experience', 
                              flops_ratio, total_xps)
            writer.add_scalar('Reward_vs_Experience', reward, total_xps)
        
    # calc cumulative reward, agent learns 
    logging.info('Agent learning')
    actions, rewards = zip(*action_reward_buffer) # both var are tuple wrapped
    # actions: tuple->tensor
    actions = torch.squeeze(torch.stack(actions)).type(torch.float)
    agent.update_policy(rewards, actions) 

if get_log:
    writer.close()
