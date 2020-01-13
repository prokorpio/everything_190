# Import libraries
import torch
import time
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent

import logging
logging.basicConfig(level=logging.INFO, 
                    format=('%(levelname)s:' +
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

# Define Agent, Training Env, & Hyper-params
env = PruningEnv()
agent = REINFORCE_agent(env.state_size, 512)

M = 5 # no reason, number of training episodes
layers_to_prune = [] # will be list of string names
for layer_name, _ in env.model.named_modules():
    if "conv" in layer_name:
        layers_to_prune.append(layer_name)

for episode in range(M):
    print("=====New Episode=====")
    env.reset_to_k() # reset CNN to full-params
    #env._evaluate_model()
    #time.sleep(1.5)
    action_reward_buffer = [] # list of (action,reward) tuples per episode

    # single rollout, layer-by-layer CNN scan
    for layer_name in layers_to_prune:
        env.layer_to_process = layer_name
        #time.sleep(1.5)
        print("===== Working on", layer_name, "layer =====")
        # get state from orig model (or should we get from pruned model?)
        state = env.get_state()
        action = agent.get_action(state)
        logging.info("Actions: {}".format(action))
        action_to_index = (action > 0.5).type(torch.int)
        #logging.info("action_to_index sum: {}".format(action_to_index.sum()))
        env.prune_layer(action_to_index)
        print("Calculating reward")
        reward = env._calculate_reward()
        action_reward_buffer.append((action, reward))  

    # calc cumulative reward, agent learns 
    actions, rewards = zip(*action_reward_buffer) # both var are tuple wrapped
    # actions: tuple->tensor
    actions = torch.squeeze(torch.stack(actions)).type(torch.float)
    agent.update_policy(rewards, actions) 

    

