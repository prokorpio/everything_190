# Import libraries
import torch
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent

# Define Agent, Training Env, & Hyper-params
env = PruningEnv()
agent = REINFORCE_agent(env.state_size, 512)

M = 50 # no reason, number of training episodes
layers_to_prune = [] # will be list of string names
for layer_name, _ in env.model.named_modules():
    if "conv" in layer_name:
        layers_to_prune.append(layer_name)

for episode in range(M):
    env.reset() # reset CNN to full-params
    action_reward_buffer = [] # list of (action,reward) tuples per episode

    # single rollout, layer-by-layer CNN scan
    for layer_name in layers_to_prune:
        env.layer_to_process = layer_name
        # get state from orig model (or should we get from pruned model?)
        state = env.get_state()
        action = agent.get_action(state)
        action = (action > 0.5).type(torch.int)
        env.prune_layer(action)
        reward = env._calculate_reward()
        action_reward_buffer.append((action_log_prob,reward))  

    # calc cumulative reward, agent learns 
    actions, rewards = zip(*action_reward_buffer)
    print(actions, "Actions")
    print(rewards, "Rewards")
    agent.update_policy(rewards, actions)

    

