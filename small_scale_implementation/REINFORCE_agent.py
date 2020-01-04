#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import gym
import numpy as np
import torch
import torch.nn as nn # neural network class, parang Keras
import torch.nn.functional as F # special functions
#from torch.autograd import Variable
import torch.optim as optim # for optimizer = Adam
#import matplotlib.pyplot as plt
from model import PolicyNetwork

class REINFORCE_agent():
    """ add description"""

    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3):

        self.state_size= state_size
        self.action_size = action_size

        self.episode_rewards = []

        self.gamma = gamma
        self.lr = lr

        self.policy = PolicyNetwork(state_size = self.state_size,
                                    action_size = self.action_size,
                                    lr = self.lr)

    # TODO: make rewards, log probs internal?
        

    # TODO: change this to get_action_distrib 
    
    def get_action(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0) # tensor convert for backprop compat
                                                             # unsqueeze bc batch-dim expected @ dim=0
        action_prob_distrib = self.policy(state) 
        #action_log_prob = torch.log(action_prob_distrib) 
        return action_prob_distrib#, action_log_prob
        #sampled_action = np.random.choice(self.action_size, \
        #                    p=np.squeeze(action_prob_distrib.detach().numpy()))
                                # detach from autograd graph
        #action_log_prob = torch.log(action_prob_distrib.squeeze(0)[sampled_action])

        
        #return sampled_action, action_log_prob

    def update_policy(self, episode_rewards, actions):

        # Compute Return function for each experience in the episode
        returns = np.zeros_like(episode_rewards)
        Gt = 0.0
        for t in reversed(range(len(episode_rewards))): # t is timestep
            Gt = episode_rewards[t] + self.gamma*Gt     # Return function Gt
            returns[t] = Gt                             # Return per time step     
        
        returns = torch.tensor(returns) 
        returns = (returns - returns.mean())/(returns.std() + 1e-9)
                             # standardized to control variance of Return

        expanded_returns = torch.zeros(len(returns), self.action_size)
        for i, Gt in enumerate(returns):
            # mult Gt only on activated channels
            expanded_returns[i, np.where(actions[i] == 1)[0]] = Gt

        # Compute gradients, 
        J_t = expanded_returns.matmul(torch.log(actions)) 
        #J_t = [] # will summands of objective function
        #for log_prob, Gt in zip(log_probs, returns):
        #    J_t.append(-log_prob*Gt) # REINFORCE policy gradient theorem, Q = Gt
         
        self.policy.Adamizer.zero_grad() # reset weight update grads to zero
        objective_func = torch.stack(J_t).sum()  # stack will concat multiple 1x1 tensors
                                                 # to single vector tensor, then sum elements
        objective_func.backward()  # assigns grad attribute to all Variables
        self.policy.Adamizer.step() # gradient ascent step

        

# =====================USE AGENT=============================
'''
# Pre-training Setup

# Create Environment
env = gym.make('CartPole-v1') 
#env = env.unwrapped # get the core version without wrappers
                    # which are additional functionalities 
                    # encapsulating the object
env.seed(1)         # pg is high variant, for reproducability

# Setup HyperParams
## ENV params
state_size = env.observation_space.shape[0] # 4
action_size = env.action_space.n

## TRAIN params
max_episodes = 2000  # stops training when reached
max_steps = 10000    # max num of steps in an episode

# Instantiate Agent 
agent = REINFORCE_agent(state_size, action_size)

# Training
agent.policy.train() # set to training mode

numsteps = [] # will indicate how long the pole was balanced
avg_numsteps = [] # running avg, for better viz
running_reward = 10 # will calculate running ave of reward

for episode in range(max_episodes+1):
    state = env.reset() # get starting state
    log_probs = []      # list of action log probs in this episode
    rewards = []        # list of action reward in this episode
    
    for steps in range(max_steps): 
        #env.render()
        action, action_log_prob = agent.get_action(state) 
        new_state, reward, done, _ = env.step(action) # perform action {0,1}
        log_probs.append(action_log_prob)
        rewards.append(reward) # TODO: move this inside agent

        if done:
            # update weights
            agent.update_policy(rewards, log_probs)
            
            # record process info
            numsteps.append(steps)
            running_reward = 0.05*np.sum(rewards) + (1-0.05)*running_reward
                
            # print info
            if episode % (max_episodes//100) == 0:
                print("==========================================")
                print("Episode:\t", episode)
                print("RunReward:\t", running_reward)
                print("Timesteps:\t", steps)
                
            break
        
        state = new_state
    
    # CartPole-v1 solved condition  
    if running_reward > env.spec.reward_threshold:
        print('CarPole-v1 solved!')
        torch.save(agent.policy.state_dict(), './saved_models/test_reinforce.pt')
        print('Model Saved.')
        break


# Test Policy

# load trained weights
agent.policy.load_state_dict(torch.load('./saved_models/test_reinforce.pt'))
agent.policy.eval() # set to evaluation mode

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0 # of the episode
    print("****************************************************")
    print("EPISODE ", episode)
    
    while True:
        env.render()
        action, _ = agent.get_action(state)    # decide on action
        new_state, reward, done, _ = env.step(action)  # perform action
        total_rewards += reward
        
        if done:
            print('Accumulated reward:', total_rewards)
            break
        state = new_state
env.close()
    
# TODO:
# make device explicit(gpu or cpu)
'''




