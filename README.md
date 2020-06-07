## Authors
This codebase was developed by Jeff Sanchez and Marcus Reyes for their senior-year project for our subject EEE 198; UCL, UP-Diliman 2020

## Background
This library contains the implementation of the Reinforcment Learning portion of our thesis project, entitled "Heuristic Approach Towards Early Compression Of Neural Networks. Inspired by recent works on Lottery Tickets [1,2], we attempted to automate the process of finding high performing and high sparsity but structured subnetworks (winning tickets) without relying on already trained networks. This form of compression allows for memory and energy savings in training neural networks. Initially we proposed to do so using Reinforcement Learning. The folder [small_scall_implementation/](small_scale_implementation) contains the code for running our RL experiments. However the github for our final chosen appoach, Simulated Annealing, is at https://github.com/marcus-reyes/EEE198_UCL10

Our process can briefly be summarized as follows.
1. We input a representation of a CNN into the RL Agent
2. The RL Agent outputs an action, pruning filters from the CNN according to a fixed sparsity.
3. The Rl Agent updates its parameters and optimizes based on the validation accuracy of the pruned network
4. The RL output mask that obtains the best untrained accuracy is taken.
5. This mask is applied on the CNN network with the CNN having its parameters at initial value
6. This CNN is pruned and the subnetwork is trained to convergence.

## Summary of Results
For the RL our final result was that if the reward was inverted (such that it did NOT follow the REINFORCE equation), a semblance of performance was achieved by the RL agent. A in-depth analysis was not done due to time constraints. Also, the bulk of our work was on using SA. This was due to the intractability of our RL agent, which led us to change our apporach to the problem to SA. A complete documentation of results, justification for our change of approach, etc. can be found in our 198 Manuscript

## ReadMe for Running the RL experiment
For running experiments
1. Using initializer.py and trainer.py, obtain Checkpoints of BasicCNN at initialization, epoch 0,2,5,90. Name properly within initializer.py and trainer.py. Edit environment.py functions to load the said checkpoints (i.e. reset_to_k_90, reset_to_init_1, etc.) with the naming convention used.

2. Run main_script.py. Name the XP numbers in increments of 5.


## Updated Table of Contents
1. [small_scale_implementation/](small_scale_implementation) : RL pruner implementation for final setup of 198
 >1. [REINFORCE_agent.py](small_scale_implementation/REINFORCE_agent.py) : the RL agent
 >2. [environment.py](small_scale_implementation/environment.py) : ties the agent and the models to be pruned
 >3. [model.py](small_scale_implementation/model.py) : Contains modules used by RL agent
 >4. [models_to_prune.py](small_scale_implementation/models_to_prune.py) : Various CNN modules to run pruning algo onto
 >5. [initializer.py](small_scale_implementation/initializer.py) : File for initializing a BasicCNN
 >6. [trainer.py](small_scale_implementation/trainer.py) : File for training the BasicCNN initialized in initializer.py
 >7. [main_script.py](small_scale_implementation/main_script.py) : Python file that runs all the .py files needed to search for a mask using RL, prune the BasicCNN indicated in environment.py, train the smaller subnetwork.
 >8. [main.py](small_scale_implementation/main.py) : Python file for main algorithm
 >9. [actual_prune.py](small_scale_implementation/actual_prune.py) : Python file that takes the output of main.py and actually prunes the network.
 >8. [train_actual_subnet.py](small_scale_implementation/train_actual_subnet.py) : Python file that takes the output of actual_prune.py and trains the subnetwork


## Everything below is outdated

## (Outdated)Table of Contents
1. [small_scale_implementation/](small_scale_implementation) : RL pruner implementation on a 4-layer CNN
  >1. [REINFORCE_agent.py](small_scale_implementation/REINFORCE_agent.py) : the RL agent
  >2. [environment.py](small_scale_implementation/environment.py) : ties the agent and the models to be pruned
  >3. [model.py](small_scale_implementation/model.py) : Contains modules used by RL agent
  >4. [models_to_prune.py](small_scale_implementation/models_to_prune.py) : Various CNN modules to run pruning algo onto
  >5. [main.py](small_scale_implementation/main.py) : Main algorithm utilizing subroutines from above
  >6. [temp_files/](small_scale_implementation/temp_files) : necessary parts needing further iterative work
  >7. [best_snapshot_78.pt](small_scale_implementation/best_snapshot_78.pt) : fine-tuned model of the 4-layer CNN
  >8. [rat11.py](small_scale_implementation/rat11.py) : sample code that evaluates the fine-tuned model, prunes a layer, the re-evaluates
2. [labrat_reyes/](labrat_reyes)
3. [misc/](misc) : Various unused files
  >1. [CompressionImageNetLinks.txt](misc/CompressionImageNetLinks.txt)
  >2. [SNIP_reimplementation.ipynb](misc/SNIP_reimplementation.ipynb)


## Requirements:
1. Tensorboard
  > to install external viewer: 'pip install tensorboard' or may use conda <br>
  > to reset plots data: delete runs/ folder
  
