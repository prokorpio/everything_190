#main_retrain_masked

import torch
import time
import os
import copy
from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter



# Basemark is 5381


trialnum = 1
writer = SummaryWriter(('runs_march17_continue/pruned_march_17_global23_'+str(trailnum)))

env = PruningEnv()
env.model.load_state_dict(torch.load(os.getcwd() + \
                                                '/pruned_march_17_global23.pth')['state_dict'])
env.optimizer.load_state_dict(torch.load(os.getcwd() + \
                            '/pruned_march_17_global23.pth')['optim'])
        
 

val_acc = env._evaluate_model()
print(val_acc)
for n_iter in range(85):
    if epoch in ([25,55]):
        for param_group in env.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",n_iter)
    env._train_model(num_epochs = 1)
    val_acc = env._evaluate_model()
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, n_iter)

writer.close()
val_acc = env._evaluate_model()
print(val_acc)

##Train the final to compare with the unpruned model
PATH = os.getcwd() + '/pruned_march_17_global_trained_23_' + str(trialnum) + '.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)