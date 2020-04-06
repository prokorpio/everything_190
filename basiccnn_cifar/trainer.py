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


import argparse
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')



# Basemark is 5381

env = PruningEnv()
# env.model.load_state_dict(torch.load(os.getcwd() + \
                                        # '/sgd_90_.pth')['state_dict']) 

writer = SummaryWriter('runs_april_model_withBN')
    
val_acc = env._evaluate_model()
print(val_acc)

val_accs = []
for epoch in range(90):
    if epoch in ([30,60]):
        for param_group in env.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",epoch)
    env._train_model(num_epochs = 1)
    
    val_acc = env._evaluate_model()
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, epoch)
    if epoch == 5:
        model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
        PATH = os.getcwd() + '/april_experiments_withBN_epoch_' + str(epoch) + '.pth'
        torch.save(model_dicts, PATH)
val_acc = env._evaluate_model()
writer.close()
print(val_acc)
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
PATH = os.getcwd() + '/april_experiment_withBN_final_epoch.pth'
torch.save(model_dicts, PATH)
