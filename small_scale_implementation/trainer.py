import torch
import time
import os
import copy
from environment import PruningEnv
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter


import argparse
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')



# Basemark is 5381

env = PruningEnv()
env.reset_to_init_1()
# env.model.load_state_dict(torch.load(os.getcwd() + \
                                        # '/sgd_90_.pth')['state_dict']) 

writer = SummaryWriter('runs_training_may_exp/init_5_unpruned')
    
val_acc = env._evaluate_model()
print(val_acc)

val_accs = []
writer.add_scalar('Test/train', val_acc, 0)
for epoch in range(90):
    if epoch in ([30,60]):
        for param_group in env.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",epoch)
    env._train_model(num_epochs = 1)
    
    val_acc = env._evaluate_model()
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, epoch + 1)
    if epoch in ([0,2,5]):
        model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
        PATH = os.getcwd() + '/may_31_init_5_trained_' + str(epoch) + '.pth'
        torch.save(model_dicts, PATH)
val_acc = env._evaluate_model()
writer.close()
print(val_acc)
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
PATH = os.getcwd() + '/may_31_init_5_trained_90.pth'
torch.save(model_dicts, PATH)
