#main_SA

import torch
import time
import os
import copy
import math
from environment import PruningEnv
#from REINFORCE_agent import REINFORCE_agent
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utilities import *

import argparse

from collections import deque


xp_num_ = 5

env = PruningEnv()

PATH = os.getcwd() + '/init_may_31_num_' + str(xp_num_) + '.pth'
model_dicts = {'state_dict': env.model.state_dict(),
        'optim': env.optimizer.state_dict()}
torch.save(model_dicts, PATH)

print(env._evaluate_model())
for i in range(10):
    print(env.forward_pass(1))
