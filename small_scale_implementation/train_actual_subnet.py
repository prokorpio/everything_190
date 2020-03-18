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
from utilities import RandSubnet
import torch

#1 is 35,68,128,261
#2 is 37,62,140,253
#3 is 32,63,120,257

env = PruningEnv()

PATH = os.getcwd() + '/actual_pruned_march_17_global_23.pth'

model_dicts = torch.load(PATH)

filters_per_layer = model_dicts['filters_per_layer']
pruned_subnet = RandSubnet(filter_counts = filters_per_layer)
pruned_subnet.build()
for name, param in pruned_subnet.model.named_modules():
    print("N,P", name, param)

pruned_subnet.model.load_state_dict(model_dicts['state_dict'])

val_acc = pruned_subnet.evaluate(env.test_dl)
print(val_acc)
writer = SummaryWriter('runs_march_18/global_23_pruned')

start = time.time()
for n_iter in range(1):
    if n_iter in ([25,55]):
        for param_group in pruned_subnet.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",n_iter)
    pruned_subnet.train_model(env.train_dl, num_epochs = 1)
    val_acc = pruned_subnet.evaluate(env.test_dl)
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, n_iter)

end = time.time()
writer.close()
val_acc = pruned_subnet.evaluate(env.test_dl)
print(val_acc)
total_time = end - start
print(total_time)
PATH = os.getcwd() + '/actual_pruned_march_17_global_23_trained.pth'

model_dicts = {'state_dict': pruned_subnet.model.state_dict(),
        'optim': pruned_subnet.optimizer.state_dict(),
        'filters_per_layer': filters_per_layer}
torch.save(model_dicts, PATH)


#unpruned 3728.4113001823425 named as pruned but no number
#pruned 75 2271.4468524456024 (error? CPU overclocked?)
#pruned 50 3788.0326392650604
#pruned 75 3844.9061708450317