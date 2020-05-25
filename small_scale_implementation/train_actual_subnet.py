import torch

import time
import os
import copy
from environment import PruningEnv
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utilities import PrunedSubnet
import torch

import argparse
###Argument parsing
parser = argparse.ArgumentParser(description='Arguments for masker')
parser.add_argument('--criterion', type=str, default='mag',
                    help='criterion to use')
parser.add_argument('--foldername', type=str, default = 'pruned_may_exp',
                    help='folder to store masked networks in')
parser.add_argument(
    "--ratio_prune", type=float, default=0.7, help="amount to prune"
)
parser.add_argument(
    "--xp_num_", type = int, default = 100, help="experiment number"
)
parser.add_argument(
    "--method", type = str, default = "SA", help="method to use"
)

args = parser.parse_args()


if args.method == "SA":
    from_string = "_pruned.pth"
    to_string = "_trained.pth"
elif args.method == "rand":
    from_string = "_rand_pruned.pth"
    to_string = "_rand_trained.pth"
    
elif args.method == "mag_rewind":
    from_string = "_mag_rewind_pruned.pth"
    to_string = "_mag_rewind_trained.pth"
    
elif args.method == "mag_sign_rewind":
    from_string = "_mag_sign_rewind_pruned.pth"
    to_string = "_mag_sign_rewind_trained.pth"
    
elif args.method == "RL":
    from_string = "_RL_pruned.pth"
    to_string = "_RL_trained.pth"


#1 is 35,68,128,261
#2 is 37,62,140,253
#3 is 32,63,120,257

env = PruningEnv()

folder = '/'+ args.foldername + '/'

PATH_from = (
    os.getcwd()
    + "/pruned_may_exp/exp"
    + "_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + str(from_string)
)
model_dicts = torch.load(PATH_from)


filters_per_layer = model_dicts['filters_per_layer']
pruned_subnet = PrunedSubnet(filter_counts = filters_per_layer)
pruned_subnet.build()
for name, param in pruned_subnet.model.named_modules():
    print("N,P", name, param)

pruned_subnet.model.load_state_dict(model_dicts['state_dict'])


best_val_acc = 0.0

writer = SummaryWriter(
    ("runs_training_may_exp/exp_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + "_"
    + str(args.method))
)


start = time.time()
for n_iter in range(90):
    if n_iter in ([30,60]):
        for param_group in pruned_subnet.optimizer.param_groups:
            param_group['lr'] *= 0.1
    print("EPOCH",n_iter)
    pruned_subnet.train_model(env.train_dl, num_epochs = 1)
    val_acc = pruned_subnet.evaluate(env.test_dl)
    print(val_acc, best_val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print("does this happen")
    print(val_acc)
    writer.add_scalar('Test/train', val_acc, n_iter)

end = time.time()
writer.close()
val_acc = pruned_subnet.evaluate(env.test_dl)
print(val_acc)
total_time = end - start
print(total_time)
print(best_val_acc)

if not os.path.exists('trained_may_exp'):
    os.makedirs('trained_may_exp')
    
    
PATH_to = (
    os.getcwd()
    + "/trained_may_exp/exp"
    + "_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + str(to_string)
)
model_dicts = {'state_dict': pruned_subnet.model.state_dict(),
        'optim': pruned_subnet.optimizer.state_dict(),
        'filters_per_layer': filters_per_layer}
torch.save(model_dicts, PATH_to)

log_file = open(
    "textlogs/exp_"
    + str(args.xp_num_)
    + "_sparsity_"
    + str(int(args.ratio_prune*100))
    + ".txt", "a"
)

log_file.write(str("best_trained_acc: " + str(best_val_acc) + "\n"))
log_file.close()
#unpruned 3728.4113001823425 named as pruned but no number
#pruned 75 2271.4468524456024 (error? CPU overclocked?)
#pruned 50 3788.0326392650604
#pruned 75 3844.9061708450317
