# Import libraries
import time
import os
import logging

import numpy as np
import torch

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from environment import PruningEnv
from REINFORCE_agent import REINFORCE_agent
from utilities import RandSubnet

import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Arguments for RL masker")
parser.add_argument(
    "--foldername",
    type=str,
    default="trash",
    help="folder to store masked networks in",
)
parser.add_argument(
    "--ratio_prune", type = float, default = 0.5, help="amount to prune"
)
parser.add_argument(
    "--num_batches",
    type=int,
    default=1,
    help="number of batches for the search evaluation forward pass",
)
parser.add_argument(
    "--max_episodes", type = int, default = 2500, help="maximum episodes"
)
parser.add_argument(
    "--xp_num_", type = int, default = 100, help="experiment number"
)
parser.add_argument(
    "--method", type = str, default = "RL", help="method to use"
)
parser.add_argument(
    "--k_epoch", type = int, default = 5, help = "which k to reset to"
)
args = parser.parse_args()
# Logging utils
logging.basicConfig(level=logging.INFO, 
                    format=('%(levelname)s:' +
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

get_log = True 
xp_num_ = 8002


###Create directories
if not os.path.exists("masked_may_exp"):
    os.makedirs("masked_may_exp")

#30 is with reset to kept
#40 is with reset to init1
#50 is reset to k but longer than 30
#600 is with trained
#7001 is with no negative
#7002 is with no negative

print ("Initializing Experiment", args.xp_num_, "Writer")
writer = SummaryWriter(
    ("runs_RL_may_exp/RL_exp_"
    + str(args.xp_num_)
    + "_"
    + str(int(args.ratio_prune*100)))
)

# Define Agent, Training Env, & HyperParams
env = PruningEnv(state_size = 960)
agent = REINFORCE_agent(env.state_size, action_size=960)


start_time = time.time()
# Training Loop
best_acc = 0
best_mask = torch.ones((env.state_size))
best_episode_count  = 0
for episode in range(args.max_episodes):
    print("\n=========== Episode", episode,"============")

    
    action_reward_buffer = [] # list of (action,reward) tuples per episode
    
    
    
    
    ###Reset CNN to k-epoch params and get the state for the entire episode 

    if args.k_epoch == 0:
        env.reset_to_k_0()
        
    elif args.k_epoch == 2:
        env.reset_to_k_2()
        
    elif args.k_epoch == 5:
        env.reset_to_k_5()
        
    elif args.k_epoch == 90:
        env.reset_to_k_90()
        
    else:
        env.reset_to_init_1()

    state = env.get_global_state_rep()
    state = state.cpu()
    
    ###Action is the probabilities
    ###log_prob is the log prob of action
    ###Actual_action is the masks using m.sample()
    action, log_prob, actual_action = agent.get_action(state)
    
    print(log_prob)
    ###Obtain the mask
    tempmask = torch.ones(action.shape[0])
    mag_rank = torch.topk(action,int(action.shape[0]*args.ratio_prune),largest = False)
    tempmask[mag_rank[1]] = 0
    
    # env.load_trained()
    
    print(tempmask.sum(), "tempmasksum")
    
    ###Apply the mask on chosen epoch k
    if args.k_epoch == 0:
        env.reset_to_k_0()
        
    elif args.k_epoch == 2:
        env.reset_to_k_2()
        
    elif args.k_epoch == 5:
        env.reset_to_k_5()
        
    elif args.k_epoch == 90:
        env.reset_to_k_90()
        
    else:
        env.reset_to_init_1()

    env.apply_mask(tempmask)

    ###Evaluate to get the reward
    reward = env.forward_pass(args.num_batches)
    if reward > best_acc:
        best_acc = reward
        best_mask = tempmask
        best_episode_count = episode
    
    
    action_reward_buffer.append((action, reward))
    actions, rewards = zip(*action_reward_buffer) # both var are tuple wrapped
    # actions: tuple->tensor
    actions = torch.squeeze(torch.stack(actions)).type(torch.float)

    agent.update_policy(reward, action, log_prob) 
    obj_func = (-log_prob*reward).sum()
    print("obj_unc", obj_func)
    amount_pruned = 960-tempmask.sum()
    print("amount_pruned", amount_pruned)
    print("acc", reward)

    ###Check the amount per layer
    ###Record the per layer_mask
    layer_mask = []
    num_per_layer = []
    for module in env.model.modules():
        # for conv2d obtain the filters to be kept.
        if isinstance(module, nn.BatchNorm2d):
            weight_copy = module.weight.data.clone()
            filter_mask = weight_copy.gt(0.0).float()
            layer_mask.append(filter_mask)

    for i, item in enumerate(layer_mask):
        num_per_layer.append(int(item.sum().item()))

    print("Filters per layer:", num_per_layer)
    print("Total", sum(num_per_layer))
    print("accuracy", reward)
    
    
    writer.add_scalar('Accuracy_vs_Episode', reward, episode)
    writer.add_scalar('Obj_func_vs_episode', obj_func, episode)





print(best_acc, "Bestacc")

writer.close()
elapsed_time = time.time() - start_time
logging.info("Total elapsed time: {}".format( 
             time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    

###Check best_eval_accuracy
if args.k_epoch == 0:
    env.reset_to_k_0()
    
elif args.k_epoch == 2:
    env.reset_to_k_2()
    
elif args.k_epoch == 5:
    env.reset_to_k_5()
    
elif args.k_epoch == 90:
    env.reset_to_k_90()
    
else:
    env.reset_to_init_1()

env.apply_mask(best_mask)
final_acc = env._evaluate_model()
print(final_acc)




###Apply to initialization and save
env.reset_to_init_1()
env.apply_mask(best_mask)
##Train the final to compare with the unpruned model
###Save into .pth
if args.method == "SA":
    from_string = ".pth"
    to_string = "_pruned.pth"
    
elif args.method == "rand":
    from_string = "_rand.pth"
    to_string = "_rand_pruned.pth"
    
elif args.method == "mag_rewind":
    from_string = "_mag_rewind.pth"
    to_string = "_mag_rewind_pruned.pth"
    
elif args.method == "mag_sign_rewind":
    from_string = "_mag_sign_rewind.pth"
    to_string = "_mag_sign_rewind_pruned.pth"
    
elif args.method == "RL":
    from_string = "_RL.pth"
    to_string = "_RL_pruned.pth"

PATH = (
    os.getcwd()
    + "/masked_may_exp/exp"
    + "_"
    + str(xp_num_)
    + "_"
    + str(int(args.ratio_prune*100))
    + "_"
    + str(args.method)
    + ".pth"
)
model_dicts = {
    "state_dict": env.model.state_dict(),
    "optim": env.optimizer.state_dict(),
    "filters_per_layer": num_per_layer,
    "episode_found": best_episode_count,
    "mask_applied": best_mask,
}
torch.save(model_dicts, PATH)

###Log k_mask_accuracy, as well as 
log_file = open(
    "textlogs/exp_"
    + str(xp_num_)
    + "_sparsity_"
    + str(int(args.ratio_prune*100))
    + ".txt", "w"
)
log_file.write(str("evaluated_accuracy_best_mask: " + str(final_acc) + "\n"))
log_file.write(str("forwardpass_accuracy_best_mask: " + str(best_acc) + "\n"))



#15 is with log of total/pruned
#1 march 13 is with amount pruned to be increased (works as intended)
#2 is with amount pruned to be decreased
#3 is still amount to be decreased  but wt threshold set to 0.5
#4 threshold at 0.5 and -amount_pruned as reward same with [5,

#2 is with 200 episodes basline
#3 is with accuracy only
#4 is with 485
#5 is with 485 and 1 epoch of training
#6 is idk
#7 is with inverted reward and 0.49
#8 is with (3+acc) reward
#9 is with (3+acc)*np.log(flops_remain) and 0.49
#10 is same as 9 but without the grads in state rep
#11 is (3+acc)*(total/amount pruned) without grads in state rep
#12 is with the inverse of 11 since 11 forces amount prund to go up