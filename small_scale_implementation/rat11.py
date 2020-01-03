###Rat 11 Sample pruning using the environment
###.no_grad() might be needed
from environment import *
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = PruningEnv()
PATH = os.getcwd() + '\\best_snapshot_78.pt'
print("HELLO")
# pretrained_model = torch.load(PATH)
# print(pretrained_model)
# env.model.state_dict() = pretrained_model

env.load_trained()
env._evaluate_model()
amount_to_prune = 16
layer_to_prune_amt = 128 #this starts at 0
layer_to_prune = 1
indices = torch.cat((torch.zeros(amount_to_prune).to(device),torch.ones(layer_to_prune_amt-amount_to_prune).to(device)),0)
env.prune_layer(layer_to_prune, indices, device,1)


env._evaluate_model()