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

# env.load_trained()
env._evaluate_model()
amount_to_prune = 16
layer_to_prune_amt = 128 #this starts at 0
layer_to_prune = 1
indices = torch.cat((torch.zeros(amount_to_prune).to(device),torch.ones(layer_to_prune_amt-amount_to_prune).to(device)),0)
indices = indices.unsqueeze(0)
env.layer_to_process = 'conv2'
env.prune_layer(indices)


env._evaluate_model()

print(env.model.state_dict())
env._train_model(num_epochs = 5)
env._evaluate_model()
print(env.model.state_dict())