# everything_190
this repo contains all our codes for EEE 190

## Table of Contents
1. [small_scale_implementation/](small_scale_implementation) : RL pruner implementation on a 4-layer CNN
  >1. [REINFORCE_agent.py](small_scale_implementation/REINFORCE_agent.py) : the RL agent
  >2. [environment.py](small_scale_implementation/environment.py) : ties the agent and the models to be pruned
  >3. [model.py](small_scale_implementation/model.py) : Contains modules used by RL agent
  >4. [models_to_prune.py](small_scale_implementation/models_to_prune.py) : Various CNN modules to run pruning algo onto
  >5. [main.py](small_scale_implementation/main.py) : Main algorithm utilizing subroutines from above
  >6. [temp_files/](small_scale_implementation/temp_files) : necessary parts needing further iterative work
  >7. [small_scale_implementation/best_snapshot_78.pt](best_snapshot_78.pt) : fine-tuned model of the 4-layer CNN
  >8. [small_scale_implementation/rat11.py](rat11.py) : sample code that evaluates the fine-tuned model, prunes a layer, the re-evaluates
2. [labrat_reyes/](labrat_reyes)
3. [misc/](misc) : Various unused files
  >1. [CompressionImageNetLinks.txt](misc/CompressionImageNetLinks.txt)
  >2. [SNIP_reimplementation.ipynb](misc/SNIP_reimplementation.ipynb)
