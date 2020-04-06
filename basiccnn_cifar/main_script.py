import subprocess as sp 


def mask_prune_train():
    
    criterion_list = ['mag', 'bn_abs', 'grad', 'rand']
    
    #for loop for masking
    for item in criterion_list:
        mask_command = ['python','main_pruner_local.py',
                '--criterion',str(item),
                '--foldername', 'comp_exp_loc_80_cifar',
                '--ratio_prune', '0.8']
        mask_inv_command = ['python','main_pruner_local.py',
                '--criterion',str(item),
                '--foldername', 'comp_exp_loc_80_cifar',
                '--ratio_prune', '0.8',
                '--inv_flag']
        sp.run(mask_command)
        sp.run(mask_inv_command)
    
    
    #for loop for pruning
    prune_command = ['python', 'actual_prune.py',
                '--foldername', 'comp_exp_loc_80_cifar']
    prune_inv_command = ['python', 'actual_prune.py',
                '--foldername', 'comp_exp_loc_80_cifar',
                '--inv_flag']
    sp.run(prune_command)
    sp.run(prune_inv_command)
    
    
    #for loop for training
    for item in criterion_list:
        train_command = ['python', 'train_actual_subnet.py',
                    '--criterion', str(item),
                    '--foldername', 'comp_exp_loc_80_cifar']
        train_inv_command = ['python', 'train_actual_subnet.py',
                    '--criterion', str(item),
                    '--foldername', 'comp_exp_loc_80_cifar',
                    '--inv_flag']
        sp.run(train_command)
        sp.run(train_inv_command)

if __name__ == '__main__':
    mask_prune_train()