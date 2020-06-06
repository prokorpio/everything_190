import subprocess as sp 


def mask_prune_train(xp_num_, ratio_prune, method_list, k_epoch):


    
    for method in method_list:
    
        if method == "SA":
            mask_py = "main_SA.py"
        elif method == "rand":
            mask_py = "main_pruner_random.py"
        elif method == "mag_rewind":
            mask_py = "main_pruner_mag_rewind.py"
        elif method == "mag_sign_rewind":
            mask_py = "main_pruner_mag_sign_rewind.py"
        elif method == "RL":
            mask_py = "main.py"
            
        mask_command = ['python', str(mask_py),
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--k_epoch', str(k_epoch),
                '--ratio_prune', str(ratio_prune)]
        print(mask_command, "MASKCOMMAND")
        sp.run(mask_command)
        
        

        prune_command = ['python', 'actual_prune.py',
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--ratio_prune', str(ratio_prune)]
        print(prune_command, "PRUNECOMMAND")       
        sp.run(prune_command)
        

        train_command = ['python', 'train_actual_subnet.py',
                '--xp_num_' , str(xp_num_),
                '--method' , str(method),
                '--ratio_prune', str(ratio_prune)]
        print(train_command, "TRAINCOMMAND")
        sp.run(train_command)

if __name__ == '__main__':

    xp_num_list = [500,505,510,515]
    ratio_prune_list = [0.5, 0.6, 0.7, 0.8]
    method_list = ["RL"]
    for xp_num_, ratio_prune in zip(xp_num_list, ratio_prune_list):

        #Do the routine for RL
        mask_prune_train(xp_num_, ratio_prune, method_list, -1)
        
        #Do the routine for RL with modified epoch k
        mask_prune_train(xp_num_+1, ratio_prune, method_list, 0)#Try k = 0
        mask_prune_train(xp_num_+2, ratio_prune, method_list, 2)#try k = 2
        mask_prune_train(xp_num_+3, ratio_prune, method_list, 5)#Try k = 5
        mask_prune_train(xp_num_+4, ratio_prune, method_list, 90)#try k = 90
        
