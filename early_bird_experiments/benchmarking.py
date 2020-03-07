import subprocess as sp
import os

#Run main.py to find the earlybird
	#base .sh is search2.sh
	#Verify that the correct "an" is used

#Run BasicCNNprune2.py to actually prune the earlybird found.
	#Base .sh is pruneBasicCNN2.sh
	#Needs the location of the EarlyBirdTicket found in main.py
	#Saves result to args.save in such a way that you can refer to it in the next shell script

#Run main_c.py to train the pruned models to convergence
	#Base .sh is retrain_continue2.sh
	#Needs the location of actually pruned model found in BasicCNNprune2.py (previous args.save)
	#Saves result to args.save
	#Get time per epoch at the end.

def find_prune_train(arch = 'BasicCNN2',
                     depth = 4,
                     file_path = './baseline/',
                     trial_num = 0,
                     timed = False):

    # Define paths
    file_path = file_path + \
                arch + str('_'+str(depth)) + \
                '/trial_' + str(trial_num)

    EB_folder = os.path.join(file_path,'found_EBs')
    timed_EB_folder = os.path.join(file_path+'_timed','found_EBs')

    pruned_EB_folder = os.path.join(file_path,'pruned_EBs')
    timed_pruned_EB_folder = os.path.join(file_path+'_timed','pruned_EBs')

    trained_EB_folder = os.path.join(file_path,'trained_EBs')
    timed_trained_EB_folder = os.path.join(file_path+'_timed','trained_EBs')

    # Run main.py to find the earlybird
    print('\nFinding EB\n')

    find_EB_cmd = ['python', 'main.py',
                    '--dataset','cifar10',
                    '--lr','0.08',
                    '--epochs','160',
                    '--schedule',' 80','120',
                    '--batch-size','256',
                    '--test-batch-size','256',
                    '--momentum','0.9',
                    '--sparsity-regularization',
                    '--arch' , arch,
                    '--depth' , str(depth),
                    '--save', EB_folder]

    timed_find_EB_cmd = find_EB_cmd[:-2] \
                        + ['--save',timed_EB_folder] \
                        + ['--timerank']

    if timed:
        print('Running:\n ' + ' '.join(timed_find_EB_cmd))
        sp.run(timed_find_EB_cmd)
    else:
        print('Running:\n ' + ' '.join(find_EB_cmd))
        sp.run(find_EB_cmd)

    # Run pruneBasicCNN2.py or vggprune.py to get compressed model
    print('\nPruning for EB\n')

    extract_EB_cmd = ['python', '_.py', # will be changed later
                    '--dataset','cifar10',
                    '--test-batch-size','256',
                    '--gpu_ids', '0',
                    '--depth' , str(depth),
                    '--percent', '_',
                    '--save', '_',
                    '--model', '_'] # will add path to which model later 

    EB_names = ['EB-30.pth.tar', 'EB-50.pth.tar', 'EB-70.pth.tar']

    if arch == 'BasicCNN2':
        extract_EB_cmd[1] = 'BasicCNNprune2.py' 
        if timed:
            for EB in EB_names:
                extract_EB_cmd[-5] = '0.' + EB[3]
                extract_EB_cmd[-3] = os.path.join(timed_pruned_EB_folder,EB[:5])
                extract_EB_cmd[-1] = os.path.join(timed_EB_folder,EB)
                print('Running:\n ' + ' '.join(extract_EB_cmd))
                sp.run(extract_EB_cmd)
        else:
            for EB in EB_names:
                extract_EB_cmd[-5] = '0.' + EB[3]
                extract_EB_cmd[-3] = os.path.join(pruned_EB_folder,EB[:5])
                extract_EB_cmd[-1] = os.path.join(EB_folder,EB)
                print('Running:\n ' + ' '.join(extract_EB_cmd))
                sp.run(extract_EB_cmd)
    elif arch == 'vgg':
        extract_EB_cmd[1] = 'vggprune.py' # assumes depth is already set
        if timed:
            for EB in EB_names:
                extract_EB_cmd[-5] = '0.' + EB[3]
                extract_EB_cmd[-3] = os.path.join(timed_pruned_EB_folder,EB[:5])
                extract_EB_cmd[-1] = os.path.join(timed_EB_folder,EB)
                print('Running:\n ' + ' '.join(extract_EB_cmd))
                sp.run(extract_EB_cmd)
        else:
            for EB in EB_names:
                extract_EB_cmd[-5] = '0.' + EB[3]
                extract_EB_cmd[-3] = os.path.join(pruned_EB_folder,EB[:5])
                extract_EB_cmd[-1] = os.path.join(EB_folder,EB)
                print('Running:\n ' + ' '.join(extract_EB_cmd))
                sp.run(extract_EB_cmd)

    # Run main_c.py to train the pruned models to convergence
    print('\nTraining EB\n')

    train_EB_cmd = ['python', 'main_c.py', 
                    '--dataset', 'cifar10', 
                    '--arch', arch, 
                    '--depth', str(depth), 
                    '--lr','0.1', 
                    '--epochs', '160', 
                    '--schedule', '80', '120', 
                    '--batch-size', '256', 
                    '--test-batch-size', '256', 
                    '--momentum', '0.9', 
                    '--sparsity-regularization', 
                    '--save',  '_',  # to be changed later
                    '--start-epoch', '_',  # to be changed later
                    '--scratch', '_'] # to be changed later

    # note: ave_time is in last element of time.txt
    #       best perf is in last element of record.txt
    #       model param new and old is in prune.txt in prune folder
    if timed:
        for EB in EB_names:
            # Get name of pruned model
            pruned_model = os.path.join(timed_pruned_EB_folder,EB[:5],'pruned.pth.tar')
            # Get epoch for --start-epoch arg
            for line in open(os.path.join(timed_EB_folder,'EB_epoch.txt')):
                if line[:2] == EB[3:5]: 
                    start_epoch = line[-2:-1] 
                    break
            train_EB_cmd[-5] = os.path.join(timed_trained_EB_folder,EB[:5])
            train_EB_cmd[-3] = start_epoch
            train_EB_cmd[-1] = pruned_model
            print('Running:\n ' + ' '.join(train_EB_cmd))
            sp.run(train_EB_cmd)
    else:
        for EB in EB_names:
            # Get name of pruned model
            pruned_model = os.path.join(pruned_EB_folder,EB[:5],'pruned.pth.tar')
            # Get epoch for --start-epoch arg
            for line in open(os.path.join(EB_folder,'EB_epoch.txt')):
                if line[:2] == EB[3:5]: 
                    start_epoch = line[-2:-1] 
                    break
            train_EB_cmd[-5] = os.path.join(trained_EB_folder,EB[:5])
            train_EB_cmd[-3] = start_epoch
            train_EB_cmd[-1] = pruned_model
            print('Running:\n ' + ' '.join(train_EB_cmd))
            sp.run(train_EB_cmd)



# =========================== MAIN LOOP ============================

if __name__ == '__main__':
    num_of_trials = 2
    models = ['BasicCNN2', 'vgg']
    model_depths = [4,16]

    #find_prune_train(arch='vgg', depth=16, timed=True) 

    for t in range(num_of_trials): 
        print('t = ',t)
        for m in range(len(models)):
            print('m = ',m)
            print('timed')
            find_prune_train(arch=models[m],
                             depth=model_depths[m],
                             trial_num=t,
                             timed=True)
            print('not_timed')
            find_prune_train(arch=models[m],
                             depth=model_depths[m],
                             trial_num=t,
                             timed=False)

    # TODO: collect all info for table


