''' Script to automate collection of benchmarking data '''

import os
import pandas as pd
import numpy as np

def get_data(base_path = './baseline/', model_list = ['vgg_16']):
    ''' Collect benchmarking data 
     inputs:
        base_path:  path to folder containing model-type_depth folders
        model_list: [list of folders containing model data]

     outputs: 
        ff is a dict({key : 4d numpy array}
        {model-type_depth: np.array([(0,1,2,..,n),    # for trial number
                                     (0, 1, or 2),    # for sparsities, 30,50,70
                                     (0 or 1),        # for not_timed or timed
                                     (acc, time_epoch, EB_epoch)]}
    '''
    # create output: {model-type_depth : 4d np.array}
    out_dict = {}
    for model in model_list:
        model_path = os.path.join(base_path,model)
        trial_dirs = os.listdir(model_path) # get all trial folders
        trial_dirs = [d for d in trial_dirs if 'trial' in d]
        out_dict[model] = np.random.rand(int(len(trial_dirs)/2),3,2,3)*-1
                                        # negative to easily spot error
                                        # 3 sparsities
                                        # 2 EB method types
                                        # 3 perf records
        for tr_dir in trial_dirs:
            # get trial num from tr_dir
            tr_num = int([c for c in tr_dir if c.isdigit()].pop())

            # get if with time_rank or orig
            if 'timed' in tr_dir:
                type_num = 1
            else:
                type_num = 0

            # go to 'trained_EBs' folder
            trained_path = os.path.join(model_path, tr_dir, 'trained_EBs')
            sparsity_dirs = os.listdir(trained_path)
            sparsity_dirs = [d for d in sparsity_dirs if 'EB' in d]
            for sp_dir in sparsity_dirs:
                sp_num = ['30','50','70'].index(sp_dir[3:])
                record_file = os.path.join(trained_path, sp_dir, 'record.txt')
                time_file = os.path.join(trained_path, sp_dir, 'time.txt')
                with open(record_file) as fp:
                    line_list = fp.readlines()
                    acc = float(line_list[-1].split(',')[0])
                with open(time_file) as fp:
                    line_list = fp.readlines()
                    time = float(line_list[-1][:-1]) # remove \n

                out_dict[model][tr_num,sp_num,type_num,0] = acc
                out_dict[model][tr_num,sp_num,type_num,1] = time

            # go to 'EBs_found' folder
            EB_epoch_file = os.path.join(model_path, tr_dir,
                                        'found_EBs','EB_epoch.txt') 
            with open(EB_epoch_file) as fp:  
                for line in fp: # get num epoch EB found
                    sp_num = ['30','50','70'].index(line[:2])
                    epoch = line[8:-1]
                    if epoch.isdigit():
                        out_dict[model][tr_num,sp_num,type_num,2] = int(line[8:-1])

    return out_dict

def dict_to_xl(output,xl_name=None):
    ''' Convert dict output to table'''
    model_list = list(output.keys())
    row_label = ['EB Orig', 'EB Time']
    col_label = ['acc','time/epoch','EB epoch found']
    for model in model_list:
        tr_dict = {'T'+str(trial):0 for trial in range(output[model].shape[0])}
        for tr_idx, trial in enumerate(tr_dict.keys()):
            sp_dict = {'30% Sparsity':0,'50% Sparsity':0,'70% Sparsity':0}
            for sp_idx, sparsity in enumerate(sp_dict.keys()): 
                sp_dict[sparsity] = pd.DataFrame(output[model][tr_idx,sp_idx], 
                                    columns = col_label,
                                    index = row_label)
            tr_dict[trial] = pd.concat(sp_dict, axis=1)

        model_table = pd.concat({model:pd.concat(tr_dict)}, axis=1)
        if xl_name == None:
            model_table.to_excel(os.path.join(os.getcwd(), (model+'.xlsx')))
        else:
            model_table.to_excel(os.path.join(os.getcwd(), (xl_name+'.xlsx')))

if __name__ == '__main__':
    out_dict = get_data(base_path='./baseline/',
                        model_list = ['BasicCNN2_4','vgg_16'])
                        
    dict_to_xl(out_dict)

    # get ave across trials
    #out_ave = out_dict['vgg_16'].mean(axis=0, keepdims=True) 
    #dict_to_xl({'vgg_16_ave':out_ave},'ave')
