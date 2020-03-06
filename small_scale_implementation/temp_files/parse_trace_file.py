# get 8 conv2d duration per iteration
import json
import numpy as np
import matplotlib.pyplot as plt

def get_time(filename='tracing.json', num_of_convs=8):
    ''' returns average fwd pass time of conv2d layers '''

    with open(filename) as js:
        json_file = js.readline() 
        json_obj = json.loads(json_file)
        
        durs = [] #durations in ms
        for dictio in json_obj:
            if (dictio['name'] == 'conv2d') & \
                (dictio['pid'] == 'CUDA functions') & \
                 (dictio['ph'] == 'X'):

                durs.append(dictio['dur'])
        np_durs = np.array(durs)
        # reshape and takaway first iteration
        np_durs = np_durs.reshape((-1,num_of_convs))[1:] 
        #legend = []
        #for i in range(num_of_convs):
        #    plt.plot(np_durs[:,i],'o-') # exec time of each layer across iters
        #    legend.append(i)
        #plt.legend(legend)
        #plt.show()
        return np_durs.mean(axis=0)

        
            

        

