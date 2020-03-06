'''VGG11/13/16/19 in Pytorch.'''
'''With hook options for measuring layer energy'''
import torch
import torch.nn as nn

import config_globals as globe
import numpy as np
import time


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class MeasureLayerPower(nn.Module):
    ''' Fake layer to initiate or end GPU tracking per layer'''
    def __init__(self, type_='begin'):
        super(MeasureLayerPower, self).__init__()
        self.begin = {'begin':True, 'end':False}[type_]

    def forward(self, x):
        globe.start_times.append(time.time())
        #if self.begin: # starts GPU power tracking
        #    print('Starts GPU power tracker')
        #    globe.start_times.append(time.time())
        #else:
        #    print('Ends GPU power tracker')
        #    globe.end_times.append(time.time())
        #    delta = globe.end_times[-1] - globe.start_times[-1] 
        #    print('Time passed:', delta)
        return x

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.starter = MeasureLayerPower('begin')
        #self.ender = MeasureLayerPower('end')

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.starter(out) 
        out = self.classifier(out)
        out = self.starter(out)

        # save & flush global times contents per forward pass
        globe.iter_layertimes.append(globe.start_times.copy())
        del globe.start_times[:]
        
        print('forward done')
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [MeasureLayerPower('begin'),
                           nn.MaxPool2d(kernel_size=2, stride=2)]
                           #MeasureLayerPower('end')]
            else:
                layers += [MeasureLayerPower('begin'),
                           nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                           #MeasureLayerPower('end')]
                in_channels = x
        layers += [MeasureLayerPower('begin'),
                   nn.AvgPool2d(kernel_size=1, stride=1)]
                   #MeasureLayerPower('end')]

        return nn.Sequential(*layers)

# With fake layers:
# define energy layer, options: start/end
# add options with _make_layers: energy=True/False
# add start/end for Maxpool
# add start meas and end meas every chunk of Conv-BN-ReLu
# add start/end for AvgPool
# add for Linear
from measure_execution_energy import TrackGPUPower
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def test():
    start = time.time()
    net = VGG('VGG11').to(device)
    x = torch.randn(128,3,32,32).to(device)
    #pow_track = TrackGPUPower(time_stamp=True, grain='ms')
    #pow_track.start()
    #time.sleep(0.5)
    y = net(x)
    y = net(x)
    y = net(x)
    print(time.time() - start)
    #time.sleep(0.5)
    #pow_track.end()

# perform gpu warmup here
#a = torch.randn(128,3,32,32).to(device)
#b = torch.randn(128,3,32,32).to(device)
#c = a @ b
#print(c.shape)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    test()
print(prof.key_averages().table(sort_by='cuda_time_total'))
prof.export_chrome_trace('tracing.json')
exit()
# ----------READ POWER READINGS-----------

from datetime import datetime 
strptime = datetime.strptime

pow_per_ms = {'pow':[] , 'ms':[]}
fmt = '%Y/%m/%d %H:%M:%S.%f'
with open('power.txt') as fp:
    for line in fp:
        # ff are vectors
        pow_per_ms['pow'].append(float(line.split()[-2]))
        pow_per_ms['ms'].append(strptime(line.split(',')[0],fmt).timestamp())

# partition power readings into each fwd pass and layers
iter_dc_layer_power = [] # cols: list whose elements are dc pow per layer 
                         # rows: different iters
# turn key:list --> key:np.array
#pow_per_ms['pow'] = np.array(pow_per_ms['pow'])
#pow_per_ms['ms'] = np.array(pow_per_ms['ms'])

# Increase data point resolution by interpolation
x = np.array(pow_per_ms['ms'])
y = np.array(pow_per_ms['pow'])
new_x = np.linspace(x.min(), x.max(), 100*x.size) # increase reso 100x
new_y = np.interp(new_x, x, y) # evaluate at new x

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(x,y,'ro-')

plt.subplot(2,1,2)
plt.plot(new_x,new_y,'bo-')
plt.plot(globe.iter_layertimes[0][0],100,'go-')
plt.plot(globe.iter_layertimes[-1][-1],100,'go-')
plt.show()

count_of_missing = 0
np.set_printoptions(precision=12)
for iter_ in range(len(globe.iter_layertimes)):
# partition per fwd pass iter
    start_time = globe.iter_layertimes[iter_][0] # start time of this iteration
    end_time = globe.iter_layertimes[iter_][-1]
    powers = new_y[(new_x >= start_time) & (new_x <= end_time)]
    times = new_x[(new_x >= start_time) & (new_x <= end_time)]
    #pow_per_iter.append(list(powers)) 

    # partition each powers withinin this iter into layers
    dc_layer_power = [] # mean of joule/ms within layer exec time
    for layer_ in range(len(globe.iter_layertimes[iter_][:-1])):
        print('iter:',iter_,'layer',layer_)
        start_time = globe.iter_layertimes[iter_][layer_]
        next_start = globe.iter_layertimes[iter_][layer_+1]
        # layer time is start_time to next start_time
        layer_powers = powers[(times >= start_time) & (times <= next_start)]
        if layer_powers.size:
            dc_layer_power.append(layer_powers.mean()) # ave across time
        else:
            print('Need higer reso data!!')
            print(layer_powers)
            count_of_missing += 1
            dc_layer_power.append(0)

    iter_dc_layer_power.append(dc_layer_power)

# get average layer exec pow per fwd pass
ave_layer_power = np.array(iter_dc_layer_power).mean(axis=0) # vector 
#print(iter_dc_layer_power)
print('power per layer:',ave_layer_power)
print('holes in data:',count_of_missing)

# get time delta's
layer_times = np.array(globe.iter_layertimes)
layer_times = layer_times[:,:-1] - layer_times[:,1:]
# get layer's average exec time per fwd pass
ave_layer_times = layer_times.mean(axis=0)
print('time_per_layer:',ave_layer_times)
print('sum time:', ave_layer_times.sum())
                

# jeff: base code curled from kuangliu/pytorch-cifar
