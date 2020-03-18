## Contains helper tools 
import time
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format=('%(levelname)s:'+
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import torchvision
import torchvision.transforms as transforms

import models_to_prune
from temp_files.measure_execution_energy import TrackGPUPower


class RandSubnet():
    '''Handles rand-init equivalent of pruned networks.
    '''
    def __init__(self,filter_counts, model_type='basic', layer_count=4):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() \
                                            else 'cpu')
        self.model_type = model_type
        self.model = None   

        # filter count per layer
        self.filter_counts = filter_counts
        # assumes all values will be replaced before calling build

    def build(self): 
        '''Builds the rand-init-ed subnet. 
        '''
        if self.model_type.lower() == 'basic' :
            self.model = models_to_prune.RandBasicCNN(self.filter_counts)
            self.model = self.model.to(self.device)
            self.optimizer = optimizer = optim.SGD(self.model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4)
        else:
            print('model not available') #TODO: use proper handling
            return -1
    
    def train_model(self, train_dl, num_epochs=0):

        # set training parameters 
        #TODO: make this an input, to be set same to env's
        loss_func = nn.CrossEntropyLoss()

        self.model.train()

        logging.info('Training RandCNN model')
        for epoch in range(num_epochs):
            train_acc = []
            start_time = time.time()
            for idx, train_data in enumerate(train_dl):
                inputs, labels = train_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                # forward
                preds = self.model(inputs) # forward pass
                loss = loss_func(preds,labels) # compute loss
                
                # backward
                loss.backward()  # compute grads
                
                self.optimizer.step() # update params w/ Adam update rule

                # print accuracy
                _, prediction = torch.max(preds, dim=1) # idx w/ max val is
                                                        # most confident class
                train_acc.append((prediction==labels).type(torch.double).mean())

                if (idx+1) % 2 == 0:
                    elapsed_time = time.time() - start_time
                    str_time = time.strftime("%H:%M:%S", 
                                             time.gmtime(elapsed_time))
                    # print(('Epoch [{}/{}] Step [{}/{}] | ' + 
                           # 'Loss: {:.4f} Acc: {:.4f} Time: {}')
                           # .format(epoch+1, num_epochs, idx+1, 
                                   # len(self.train_dl), 
                                   # loss.item(), train_acc[-1], 
                                   # str_time))
        logging.info('Training Done')

    def evaluate(self, test_dl):
        self.model.eval()
        total = 0 # total number of labels
        correct = 0 # total correct preds

        logging.info('Evaluating RandCNN model''')
        with torch.no_grad():
            for test_data in test_dl:
                inputs, labels = test_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(inputs) #forward pass
                
                _, prediction = torch.max(preds,dim=1)
                total += labels.size(0) # number of rows = num of samples
                correct += (prediction == labels).sum().item() 

        val_acc = float(correct/total)
        
        return val_acc

def extract_feature_map_sizes(model, input_data_shape, device = None):
    ''' Hook to get conv and fc layerwise feat map sizes

        Input:
            `model`: model which we want to get layerwise feature map size.
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `fmap_sizes_dict`: (dict) {layer_name : [feat_size]}

        Source:
            denru01/netadapt/functions.py
    '''
    fmap_sizes_dict = {}
    hooks = []
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() \
                                             else 'cpu')
    model = model.to(device)
    model.eval() # set to evaluation mode, disable dropout etc  

    def _register_hook(submodule):
       # wrapper for hook registration to perform conditions
       
       def _hook(submodule, input, output):
           type_str = submodule.__class__.__name__
           if type_str in ['Conv2d','Linear']: # Linear is just conv w/
                                               # kernel = 1x1     
               module_id = id(submodule) # unique num for the python object
               in_fmap_size = list(input[0].size()) # [B,C,H,W], 1st tuplet
               out_fmap_size = list(output.size()) #[B,C,H,W]
               fmap_sizes_dict[module_id] = {'in_fmap_size':in_fmap_size,
                                             'out_fmap_size':out_fmap_size}

       if (not isinstance(submodule,torch.nn.Sequential) \
                and not isinstance(submodule, torch.nn.ModuleList) \
                and not (submodule == model)):
            # attach hook to submodule & append handle for removal later
            hooks.append(submodule.register_forward_hook(_hook))

    model.apply(_register_hook) # recursively apply fn to all children submods
    _ = model(torch.randn([1,*input_data_shape]).to(device)) # execute hooks

    for hook in hooks:
        hook.remove()   # remove submodule hooks using handle

    return fmap_sizes_dict

def get_network_def_from_model(model, input_data_shape):
    ''' Input: 
            `model`: model we want to get network_def from
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `network_def`: (OrderedDict)
                           keys(): layer name (e.g. model.0.1, feature.2 ...)
                           values(): dict of layer properties 

        Source:
            denru01/netadapt/functions.py
    '''
    network_def = OrderedDict()
    #state_dict_keys = list(model.state_dict().keys()) # ordered layer_names
                                       # of each learnable layer params
    fmap_sizes_dict = extract_feature_map_sizes(model, input_data_shape)

    #for layer_param_name in state_dict_keys:
    for layer_name, layer_module in model.named_modules():
        layer_id = id(layer_module) # for fmap_sizes_dict access
        layer_type_str = layer_module.__class__.__name__

        # only process linear or conv for now (others include BN, ConvTran)
        if layer_type_str in ['Linear']:
            # Populate info
            network_def[layer_name] = {
                'num_in_channels': layer_module.in_features,
                'num_out_channels': layer_module.out_features,
                'kernel_size': (1,1),
                'stride': (1,1),
                'padding': (0,0),
                'groups': 1,
                'in_fmap_size':[1,
                                fmap_sizes_dict[layer_id]['in_fmap_size'][1],
                                1, 
                                1],
                'out_fmap_size':[1,
                                 fmap_sizes_dict[layer_id]['out_fmap_size'][1],
                                 1,
                                 1],
                # each linear element is considered as a channel
                # linear size = [1, in_features]
            }

        if layer_type_str in ['Conv2d']:
            # Populate info
            network_def[layer_name] = {
                'num_in_channels': layer_module.in_channels,
                'num_out_channels': layer_module.out_channels,
                'kernel_size': layer_module.kernel_size,
                'stride': layer_module.stride,
                'padding': layer_module.padding,
                'groups': layer_module.groups,
                'in_fmap_size': fmap_sizes_dict[layer_id]['in_fmap_size'],
                'out_fmap_size': fmap_sizes_dict[layer_id]['out_fmap_size'],
            }

    return network_def

def compute_weights_and_flops(network_def):
    ''' Get number of flops and weights of entire network 

        Input: 
            `network_def`: defined in get_network_def_from_model()
        
        Output:
            `layer_weights_dict`: (OrderedDict) records layerwise num of weights.
            `total_num_weights`: (int) total num of weights. 
            `layer_flops_dict`: (OrderedDict) recordes layerwise num of FLOPs.
            `total_num_flops`: (int) total num of FLOPs.     
    '''
        
    total_num_weights, total_num_flops = 0, 0

    # Init dict to store num resources for each layer.
    layer_weights_dict = OrderedDict()
    layer_flops_dict = OrderedDict()

    # Iterate over conv layers in network def.
    for layer_name in network_def.keys():
        # Take product of filter size dimensions to get num weights for layer.
        layer_num_weights = (network_def[layer_name]['num_out_channels'] / \
                             network_def[layer_name]['groups']) * \
                            network_def[layer_name]['num_in_channels'] * \
                            network_def[layer_name]['kernel_size'][0] * \
                            network_def[layer_name]['kernel_size'][1]

        # Store num weights in layer dict and add to total.
        layer_weights_dict[layer_name] = layer_num_weights
        total_num_weights += layer_num_weights
        
        # Determine num flops for layer using output size.
        output_size = network_def[layer_name]['out_fmap_size']
        output_height, output_width = output_size[2], output_size[3]
        layer_num_flops = layer_num_weights * output_width * output_height

        # Store num macs in layer dict and add to total.
        layer_flops_dict[layer_name] = layer_num_flops
        total_num_flops += layer_num_flops

    return layer_weights_dict, total_num_weights, layer_flops_dict, total_num_flops

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() \
                          else 'cpu')
#    model = models_to_prune.BasicCNN().to(device)
#    print(compute_weights_and_flops(get_network_def_from_model(model,[3,32,32])))

# Following code used these references:
# kuangliu/pytorch-cifar
# hyang1990/energy_constrained_compression

    # import dataset
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.getcwd(), 
                                            train=True, 
                                            download=True, 
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=128, 
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=os.getcwd(), 
                                           train=False, 
                                           download=True, 
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=100, 
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # import model, curled from kuangliu
    device = torch.device("cuda:0" if torch.cuda.is_available() \
                                    else "cpu")
    from temp_files.vgg import VGG
    net = VGG('VGG16').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, 
                          momentum=0.9, weight_decay=5e-4)
    power_tracker = TrackGPUPower()

    train = True
    if train:
        # start energy tracking
        power_tracker.start()

        # start time tracking
        time_per_epoch = []
        # train for N epoch
        for epoch in range(10):
            net.train()
            total = 0
            correct = 0
            train_loss = 0
            start_time = time.time()
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                preds = net(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = preds.max(dim=1) # outs (val,idx)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 100 == 0:
                    print("Batch %d | Loss: %.3f Acc: %.3f" % \
                            (batch_idx, train_loss/(batch_idx+1), correct/total))

            time_per_epoch.append(time.time() - start_time) # end time tracking
        # end energy tracking
        power_tracker.end()

    # parse power measurements
    joule_per_sec = []
    with open(power_tracker.filename) as fp:
        for line in fp:
            joule_per_sec.append(float(line[:-3])) # get rid of ' W\n'

    see_plots = True
    if see_plots:
        import matplotlib.pyplot as plt
        #plt.subplot(2,1,1)
        plt.plot(joule_per_sec)
        plt.ylabel('J / sec')

        #plt.subplot(2,1,2)
        #plt.plot(time_per_epoch)
        #plt.ylabel('time / epoch')
        plt.show()

    # divide into three parts, get DC power of middle part (J/s)
    third = int(len(joule_per_sec)/3)
    DC_joule_per_sec = sum(joule_per_sec[third:2*third]) / third

    # get energy_per_epoch = DC_power * sec/epoch
    sec_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    energy_per_epoch = DC_joule_per_sec * sec_per_epoch 

    print('sec/epoch', sec_per_epoch)
    print('J/sec', DC_joule_per_sec)
    print('J/epoch', energy_per_epoch)

