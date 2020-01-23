## Contains helper tools 
import time
import logging
logging.basicConfig(level=logging.INFO,
                    format=('%(levelname)s:'+
                            '[%(filename)s:%(lineno)d]' +
                            ' %(message)s'))

import torch
from collections import deque

import models_to_prune


class RandSubnet():
    '''Handles rand-init equivalent of pruned networks.
    '''

    def __init__(self, model_type='basic', layer_count=4):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() \
                                            else 'cpu')
        self.model_type = model_type
        self.model = None   

        # filter count per layer
        self.filter_counts = deque([], maxlen=layer_count)  
        # assumes all values will be replaced before calling build

    def build(self): 
        '''Builds the rand-init-ed subnet. 
        '''
        if self.model_type.lower() == 'basic' :
            self.model = models_to_prune.RandBasicCNN(self.filter_counts)
            self.model = self.model.to(self.device)
        else:
            print('model not available') #TODO: use proper handling
            return -1
    
    def train_model(self, train_dl, num_epochs=0):

        # set training parameters 
        #TODO: make this an input, to be set same to env's
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(0.0008))

        self.model.train()

        logging.info('Training RandCNN model')
        for epoch in range(num_epochs):
            train_acc = []
            start_time = time.time()
            for idx, train_data in enumerate(train_dl):
                inputs, labels = train_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()

                # forward
                preds = self.model(inputs) # forward pass
                loss = loss_func(preds,labels) # compute loss
                
                # backward
                loss.backward()  # compute grads
                
                optimizer.step() # update params w/ Adam update rule

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

        val_acc = correct/total
        
        return val_acc


#def estimate_layer_flops(self, layer, 
#                               input_x, # input feat map
#                               amount_pruned=0, # amt pruned on current layer 
#                               pruned_prev_layer=0): # amt pruned on prev layer
#    ''' Estimate conv layer flops.
#        Implementation derived from AMC'''
#
#    # count multiply-add as 1 flop
#    multipy_add = 1
#
#    # get layer type
#    layer_str = str(layer)
#    type_name = layer_str[:layer_str.find('(')].strip() # strip off whitespaces
#
#    # estimate Conv flops
#    if type_name in 'Conv2d':
#        input_h = input_x.size()[2]
#        input_w = input_x.size()[3]
#        kernel_h = conv_layer.kernel_size[0]
#        kernel_w = conv_layer.kernel_size[1]
#        pad_h = conv_layer.padding[0]
#        pad_w = conv_layer.padding[1]
#        stride_h = conv_layer.stride[0]
#        stride_w = conv_layer.stride[1]
#        C_in = conv_layer.in_channels - pruned_prev_layer
#        C_out = conv_layer.out_channels - amount_pruned
#        groups = conv_layer.groups
#
#        out_h = (input_h + 2*pad_h - kernel_h)/stride_h + 1  # also filtr steps
#        out_w = (input_w + 2*pad_w - kernel_w)/stride_w + 1  
#        
#        # define new attribute in layer
#        layer.flops = C_out * (C_in/groups) * kernel_h*kernel_w \
#                            * out_h * out_w * multiply_add
#      
#    # estimate Linear flops
#    elif type_name in 'Linear':
#        pass
#
#    return 


        
