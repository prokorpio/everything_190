from __future__ import print_function
import os, time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from filter import *
from scipy.ndimage import filters
from compute_flops import print_model_param_flops


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='BasicCNN2', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=4, type=int,
                    help='depth of the neural network')
parser.add_argument('--scratch',default='', type=str,
                    help='the PATH to the pruned model')
# filter
parser.add_argument('--filter', default='none', type=str, choices=['none', 'lowpass', 'highpass'])
parser.add_argument('--sigma', default=1.0, type=float, help='gaussian filter hyper-parameter')

# sparsity
parser.add_argument('--sparsity_gt', default=0, type=float, help='sparsity controller')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--port", type=str, default="15000")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
    id = int(gpu_id)
    args.gpu_ids.append(id)
print(args.gpu_ids)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {} # jeff: changed 
                                                                     # numwork 1->0
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.dataset == 'imagenet':
    model = models.__dict__[args.arch](pretrained=False)
    if args.scratch:
        checkpoint = torch.load(args.scratch)
        if args.dataset == 'imagenet':
            cfg_input = checkpoint['cfg']
            model = models.__dict__[args.arch](pretrained=False, cfg=cfg_input)
    if args.cuda:
        model.cuda()
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids, find_unused_parameters=True)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    if args.cuda:
        model.cuda()
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids, find_unused_parameters=True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)



import copy
def fake_train(epoch):
    ''' modified train just to take layer fwd and bwd runtimes'''
    model_replica = copy.deepcopy(model).cuda()
    model_replica.train()
    #global history_score
    #avg_loss = 0.
    #train_acc = 0.
    # start_time = time.time()
    #end_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print('data load time: ', time.time()-end_time)
        # data_time = time.time()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_replica(data)
        loss = F.cross_entropy(output, target)
        #avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        #prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        #train_acc += prec1.item()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        print('fake train:',batch_idx)
        if batch_idx == 700:
            break


# jeff: record fake_train exec times 20
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    fake_train(1)
prof.export_chrome_trace('tracing1.json')
print('Done')
import numpy
import parse_trace_file as ptf
time_per_layer = ptf.get_time(num_of_convs=4) #16 depends on which model
time_per_layer = numpy.asarray(time_per_layer)
numpy.savetxt("energy.csv",time_per_layer, delimiter = ",")
print(time_per_layer)

exit()
