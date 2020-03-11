import csv
import torch

def energyreader(path):

    with open(path, newline = '', encoding = 'utf-8') as csvfile:
        data = csv.reader(csvfile)
        for item in data:
            line = item
            break
    return line
    
def minmaxer(line):
    scaledline = list(map(float, line))
    scaledline = torch.as_tensor(scaledline)

    #scaledline = scaledline - torch.min(scaledline)
    #scaledline = scaledline/torch.max(scaledline)
    #scaledline = scaledline/4
    #scaledline = scaledline + 0.75
    #scaledline = scaledline/torch.max(scaledline)
    #scaledline = scaledline - torch.min(scaledline)
    #scaledline = torch.exp(scaledline) / torch.exp(scaledline).sum()

    # direct relative percentage (just to maintain normalized proportion)
    # since apparently, softmax is also magnitude dependent (not just rel mag)
    # see softmax([1,2]) vs softmax([10,20])
    #scaledline = scaledline / scaledline.sum()

	# original version of [0,1] but with the limiter in place
    scaledline = scaledline - torch.min(scaledline)
    scaledline = scaledline/torch.max(scaledline)
    return scaledline
