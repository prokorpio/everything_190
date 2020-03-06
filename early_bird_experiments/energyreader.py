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
    scaledline = scaledline/torch.max(scaledline)
    #scaledline = scaledline + 1 #range is 1 to 2
    return scaledline
