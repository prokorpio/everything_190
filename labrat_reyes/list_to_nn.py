slf_list = [20,20,10]


def createmodel(slf_list,inputsize,device):
    fclist = [] 
    #idea from amc
    for i, size in enumerate(slf_list):
        #First layer
        if i == 0:
            fclist.append(nn.Linear(inputsize, size))
        else:
            fclist.append(nn.Linear(slf_list[i-1], size))

    net = nn.Sequential(*fclist).to(device)
    print(net)
    return net