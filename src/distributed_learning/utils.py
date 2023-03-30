import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models.vgg import VGG
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

def get_local_dataloader(CLIENT_IDEX, cpu_count):
    indices = list(range(N))
    part_tr = indices[int((N/K) * CLIENT_IDEX) : int((N/K) * (CLIENT_IDEX+1))]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transform_train
    )
    subset = Subset(trainset, part_tr)
    trainloader = DataLoader(
        subset, batch_size=B, shuffle=True, num_workers=cpu_count
    )
    return trainloader

def get_model(location, model_name, device, cfg, layer) ->  nn.Module:
    """
    Get the pytorch NN model. 

    Returns: 
        torch.nn.Module
    """
    cfg = cfg.copy()
    net = VGG(location, model_name, layer, cfg)
    net = net.to(device)
    logger.debug(str(net))
    return net

def split_weights_client(weights,cweights):
    for key in cweights:
        assert cweights[key].size() == weights[key].size()
        cweights[key] = weights[key]
    return cweights

def split_weights_server(weights, sweights):
    skeys = list(sweights)
    keys = list(weights)
    end = len(weights)
    start = end - len(sweights)
    for sidx, widx in enumerate(range(start, end)):
        assert sweights[skeys[sidx]].size() == weights[keys[widx]].size()
        sweights[skeys[sidx]] = weights[keys[widx]]
    return sweights

def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()

    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]

    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

    return concat_dict

def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net

def fed_avg(zero_model, w_local_list):
    keys = w_local_list[0][0].keys()
    
    for k in keys:
        for w in w_local_list:
            if 'num_batches_tracked' in k:
                zero_model[k] = w[0][k]
            else:
                zero_model[k] += (w[0][k] * w[1])

    return zero_model
