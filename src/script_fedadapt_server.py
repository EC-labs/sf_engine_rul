import time
import torch
import pickle
import argparse
import logging
import torchvision
import torchvision.transforms as transforms

from functools import partial

import config

from distributed_learning import utils
from distributed_learning.server import SplitFedServer


logger = logging.getLogger(__name__)

LR = config.LR
split_layer = config.split_layer

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root=config.dataset_path, 
    train=False,
    download=True, 
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4
)

logger.info('Preparing Server.')
neural_network_unit = utils.get_model(
    'Unit', 'VGG5', 'cpu', config.model_cfg, config.model_len-1
)
nn_server_creator = partial(
    utils.get_model, 'Server', 'VGG5', 'cpu', config.model_cfg
)
server = SplitFedServer(
    '0.0.0.0', config.SERVER_PORT, neural_network_unit, torch.optim.SGD,
    torch.nn.CrossEntropyLoss(), nn_server_creator, split_layer
)
server.optimizer(lr=LR, momentum=0.9)
server.listen()

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
    logger.info(f"Epoch {r}")
    server.train()

    test_acc = server.test(testloader)
    res['test_acc_record'].append(test_acc)

    with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
        pickle.dump(res,f)

