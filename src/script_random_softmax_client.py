import torch
import time
import multiprocessing
import logging
import sys

from torch.utils.data import DataLoader

import config

from distributed_learning import utils
from distributed_learning.client import SplitFedClient
from models.turbofan import CreatorCNNEngine 


logger = logging.getLogger(__name__)
logger.propagate = False
handler_console = logging.StreamHandler(stream=sys.stdout)
format_console = logging.Formatter('%(asctime)s [%(levelname)s]: %(name)s : %(message)s')
handler_console.setFormatter(format_console)
handler_console.setLevel(logging.DEBUG)
logger.addHandler(handler_console)

split_layer = config.split_layer
LR = config.LR

logger.info('Prepare Data')
cpu_count = multiprocessing.cpu_count()
creator = CreatorCNNEngine()
neural_client, training_partitions = creator.create_model_datasets(split_layer)
dataset_train, dataset_validate = training_partitions["train"], training_partitions["validation"]
dataloader_train = DataLoader(
    dataset_train, batch_size=config.B, shuffle=True, num_workers=cpu_count
)
dataloader_validate = DataLoader(
    dataset_validate, batch_size=config.B, shuffle=False, num_workers=cpu_count
)

logger.info('Create Client')
client = SplitFedClient(
    config.SERVER_ADDR, 
    config.SERVER_PORT, 
    'VGG5', 
    split_layer, 
    torch.nn.MSELoss(), 
    torch.optim.Adam, 
    neural_client,
    creator.nn_unit_create(None),
    dataloader_validate=dataloader_validate,
)
client.optimizer(lr=LR)

logger.info("Start Training")
for r in range(config.R):
    logger.info(f'ROUND {r} START')
    training_time = client.train(dataloader_train)
    client.aggregate("validation_softmax")
    client.validate(dataloader_validate)
