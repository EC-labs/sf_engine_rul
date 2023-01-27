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

index = 0
datalen = 10
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Create Client')
neural_client, training_partitions = CreatorCNNEngine.create_model_datasets(split_layer)
dataset_train, dataset_valid = training_partitions["train"], training_partitions["validation"]
dataloader_train = DataLoader(dataset_train, batch_size=config.B, shuffle=True)
dataloader_validation = DataLoader(dataset_valid, batch_size=config.B, shuffle=False)

client = SplitFedClient(
    config.SERVER_ADDR, config.SERVER_PORT, 'VGG5', split_layer, 
    torch.nn.MSELoss(), torch.optim.Adam, neural_client
)
client.optimizer(lr=LR)

logger.info('Prepare Data')
cpu_count = multiprocessing.cpu_count()

logger.info("Start Training")
for r in range(config.R):
    logger.info(f'ROUND {r} START')
    training_time = client.train(dataloader_train)

    logger.info("Weights upload")
    client.weights_upload()
    s_time_rebuild = time.time()
    logger.info("Weights receive")
    client.weights_receive()
    e_time_rebuild = time.time()
    logger.info(f'Rebuild time: {e_time_rebuild - s_time_rebuild}')
