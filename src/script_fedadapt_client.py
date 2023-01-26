import torch
import time
import multiprocessing
import logging
import sys

from distributed_learning.client import SplitFedClient
import config
import utils


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
neural_client = utils.get_model('Client', 'VGG5', 'cpu', config.model_cfg, split_layer)
client = SplitFedClient(
    config.SERVER_ADDR, config.SERVER_PORT, 'VGG5', split_layer, 
    torch.nn.CrossEntropyLoss(), torch.optim.SGD, neural_client
)
client.optimizer(lr=LR, momentum=0.9)

logger.info('Prepare Data')
cpu_count = multiprocessing.cpu_count()
trainloader = utils.get_local_dataloader(index, cpu_count)

logger.info("Start Training")
for r in range(config.R):
    logger.info(f'ROUND {r} START')
    training_time = client.train(trainloader)

    logger.info("Weights upload")
    client.weights_upload()
    s_time_rebuild = time.time()
    logger.info("Weights receive")
    client.weights_receive()
    e_time_rebuild = time.time()
    logger.info(f'Rebuild time: {e_time_rebuild - s_time_rebuild}')
