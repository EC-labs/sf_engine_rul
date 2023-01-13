import socket
import time
import multiprocessing
import argparse
import logging
import sys

from .client import Client
import config
import utils


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=utils.str2bool, default=False)
args = parser.parse_args()

index = 0
datalen = 10
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(config.SERVER_ADDR, config.SERVER_PORT, datalen, 'VGG5', split_layer)

offload = args.offload
client.initialize(split_layer, offload, True, LR)

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

if offload:
    logger.info('FedAdapt Training')
else:
    logger.info('Classic FL Training')

flag = False # Bandwidth control flag.

for r in range(config.R):
    logger.info('====================================>')
    logger.info('ROUND: {} START'.format(r))

    training_time = client.train(trainloader)
    logger.info('ROUND: {} END'.format(r))
    
    logger.info('==> Waiting for aggregration')
    client.upload()

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    s_time_rebuild = time.time()
    if offload:
        config.split_layer = client.recv_msg(client.sock)[1]

    if r > 49:
        LR = config.LR * 0.1

    client.reinitialize(config.split_layer[index], offload, False, LR)
    e_time_rebuild = time.time()
    logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
    logger.info('==> Reinitialization Finish')
