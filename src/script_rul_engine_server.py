import time
import torch
import pickle
import logging

import config

from distributed_learning.server import SplitFedServer
from models.turbofan import CreatorCNNEngine


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LR = config.LR
split_layer = config.split_layer

logger.info('Preparing Server.')
nn_unit = CreatorCNNEngine.nn_unit_create()
nn_server_creator = CreatorCNNEngine.nn_server_create
server = SplitFedServer(
    '0.0.0.0', config.SERVER_PORT, nn_unit, torch.optim.Adam,
    torch.nn.MSELoss(), nn_server_creator, split_layer
)
server.optimizer(lr=LR)
server.listen()
time.sleep(20)

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
    logger.info(f"Epoch {r}")
    server.train()

    with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
        pickle.dump(res,f)
