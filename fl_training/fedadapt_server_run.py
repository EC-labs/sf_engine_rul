import time
import torch
import pickle
import argparse
import logging
import sys
import os

import config
import utils
import PPO
from .server import Server


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=utils.str2bool, default=False)
args=parser.parse_args()

LR = config.LR
offload = args.offload

logger.info('Preparing Server.')
server = Server('0.0.0.0', config.SERVER_PORT, 'VGG5', offload, LR)

if offload:
    state_dim = 2*config.G
    action_dim = config.G
    agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
    agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))
    logger.info('FedAdapt Training')
else:
    logger.info('Classic FL Training')

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
    logger.info('====================================>')
    logger.info('==> Round {:} Start'.format(r))

    s_time = time.time()
    bandwidth = server.train()
    aggregrated_model = server.aggregate()
    e_time = time.time()

    # Recording each round training time, bandwidth and test accuracy
    training_time = e_time - s_time
    res['training_time'].append(training_time)
    res['bandwidth_record'].append(bandwidth)

    test_acc = server.test(r)
    res['test_acc_record'].append(test_acc)

    with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
        pickle.dump(res,f)

    logger.info('Round Finish')
    logger.info('==> Round Training Time: {:}'.format(training_time))

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    if offload:
        split_layers = server.adaptive_offload(agent, state)
    else:
        split_layers = config.split_layer

    if r > 49:
        LR = config.LR * 0.1

    server.reinitialize(split_layers, offload, False, LR)
    logger.info('==> Reinitialization Finish')

