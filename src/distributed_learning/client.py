import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

import config
import utils
from communicator import Communicator
import logging


logger = logging.getLogger(__name__)
logger.propagate = False
handler_console = logging.StreamHandler(stream=sys.stdout)
format_console = logging.Formatter('[%(levelname)s]: %(name)s : %(message)s')
handler_console.setFormatter(format_console)
handler_console.setLevel(logging.DEBUG)
logger.addHandler(handler_console)

np.random.seed(0)
torch.manual_seed(0)

class SplitFedClient:


    def __init__(
        self, server_addr, server_port, datalen, 
        model_name, split_layer, offload, LR
    ):
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
        logger.info('Connecting to Server.')
        self.conn = Communicator()
        self.conn.connect((server_addr, server_port))
        self.initialize(split_layer, offload, True, LR)

    def initialize(self, split_layer, offload, first, LR):
        if offload or first:
            self.split_layer = split_layer
            logger.debug('Building Model.')
            self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, config.model_cfg)
            logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)

        logger.debug('Receiving Global Weights..')
        weights = self.conn.recv_msg()[1]
        if self.split_layer == (config.model_len -1):
            self.net.load_state_dict(weights)
        else:
            pweights = utils.split_weights_client(weights,self.net.state_dict())
            self.net.load_state_dict(pweights)
        logger.debug('Initialize Finished')

    def train(self, trainloader):
        # Training start
        s_time_total = time.time()
        time_training_c = 0
        self.net.to(self.device)
        self.net.train()
        if self.split_layer == (config.model_len -1): # No offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
        else: # Offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)

                msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
                self.conn.send_msg(msg)

                # Server gradients
                gradients = self.conn.recv_msg()[1].to(self.device)

                outputs.backward(gradients)
                self.optimizer.step()

        e_time_total = time.time()
        logger.info('Total time: ' + str(e_time_total - s_time_total))

        training_time_pr = (e_time_total - s_time_total) / int((config.N / (config.K * config.B)))
        logger.info('training_time_per_iteration: ' + str(training_time_pr))

        msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.conn.ip, training_time_pr]
        self.conn.send_msg(msg)

        return e_time_total - s_time_total
        
    def upload(self):
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
        self.conn.send_msg(msg)

    def reinitialize(self, split_layers, offload, first, LR):
        self.initialize(split_layers, offload, first, LR)

