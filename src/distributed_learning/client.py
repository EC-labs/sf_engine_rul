import torch
import tqdm
import time
import numpy as np
import sys
import logging

from typing import Type, Optional

from . import utils
from .communicator import Communicator


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


    net: torch.nn.Module
    split_layer: int
    conn: Communicator
    cls_optimizer: Type[torch.optim.Optimizer]
    _optimizer: torch.optim.Optimizer

    def __init__(
        self, server_addr, server_port, model_name, 
        split_layer, criterion, cls_optimizer: Type[torch.optim.Optimizer], 
        neural_network: torch.nn.Module,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.neural_network = neural_network
        self.criterion = criterion
        self.cls_optimizer = cls_optimizer
        logger.info('Connecting to Server.')
        self.conn = Communicator()
        self.conn.connect((server_addr, server_port))
        self._weights_receive()


    def optimizer(self, *args, **kwargs): 
        self._optimizer = self.cls_optimizer(
            self.neural_network.parameters(), *args, **kwargs
        )

    def train(self, dataloader_train, dataloader_validate=None):
        try: 
            assert hasattr(self, "_optimizer")
        except: 
            logger.exception("Optimizer has not been initialized.")
            raise

        msg = ['CLIENT_TRAINING_ITERATIONS_NUMBER', len(dataloader_train)]
        self.conn.send_msg(msg)
        s_time_total = time.time()
        self.neural_network.to(self.device)
        self.neural_network.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader_train)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self._optimizer.zero_grad()
            outputs = self.neural_network(inputs)
            msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
            self.conn.send_msg(msg)
            gradients = self.conn.recv_msg()[1].to(self.device)
            outputs.backward(gradients)
            self._optimizer.step()
        e_time_total = time.time()
        logger.info('Total time: ' + str(e_time_total - s_time_total))
        training_time_pr = (e_time_total - s_time_total) / len(dataloader_train)
        logger.info('training_time_per_iteration: ' + str(training_time_pr))
        msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.conn.ip, training_time_pr]
        self.conn.send_msg(msg)
        self._weights_upload()
        self._weights_receive()
        self._validate(dataloader_validate)
        return e_time_total - s_time_total
        
    def _validate(
        self, dataloader_validate: Optional[torch.utils.data.DataLoader] = None
    ): 
        iter_validate = (len(dataloader_validate) 
            if dataloader_validate != None else 0
        )
        msg = ['CLIENT_VALIDATION_ITERATIONS_NUMBER', iter_validate]
        self.conn.send_msg(msg)
        self.neural_network.eval() 
        if iter_validate == 0: 
            return
        with torch.no_grad(): 
            for inputs, targets in tqdm.tqdm(dataloader_validate):
                outputs = self.neural_network(inputs)
                msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
                self.conn.send_msg(msg)

    def _weights_upload(self):
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.neural_network.cpu().state_dict()]
        self.conn.send_msg(msg)

    def _weights_receive(self):
        logger.debug('Receive Global Weights..')
        weights = self.conn.recv_msg()[1]
        pweights = utils.split_weights_client(weights, self.neural_network.state_dict())
        self.neural_network.load_state_dict(pweights)

