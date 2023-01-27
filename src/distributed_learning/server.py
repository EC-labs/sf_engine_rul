import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import threading
import tqdm
import functools
import numpy as np
import socket
import time
import logging

from typing import List, Dict, Type, Iterable, Dict, Any
from dataclasses import dataclass

import config

from .communicator import Communicator
from . import utils


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)


class InitSplitFedServerException(Exception): 
    pass


class ThreadValidationException(Exception): 
    pass


@dataclass 
class StructOptimizerConstructor: 
    cls_optimizer: Type[torch.optim.Optimizer]
    args: Iterable
    kwargs: Dict[str, Any]


class SplitFedServerThread:


    comm: Communicator
    _optimizer: torch.optim.Optimizer

    def __init__(
        self, comm, neural_network, cls_optimizer, criterion
    ):
        self.comm = comm
        self.criterion = criterion
        self.cls_optimizer = cls_optimizer
        self.neural_network = neural_network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
    def optimizer(self, *args, **kwargs): 
        self._optimizer = self.cls_optimizer(
            self.neural_network.parameters(), *args, **kwargs
        )

    @property
    def loss_validation(self): 
        if not hasattr(self, "_loss_validation"): 
            raise ThreadValidationException(
                "Validation has not been executed after training"
            )
        return self._loss_validation

    def train_offloading(self):
        if hasattr(self, "_loss_validation"): 
            del self._loss_validation
        self.neural_network.train()
        _, iterations_number = self.comm.recv_msg(
            expect_msg_type='CLIENT_TRAINING_ITERATIONS_NUMBER'
        )
        logger.debug(f"Number training iterations: {iterations_number}")
        for i in tqdm.tqdm(range(iterations_number)):
            msg = self.comm.recv_msg('MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
            smashed_layers = msg[1]
            labels = msg[2]

            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self._optimizer.zero_grad()
            outputs = self.neural_network(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self._optimizer.step()

            msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT', inputs.grad]
            self.comm.send_msg(msg)
        training_time = self.comm.recv_msg(
            expect_msg_type='MSG_TRAINING_TIME_PER_ITERATION'
        )

    def validate(self): 
        _, iterations_number = self.comm.recv_msg(
            expect_msg_type='CLIENT_VALIDATION_ITERATIONS_NUMBER'
        )
        logger.debug(f"Number validation iterations: {iterations_number}")
        loss_validation_total = 0
        self.neural_network.eval()
        with torch.no_grad(): 
            for i in tqdm.tqdm(range(iterations_number)):
                msg = self.comm.recv_msg('MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                smashed_layers = msg[1]
                labels = msg[2]
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                self._optimizer.zero_grad()
                outputs = self.neural_network(inputs)
                loss = self.criterion(outputs, targets)
                loss_validation_total += loss.item()
        self._loss_validation = loss_validation_total 




class SplitFedServer: 


    sock: socket.socket
    threads: List[SplitFedServerThread]
    pending_clients: List[SplitFedServerThread]
    pending_lock: threading.Lock
    cls_optimizer: Type[torch.optim.Optimizer]
    struct_optimizer_constructor: StructOptimizerConstructor
    thread_listen: threading.Thread
    thread_train: threading.Thread

    def __init__(
        self, ip_address, server_port, neural_network_unit, 
        cls_optimizer: Type[torch.optim.Optimizer], criterion,
        nn_server_creator, split_layer
    ): 
        self.sock = socket.socket()
        self.sock.bind((ip_address, server_port))
        self.neural_network_unit = neural_network_unit
        self.cls_optimizer = cls_optimizer
        self.threads = []
        self.pending_clients = []
        self.pending_lock = threading.Lock()
        self.criterion = criterion
        self.nn_server_creator = nn_server_creator
        self.split_layer = split_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def optimizer(self, *args, **kwargs): 
        self.struct_optimizer_constructor = StructOptimizerConstructor(
            self.cls_optimizer, [*args], {**kwargs}
        )

    def create_thread(self, comm): 
        thread_sf = SplitFedServerThread(
            comm, self.nn_server_creator(self.split_layer),
            self.cls_optimizer, self.criterion
        )
        thread_sf.optimizer(
            *self.struct_optimizer_constructor.args,
            **self.struct_optimizer_constructor.kwargs,
        )
        with self.pending_lock:
            self.pending_clients.append(thread_sf)

    def _listen(self): 
        logger.info("Ready to connect")
        while True:
            self.sock.listen(5)
            (sock, (ip, _)) = self.sock.accept()
            logger.info(f'Client connected: {ip}')
            self.create_thread(Communicator(sock=sock))

    def listen(self): 
        if not hasattr(self, "struct_optimizer_constructor"): 
            raise InitSplitFedServerException(
                "Optimizer was not initialized."
            )
        self.thread_listen = threading.Thread(
            target=self._listen, name="thread_listen"
        )
        self.thread_listen.start()

    def _train(self): 
        threads_training = [
            threading.Thread(
                target=t.train_offloading, 
                name=f"thread_training_{i}"
            ) 
            for i, t in enumerate(self.threads)
        ]
        logger.debug("Start threads training")
        for t in threads_training: 
            t.start()
        for t in threads_training: 
            t.join()
        logger.debug("End threads training")
        self.aggregate()
        self._weights_nn_unit_send(self.threads)
        self._validate()

    def train(self, min_clients=1):
        self._add_pending_clients()
        while len(self.threads) < min_clients: 
            logger.info("Not enough clients connected")
            self._add_pending_clients()
            time.sleep(2)
        self._train()

    def _add_pending_clients(self): 
        with self.pending_lock: 
            self.threads.extend(self.pending_clients)
            list_clients_init = self.pending_clients
            self.pending_clients = []
        self._weights_nn_unit_send(list_clients_init)

    def _weights_nn_unit_send(self, list_client_threads): 
        msg = [
            'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', 
            self.neural_network_unit.state_dict(),
        ]
        for client_thread in list_client_threads: 
            client_thread.comm.send_msg(msg)

    def aggregate(self): 
        list_weights_concat = []
        for thread in self.threads: 
            _, weights_client = thread.comm.recv_msg(
                expect_msg_type='MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER'
            )
            weights_concat = utils.concat_weights(
                self.neural_network_unit.state_dict(),
                weights_client,
                thread.neural_network.state_dict(),
            )
            list_weights_concat.append((weights_concat, config.N/config.K))
        zero_model = utils.zero_init(self.neural_network_unit).state_dict()
        aggregated_model = utils.fed_avg(
            zero_model, list_weights_concat, config.N
        )
        self.neural_network_unit.load_state_dict(aggregated_model)

    def _validate(self): 
        threads_training = [
            threading.Thread(
                target=t.validate, 
                name=f"thread_validate_{i}"
            ) 
            for i, t in enumerate(self.threads)
        ]
        logger.debug("Start threads validation")
        for t in threads_training: 
            t.start()
        for t in threads_training: 
            t.join()
        total_validation_loss = functools.reduce(
            lambda acc, x: x.loss_validation+acc, self.threads, 0
        )
        logger.debug("End threads validation")


    def test(self, testloader): 
        self.neural_network_unit.eval()
        with torch.no_grad(): 
            test_loss = correct = total = 0
            for inputs, targets in tqdm.tqdm(testloader): 
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.neural_network_unit(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        logger.info(f"Test Accuracy: {acc}")
