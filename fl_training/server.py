import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import random
import numpy as np

import logging

import sys
sys.path.append('../')
import utils
import config

from typing import List

from Communicator import *

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Server(Communicator):


    def __init__(
        self, 
        ip_address: str, 
        server_port: int, 
        model_name: str, 
        offload: bool, 
        LR: float, 
    ):
        super(Server, self).__init__(ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.client_socks = {}

        while len(self.client_socks) < config.K:
            self.sock.listen(5)
            logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, _)) = self.sock.accept()
            logger.info('Got connection from ' + str(ip))
            logger.info(client_sock)
            self.client_socks[str(ip)] = client_sock

        self.uninet = utils.get_model(
            'Unit', self.model_name, config.model_len-1, self.device, config.model_cfg
        )
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.testset = torchvision.datasets.CIFAR10(
            root=config.dataset_path, 
            train=False, 
            download=True, 
            transform=self.transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=4
        )
        self.initialize(config.split_layer, offload, True, LR)
         
    def initialize(
        self, 
        split_layers: List[int], 
        offload: bool, 
        first: bool, 
        LR: float
    ):
        """Create a NN model for each client.

        The model is split between the client- and server-side models. In which
        layer the split occurs is determined by each value in the `split_layers`
        array. 

        Given the split model, the server stores the server side model, and
        broadcasts the weights of the "Unit" model to each client. Based on each
        client's split, they only use the weights relative to their layers. 

        Args: 
            split_layers: 
                A list wherein each index and value, represents a client and its
                respective NN model split layer.
            offload: 
                A boolean indicating whether RL agent is being used. Since the
                agent returns each client's new split layer after every epoch,
                the server-side model for each client has to be updated.
            first: 
                A boolean indicating whether it is the first time creating the
                split NN models. Should only be `True` while the server is being
                setup.
            LR:
                A float indicating the learning rate for the optimizer.
        """

        if offload or first:
            self.split_layers = split_layers
            self.nets = {}
            self.optimizers= {}

            for i, client_ip in enumerate(self.client_socks):
                if split_layers[i] < len(config.model_cfg[self.model_name])-1:
                    self.nets[client_ip] = utils.get_model(
                        'Server', self.model_name, split_layers[i], self.device, config.model_cfg
                    )
                    cweights = utils.get_model(
                        'Client', self.model_name, split_layers[i], self.device, config.model_cfg
                    ).state_dict()
                    pweights = utils.split_weights_server(
                        self.uninet.state_dict(), cweights, self.nets[client_ip].state_dict()
                    )
                    self.nets[client_ip].load_state_dict(pweights)

                    self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                      momentum=0.9)
                else:
                    self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
            self.criterion = nn.CrossEntropyLoss()

        msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)

    def train(self):
        # Network test
        self.net_threads = {}
        self.bandwidth = {}

        # Training start
        self.threads = {}
        for i, client_ip in enumerate(self.client_socks):
            if config.split_layer[i] == (config.model_len -1):
                self.threads[client_ip] = threading.Thread(
                    target=self._thread_training_no_offloading,
                    args=(client_ip,)
                )
                logger.info(client_ip + ' no offloading training start')
                self.threads[client_ip].start()
            else:
                logger.info(client_ip)
                self.threads[client_ip] = threading.Thread(
                    target=self._thread_training_offloading, 
                    args=(client_ip,)
                )
                logger.info(client_ip + ' offloading training start')
                self.threads[client_ip].start()

        for client_ip in self.client_socks:
            self.threads[client_ip].join()

        self.ttpi = {} # Training time per iteration
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
            self.ttpi[msg[1]] = msg[2]

        return self.bandwidth

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip):
        iteration = int((config.N / (config.K * config.B)))
        for i in range(iteration):
            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
            smashed_layers = msg[1]
            labels = msg[2]

            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizers[client_ip].step()

            # Send gradients to client
            msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
            self.send_msg(self.client_socks[client_ip], msg)

        logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self):
        w_local_list =[]
        for i, client_ip in enumerate(self.client_socks):
            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            if config.split_layer[i] != (config.model_len-1):
                w_local = (
                    utils.concat_weights(
                        self.uninet.state_dict(), # weights 
                        msg[1], # cweights 
                        self.nets[client_ip].state_dict() # sweights
                    ), 
                    config.N / config.K
                )
                w_local_list.append(w_local)
            else:
                w_local = (msg[1],config.N / config.K)
                w_local_list.append(w_local)
        zero_model = utils.zero_init(self.uninet).state_dict()
        aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
        
        self.uninet.load_state_dict(aggregrated_model)
        return aggregrated_model

    def test(self, r):
        self.uninet.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.uninet(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        logger.info('Test Accuracy: {}'.format(acc))

        # Save checkpoint.
        torch.save(self.uninet.state_dict(), './'+ config.model_name +'.pth')

        return acc

    def reinitialize(self, split_layers, offload, first, LR):
        self.initialize(split_layers, offload, first, LR)
