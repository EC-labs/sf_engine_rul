import time
import torch
import pickle
import argparse
import logging
import torchvision
import torchvision.transforms as transforms

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
neural_network_unit = utils.get_model(
    'Unit', 'VGG5', config.model_len-1, 'cpu', config.model_cfg
)
server = Server(
    '0.0.0.0', config.SERVER_PORT, 'VGG5', offload, LR, neural_network_unit,
    torch.optim.SGD, torch.nn.CrossEntropyLoss()
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root=config.dataset_path, 
    train=False, 
    download=True, 
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4
)

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

    test_acc = server.test(testloader)
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

