import sys
import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL") or logging.INFO
logging.basicConfig(
    format='%(asctime)s [%(levelname)s]:%(name)s:%(threadName)s: %(message)s',
    level=LOG_LEVEL,
)

# Network configration
SERVER_ADDR= 'fedadapt_server'
SERVER_PORT = 51000

K = int(os.getenv("NCLIENTS", "1")) # Number of devices
G = 3 # Number of groups

# Dataset configration
dataset_name = 'CIFAR10'
home = '/usr/src/app'
dataset_path = '/usr/src/app/data/raw'
N = 50 # data length


# Model configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    'VGG5' : [
        ('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), 
        ('M', 32, 32, 2, 32*16*16, 0), 
        ('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), 
        ('M', 64, 64, 2, 64*8*8, 0), 
        ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
        ('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
        ('D', 128, 10, 1, 10, 128*10),
    ]
}

model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = 2 #Initial split layers
model_len = 7


# FL training configration
R = 50 # FL rounds
LR = 0.001 # Learning rate
B = 128 # Batch size


# RL training configration
max_episodes = 100         # max training episodes
max_timesteps = 100        # max timesteps in one episode
exploration_times = 20       # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100                   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {
    '192.168.0.14': 5, 
    '192.168.0.15': 5, 
    '192.168.0.25': 50, 
    '192.168.0.36': 5, 
    '192.168.0.29': 5,
}  # infer times for each device

random = True
random_seed = 0
