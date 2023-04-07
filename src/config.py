import sys
import os
import torch
import numpy as np
import logging
import yaml

LOG_LEVEL = os.getenv("LOG_LEVEL") or logging.INFO
PROGRAM_NAME = os.getenv("PROGRAM_NAME") 
if PROGRAM_NAME == None: 
    raise Exception()
FAULTY = bool(int(os.getenv("FAULTY", "0")))
FAULTY_CLIENT = int(os.getenv("FAULTY_CLIENT", "0"))

logging.basicConfig(
    format='%(asctime)s [%(levelname)s]:%(name)s:%(threadName)s: %(message)s',
    level=LOG_LEVEL,
)

SERVER_ADDR= 'fedadapt_server'
SERVER_PORT = 51000

dataset_name = 'CIFAR10'
home = '/usr/src/app'
dataset_path = os.path.join(home, "data/raw")

runtime_config_file_path = os.path.join(home, "config.yml")
with open(runtime_config_file_path, "r") as f: 
    runtime_config = yaml.safe_load(f)

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
split_layer = 2
model_len = 7
results_dir = os.path.join(home, "results")
evaluation_directory = os.path.join(
    results_dir, runtime_config["evaluation_directory"]
)
if not os.path.isdir(evaluation_directory): 
    os.mkdir(evaluation_directory)



R = runtime_config["epochs"]
LR = runtime_config["learning_rate"]
B = runtime_config["batch_size"]

seed = runtime_config["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
