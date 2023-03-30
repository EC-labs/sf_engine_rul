import time
import torch
import logging
import os
import yaml

import config

from distributed_learning.server import SplitFedServer
from models.turbofan import (
    CreatorCNNEngine, compute_rmse_mae, test, FileCNNRULStruct,
    equivalent_config_cnnrul, model_recreate_cnnrul, improved_validation_cnnrul
)
from models import file_model



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LR = config.LR
split_layer = config.split_layer


file_path = os.path.join(config.results_dir, "engine_turbofan_rul.pkl")
model_config_path = os.path.join(config.home, "models/turbofan.yml")
with open(model_config_path, "r") as f: 
    model_config = yaml.safe_load(f)
try:
    persisted_model = file_model.file_load(file_path)
    logger.info(f"Model persisted")
    if equivalent_config_cnnrul(model_config, persisted_model.model_config_context):
        logger.info(
            f"Load model. Validation results: {persisted_model.validation_results}"
        )
        neural = model_recreate_cnnrul(persisted_model, model_config)
    else:
        neural = None
except file_model.MissingFile:
    persisted_model = None
    neural = None

logger.info('Preparing Server.')
creator = CreatorCNNEngine(model_config=model_config, neural_network=neural)
nn_unit = creator.neural_network
nn_server_creator = creator.nn_server_create
server = SplitFedServer(
    '0.0.0.0', config.SERVER_PORT, nn_unit, torch.optim.Adam,
    torch.nn.MSELoss(), nn_server_creator, split_layer
)
server.optimizer(lr=LR)
server.listen()
time.sleep(10)

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
    logger.info(f"Epoch {r}")
    server.train()
    outputs, targets = server.validate()
    rmse, mae = compute_rmse_mae(outputs, targets)
    logger.info(f"Validate: RMSE {rmse}\tMAE {mae}")
    candidate_model = FileCNNRULStruct(
        server.neural_network_unit.state_dict(),
        creator.model_config,config.runtime_config, rmse,
    )
    if (persisted_model == None) or improved_validation_cnnrul(persisted_model, candidate_model): 
        logger.info(f"Store candidate. Validation Results: {rmse}")
        file_model.file_store(file_path, candidate_model)
        persisted_model = candidate_model
