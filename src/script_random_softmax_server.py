import time
import torch
import logging
import os
import yaml
import json

import config

from distributed_learning.server import SplitFedServer
from models.turbofan import (
    CreatorCNNEngine, compute_rmse_mae, test, FileCNNRULStruct,
    equivalent_config_cnnrul, model_recreate_cnnrul, improved_validation_cnnrul
)
from models import file_model


logger = logging.getLogger(__name__)

def persist_json(json_serializable, file_path): 
    with open(file_path, "w") as f: 
        json.dump(json_serializable, f)

def load_persisted_model(model_config, persisted_model_path): 
    try: 
        persisted_model = file_model.file_load(persisted_model_path)
        persisted_config = persisted_model.model_config_context
        if not equivalent_config_cnnrul(model_config, persisted_config):
            return persisted_model, None
        logger.info("Load model. Validation results: "
                    f"{persisted_model.validation_results}")
        neural = model_recreate_cnnrul(persisted_model, model_config)
        return persisted_model, neural

    except file_model.MissingFile: 
        return None, None

def main(): 
    training_times = []
    validations = []

    model_config_path = os.path.join(config.home, "models/turbofan.yml")
    with open(model_config_path, "r") as f: 
        model_config = yaml.safe_load(f)

    program_directory = config.evaluation_directory
    model_path = os.path.join(program_directory, "model.pkl")
    training_time_path = os.path.join(program_directory, "training_time.json")
    validations_path = os.path.join(program_directory, "validations.json")
    persisted_model, neural = load_persisted_model(model_config, model_path)

    logger.info('Preparing Server.')
    creator = CreatorCNNEngine(model_config=model_config, neural_network=neural)
    nn_unit = creator.neural_network
    nn_server_creator = creator.nn_server_create
    server = SplitFedServer(
        '0.0.0.0', config.SERVER_PORT, nn_unit, torch.optim.Adam,
        torch.nn.MSELoss(), nn_server_creator, config.split_layer
    )
    server.optimizer(lr=config.LR)
    server.listen()

    for r in range(config.R):
        start = time.time()
        logger.info(f"Epoch {r}")
        server.train(min_clients=config.NCLIENTS)
        server.aggregate("validation_softmax")
        outputs, targets = server.validate()
        rmse, mae = compute_rmse_mae(outputs, targets)
        logger.info(f"Validate: RMSE {rmse}\tMAE {mae}")

        end = time.time()
        training_times.append(end-start)
        persist_json(training_times, training_time_path)
        validations.append(rmse)
        persist_json(validations, validations_path)
        candidate_model = FileCNNRULStruct(
            server.neural_network_unit.state_dict(),
            creator.model_config,config.runtime_config, rmse,
        )
        if (persisted_model == None) or improved_validation_cnnrul(persisted_model, candidate_model): 
            logger.info(f"Store candidate. Validation Results: {rmse}")
            file_model.file_store(model_path, candidate_model)
            persisted_model = candidate_model
    server.stop_server = True

if __name__ == "__main__": 
    main()
