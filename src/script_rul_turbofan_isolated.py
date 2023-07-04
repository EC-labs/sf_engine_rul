import os 
import torch
import multiprocessing
import logging
import yaml
import json
import time
from torch.utils.data import DataLoader

import config
from models.turbofan import (
    CreatorCNNTurbofanIsolated, train_one_epoch, validate, FileCNNRULStruct,
    model_recreate_cnnrul, improved_validation_cnnrul, equivalent_config_cnnrul
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

    cpu_count = multiprocessing.cpu_count()
    creator = CreatorCNNTurbofanIsolated(model_config=model_config)
    neural, datasets = creator.create_model_datasets(neural)
    dataloader_train = DataLoader(
        datasets["train"], batch_size=config.B, shuffle=True, num_workers=cpu_count
    )
    dataloader_validation = DataLoader(
        datasets["validation"], batch_size=config.B, shuffle=True, num_workers=cpu_count
    )
    dataloader_validation_total = DataLoader(
        datasets["validation_total"], batch_size=config.B, shuffle=True, num_workers=cpu_count
    )
    optimizer = torch.optim.Adam(neural.parameters(), lr=config.LR)
    loss_criterion = torch.nn.MSELoss()
    logger.info(f"Total Batches: {len(dataloader_train)}")
    logger.info(f"Dataset size: {len(datasets['train'])}")

    for epoch in range(config.R):
        logger.info(f"Epoch {epoch}")
        logger.info("Train")
        start = time.time()
        train_one_epoch(neural, dataloader_train, optimizer, loss_criterion)
        logger.info("Validate")
        loss_validation = validate(neural, dataloader_validation)
        end = time.time()
        loss_validation_total = validate(neural, dataloader_validation_total)
        training_times.append(end-start)
        persist_json(training_times, training_time_path)
        validations.append((loss_validation, loss_validation_total))
        persist_json(validations, validations_path)
        candidate_model = FileCNNRULStruct(
            neural.state_dict(), creator.model_config, config.runtime_config,
            loss_validation,
        )
        if (persisted_model == None) or improved_validation_cnnrul(persisted_model, candidate_model): 
            logger.info(f"Store candidate. Validation Results: {loss_validation}")
            file_model.file_store(model_path, candidate_model)
            persisted_model = candidate_model

if __name__ == "__main__": 
    main()
