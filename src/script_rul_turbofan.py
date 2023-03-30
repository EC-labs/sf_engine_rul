import os 
import torch
import multiprocessing
import logging
import yaml
from torch.utils.data import DataLoader
from typing import Optional

import config
from models.turbofan import (
    CreatorCNNTurbofan, train_one_epoch, validate, test,
    test_per_flight, FileCNNRULStruct, model_recreate_cnnrul, CNNRUL, 
    improved_validation_cnnrul, equivalent_config_cnnrul
)
from models import file_model


import pdb; pdb.set_trace();
logger = logging.getLogger(__name__)
file_path = os.path.join(config.results_dir, "turbofan_rul.pkl")

model_config_path = os.path.join(config.home, "models/turbofan.yml")
with open(model_config_path, "r") as f: 
    model_config = yaml.safe_load(f)
try:
    persisted_model = file_model.file_load(file_path)
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

cpu_count = multiprocessing.cpu_count()
creator = CreatorCNNTurbofan(model_config=model_config)
neural, datasets = creator.create_model_datasets(neural)
dataloader_train = DataLoader(
    datasets["train"], batch_size=config.B, shuffle=True, num_workers=cpu_count
)
dataloader_validation = DataLoader(
    datasets["validation"], batch_size=config.B, shuffle=True, num_workers=cpu_count
)
optimizer = torch.optim.Adam(neural.parameters(), lr=config.LR)
loss_criterion = torch.nn.MSELoss()
logger.info(f"Total Batches: {len(dataloader_train)}")
logger.info(f"Dataset size: {len(datasets['train'])}")

for epoch in range(config.R):
    logger.info(f"Epoch {epoch}")
    logger.info("Train")
    train_one_epoch(neural, dataloader_train, optimizer, loss_criterion)
    logger.info("Validate")
    loss_validation = validate(neural, dataloader_validation)
    candidate_model = FileCNNRULStruct(
        neural.state_dict(), creator.model_config, config.runtime_config,
        loss_validation,
    )
    if (persisted_model == None) or improved_validation_cnnrul(persisted_model, candidate_model): 
        logger.info(f"Store candidate. Validation Results: {loss_validation}")
        file_model.file_store(file_path, candidate_model)
        persisted_model = candidate_model

test_per_flight(neural, datasets["test"])
