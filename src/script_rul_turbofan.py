import torch
import time
import multiprocessing
import logging
from torch.utils.data import DataLoader

import config
from models.turbofan import (
    CreatorCNNTurbofan, train_one_epoch, validate, test
)

logger = logging.getLogger(__name__)

cpu_count = multiprocessing.cpu_count()
neural, datasets = CreatorCNNTurbofan.create_model_datasets()
dataloader_train = DataLoader(
    datasets["train"], batch_size=config.B, shuffle=True, num_workers=cpu_count
)
dataloader_validation = DataLoader(
    datasets["validation"], batch_size=config.B, shuffle=True, num_workers=cpu_count
)
optimizer = torch.optim.Adam(neural.parameters(), lr=config.LR)
loss_criterion = torch.nn.MSELoss()
logger.info(len(dataloader_train)) 
logger.info(len(datasets["train"]))

for epoch in range(config.R): 
    logger.info(f"Epoch {epoch}")
    logger.info("Train")
    train_one_epoch(neural, dataloader_train, optimizer, loss_criterion)
    logger.info("Validate")
    loss_validation = validate(neural, dataloader_validation)

test(neural, datasets["test"])
