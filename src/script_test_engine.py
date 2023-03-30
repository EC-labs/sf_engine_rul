import logging
import os 
import yaml

import config
from models import file_model
from models.turbofan import (
    model_recreate_cnnrul, test_per_flight, CreatorCNNTurbofan
)


logger = logging.getLogger(__name__)
file_path = os.path.join(config.results_dir, "engine_turbofan_rul.pkl")
model_config_path = os.path.join(config.home, "models/turbofan.yml")
with open(model_config_path, "r") as f: 
    model_config = yaml.safe_load(f)
try:
    persisted_model = file_model.file_load(file_path)
    neural = model_recreate_cnnrul(
        persisted_model, persisted_model.model_config_context
    )
except file_model.MissingFile:
    logger.info(f"Missing persisted model {file_path}")
    exit(0)

_, datasets = CreatorCNNTurbofan().create_model_datasets(neural=neural)
test_file_path = os.path.join(config.results_dir, "engine_predicted_real.json")
test_per_flight(neural, datasets["test"], test_file_path)
