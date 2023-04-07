import logging
import os 
import sys
import json

import config
from models import file_model
from models.turbofan import (
    model_recreate_cnnrul, test_per_flight, CreatorCNNTurbofan
)

def persist_json(json_serializable, file_path): 
    with open(file_path, "w") as f: 
        json.dump(json_serializable, f)

logger = logging.getLogger(__name__)

def main(): 
    result_relative_model_directory = sys.argv[1] 
    directory_path = os.path\
        .join(config.evaluation_directory, result_relative_model_directory)
    model_path = os.path.join(directory_path, "model.pkl")
    try:
        persisted_model = file_model.file_load(model_path)
        neural = model_recreate_cnnrul(
            persisted_model, persisted_model.model_config_context
        )
    except file_model.MissingFile:
        logger.info(f"Missing persisted model {model_path}")
        return

    _, datasets = CreatorCNNTurbofan(model_config=persisted_model.model_config_context)\
        .create_model_datasets(neural=neural)
    test_predicted_path = os.path.join(directory_path, "predicted_real.json")
    test_metrics_path = os.path.join(directory_path, "test_metrics.json")
    rmse, mae = test_per_flight(neural, datasets["test"], test_predicted_path)
    persist_json({"rmse": rmse, "mae": mae}, test_metrics_path)

if __name__ == "__main__": 
    main()
