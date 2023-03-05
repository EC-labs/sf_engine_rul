import torch

from typing import Tuple, Dict
from torch.utils.data import Dataset

from . import FactoryModelDatasets


class CreatorVGGCifar(FactoryModelDatasets): 


    @staticmethod
    def create_model_datasets() -> Tuple[torch.nn.Module, Dict[str, Dataset]]:
        pass
