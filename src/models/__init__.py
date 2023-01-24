import torch

from abc import abstractmethod
from typing import Tuple, Dict, Type
from torch.utils.data import Dataset


class FactoryModelDatasets: 


    @staticmethod
    @abstractmethod
    def create_model_datasets() -> Tuple[Type[torch.nn.Module], Dict[str, Type[Dataset]]]:
        """This function provides the neural network, and the training,
        validation, and testing datasets. 

        Each implementation of this method is responsible for: 

            * Reading in the data
            * Data processing
            * Loading the data into a Dataset class
            * Selecting the respective neural network model

        Returns:
            A tuple wherein the first value is the neural network, and the
            second is a dictionary with train, validation and test generators.
        """
        pass
