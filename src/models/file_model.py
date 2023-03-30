import os
import torch
import pickle

from typing import Type
from dataclasses import dataclass


class MissingFile(Exception): 
    pass

def file_load(file_path):
    if not os.path.isfile(file_path):
        raise MissingFile()

    with open(file_path, "rb") as f: 
        return pickle.load(f)

def file_store(file_path, serialize_object):
    with open(file_path, "wb") as f:
        pickle.dump(serialize_object, f)

