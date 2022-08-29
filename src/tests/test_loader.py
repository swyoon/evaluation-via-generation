import os
import pickle

import numpy as np
import pytest
import torch
import yaml
from skimage import io
from torch.utils import data

from loader import get_dataloader, get_dataset


def test_get_dataloader():
    cfg = {
        "dataset": "FashionMNISTpad_OOD",
        "path": "datasets",
        "shuffle": True,
        "n_workers": 0,
        "batch_size": 1,
        "split": "training",
    }
    dl = get_dataloader(cfg)


def test_concat_dataset():
    data_cfg = {
        "concat1": {
            "dataset": "FashionMNISTpad_OOD",
            "path": "datasets",
            "shuffle": True,
            "split": "training",
        },
        "concat2": {
            "dataset": "MNISTpad_OOD",
            "path": "datasets",
            "shuffle": True,
            "n_workers": 0,
            "batch_size": 1,
            "split": "training",
        },
        "n_workers": 0,
        "batch_size": 1,
    }
    get_dataset(data_cfg)
