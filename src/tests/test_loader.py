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


def test_rimgnet_dataset():
    cfg = {
        "dataset": "RImgNet",
        "path": "datasets",
        "shuffle": False,
        "n_workers": 0,
        "batch_size": 1,
        "split": "evaluation",
    }
    dl = get_dataloader(cfg)
    xx, yy = next(iter(dl))
    assert xx.shape == (1, 3, 224, 224)
    assert yy.shape == (1,)
    print(len(dl.dataset))


def test_flowers_dataset():
    cfg = {
        "dataset": "Flowers",
        "path": "datasets",
        "shuffle": False,
        "n_workers": 0,
        "batch_size": 1,
        "split": "evaluation",
    }
    dl = get_dataloader(cfg)
    xx, yy = next(iter(dl))
    assert xx.shape == (1, 3, 224, 224)
    assert yy.shape == (1,)


def test_cars_dataset():
    cfg = {
        "dataset": "Cars",
        "path": "datasets",
        "shuffle": False,
        "n_workers": 0,
        "batch_size": 1,
        "split": "evaluation",
    }
    dl = get_dataloader(cfg)
    xx, yy = next(iter(dl))
    assert xx.shape == (1, 3, 224, 224)
    assert yy.shape == (1,)


def test_fgvc_dataset():
    cfg = {
        "dataset": "FGVC",
        "path": "datasets",
        "shuffle": False,
        "n_workers": 0,
        "batch_size": 1,
        "split": "evaluation",
    }
    dl = get_dataloader(cfg)
    xx, yy = next(iter(dl))
    assert xx.shape == (1, 3, 224, 224)
    assert yy.shape == (1,)
