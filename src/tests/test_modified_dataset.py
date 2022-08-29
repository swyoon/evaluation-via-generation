import os
import pickle

import numpy as np
import pytest
import torch
import yaml
from skimage import io
from torch.utils import data
from torchvision.transforms import Compose, Resize, ToTensor

from loader import get_dataset
from loader.modified_dataset import MNIST_OOD, FashionMNIST_OOD, Gray2RGB, NotMNIST


def test_mnist():
    data_root = "datasets"
    ds = MNIST_OOD(data_root, split="training")
    ds[0]
    img = ds[0][0]
    assert isinstance(ToTensor()(img), torch.Tensor)


def test_fmnist():
    data_root = "datasets"
    ds = FashionMNIST_OOD(data_root, split="training")
    img = ds[0][0]
    assert isinstance(ToTensor()(img), torch.Tensor)


# def test_notmnist():
#     data_root = 'datasets'
#     ds = NotMNIST(data_root, split='training')
#     img = ds[0][0]
#     assert isinstance(ToTensor()(img), torch.Tensor)


@pytest.mark.parametrize(
    "dataset_name",
    [
        "MNIST_OOD",
        "FashionMNIST_OOD",
        "CIFAR10_OOD",
        "CIFAR100_OOD",
        "SVHN_OOD",
        "Constant_OOD",
        "ConstantGray_OOD",
        "Noise_OOD",
        "CelebA_OOD",
    ],
)
def test_get_dataset(dataset_name):
    data_dict = {
        "dataset": dataset_name,
        "path": "datasets",
        "split": "training",
    }
    if dataset_name in {"MNIST_OOD", "FashionMNIST_OOD"}:
        data_dict["size"] = 32
    ds = get_dataset(data_dict)
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert isinstance(len(ds), int)


@pytest.mark.parametrize("dataset_name", ["Constant_OOD", "ConstantGray_OOD"])
def test_constant_dataset(dataset_name):
    data_dict = {
        "dataset": dataset_name,
        "path": "datasets",
        "channel": 1,
        "size": 28,
        "split": "training",
    }
    ds = get_dataset(data_dict)
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert isinstance(len(ds), int)


@pytest.mark.parametrize(
    "dataset_name", ["Constant_OOD", "ConstantGray_OOD", "Noise_OOD"]
)
def test_get_dataset_channel(dataset_name):
    data_dict = {
        "dataset": dataset_name,
        "path": "datasets",
        "size": 28,
        "channel": 1,
        "split": "training",
    }
    ds = get_dataset(data_dict)
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert isinstance(len(ds), int)
