import copy
import json
import os

import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import FashionMNIST, ImageFolder, Omniglot
from torchvision.transforms import CenterCrop, Compose, Normalize, Pad, Resize, ToTensor

from augmentations import get_composed_augmentations
from augmentations.augmentations import Fragment, Invert, ToGray
from loader.basic_dataset import basic_dataset
from loader.inmemory_loader import InMemoryDataset, InMemoryLoader
from loader.modified_dataset import (
    CIFAR10_OOD,
    CIFAR100_OOD,
    MNIST_OOD,
    SVHN_OOD,
    CelebA_OOD,
    Constant_OOD,
    ConstantGray_OOD,
    FashionMNIST_OOD,
    Gray2RGB,
    ImageNet32,
    Noise_OOD,
    NotMNIST,
)

OOD_SIZE = 32  # common image size for OOD detection experiments


def get_dataloader(data_dict, mode=None, mode_dict=None, data_aug=None, subset=None):
    """constructs DataLoader
    data_dict: data part of cfg

    mode: deprecated argument
    mode_dict: deprecated argument
    data_aug: deprecated argument
    subset: the list of indices in dataset

    Example data_dict
        dataset: FashionMNISTpad_OOD
        path: datasets
        shuffle: True
        batch_size: 128
        n_workers: 8
        split: training
        dequant:
          UniformDequantize: {}
    """

    # dataset loading
    aug = get_composed_augmentations(data_dict.get("augmentations", None))
    dequant = get_composed_augmentations(data_dict.get("dequant", None))
    dataset = get_dataset(data_dict, split_type=None, data_aug=aug, dequant=dequant)

    if subset is not None:
        dataset = data.Subset(dataset, subset)

    # dataloader loading
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        num_workers=data_dict["n_workers"],
        shuffle=data_dict.get("shuffle", False),
        pin_memory=False,
    )

    return loader


def get_dataset(data_dict, split_type=None, data_aug=None, dequant=None):
    """
    split_type: deprecated argument
    """
    do_concat = any([k.startswith("concat") for k in data_dict.keys()])
    if do_concat:
        if data_aug is not None:
            return data.ConcatDataset(
                [
                    get_dataset(d, data_aug=data_aug)
                    for k, d in data_dict.items()
                    if k.startswith("concat")
                ]
            )
        elif dequant is not None:
            return data.ConcatDataset(
                [
                    get_dataset(d, dequant=dequant)
                    for k, d in data_dict.items()
                    if k.startswith("concat")
                ]
            )
        else:
            return data.ConcatDataset(
                [get_dataset(d) for k, d in data_dict.items() if k.startswith("concat")]
            )
    name = data_dict["dataset"]
    split_type = data_dict["split"]
    data_path = (
        data_dict["path"][split_type]
        if split_type in data_dict["path"]
        else data_dict["path"]
    )

    # default tranform behavior.
    original_data_aug = data_aug
    if data_aug is not None:
        # data_aug = Compose([data_aug, ToTensor()])
        data_aug = Compose([ToTensor(), data_aug])
    else:
        data_aug = ToTensor()

    if dequant is not None:  # dequantization should be applied last
        data_aug = Compose([data_aug, dequant])

    # datasets
    if name == "MNISTLeaveOut":
        l_out_class = data_dict["out_class"]
        dataset = MNISTLeaveOut(
            data_path,
            l_out_class=l_out_class,
            split=split_type,
            download=True,
            transform=data_aug,
        )
    elif name == "MNISTLeaveOutFragment":
        l_out_class = data_dict["out_class"]
        fragment = data_dict["fragment"]
        dataset = MNISTLeaveOut(
            data_path,
            l_out_class=l_out_class,
            split=split_type,
            download=True,
            transform=Compose([ToTensor(), Fragment(fragment)]),
        )
    elif name == "MNIST_OOD":
        size = data_dict.get("size", 28)
        if size == 28:
            l_transform = [ToTensor()]
        else:
            l_transform = [Gray2RGB(), Resize(size), ToTensor()]
        dataset = MNIST_OOD(
            data_path, split=split_type, download=True, transform=Compose(l_transform)
        )
        dataset.img_size = (size, size)

    elif name == "MNISTpad_OOD":
        dataset = MNIST_OOD(
            data_path,
            split=split_type,
            download=True,
            transform=Compose([Gray2RGB(), Pad(2), ToTensor()]),
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "FashionMNIST_OOD":
        size = data_dict.get("size", 28)
        if size == 28:
            l_transform = [ToTensor()]
        else:
            l_transform = [Gray2RGB(), Resize(size), ToTensor()]

        dataset = FashionMNIST_OOD(
            data_path, split=split_type, download=True, transform=Compose(l_transform)
        )
        dataset.img_size = (size, size)

    elif name == "FashionMNISTpad_OOD":
        dataset = FashionMNIST_OOD(
            data_path,
            split=split_type,
            download=True,
            transform=Compose([Gray2RGB(), Pad(2), ToTensor()]),
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "HalfMNIST":
        mnist = MNIST_OOD(
            data_path, split=split_type, download=True, transform=ToTensor()
        )
        dataset = Chimera(mnist, mode="horizontal_blank")
    elif name == "ChimeraMNIST":
        mnist = MNIST_OOD(
            data_path, split=split_type, download=True, transform=ToTensor()
        )
        dataset = Chimera(mnist, mode="horizontal")
    elif name == "CIFAR10_OOD":
        dataset = CIFAR10_OOD(
            data_path, split=split_type, download=True, transform=data_aug
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "CIFAR10LeaveOut":
        l_out_class = data_dict["out_class"]
        seed = data_dict.get("seed", 1)
        dataset = CIFAR10LeaveOut(
            data_path,
            l_out_class=l_out_class,
            split=split_type,
            download=True,
            transform=data_aug,
            seed=seed,
        )

    elif name == "CIFAR10_GRAY":
        dataset = CIFAR10_OOD(
            data_path,
            split=split_type,
            download=True,
            transform=Compose([ToTensor(), ToGray()]),
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "CIFAR100_OOD":
        dataset = CIFAR100_OOD(
            data_path, split=split_type, download=True, transform=data_aug
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "SVHN_OOD":
        size = data_dict.get("size", 32)
        if size != 32:
            data_aug = Compose([Resize(size), data_aug])

        dataset = SVHN_OOD(
            data_path, split=split_type, download=True, transform=data_aug
        )
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "Constant_OOD":
        size = data_dict.get("size", OOD_SIZE)
        channel = data_dict.get("channel", 3)
        dataset = Constant_OOD(
            data_path,
            split=split_type,
            size=(size, size),
            channel=channel,
            transform=ToTensor(),
        )

    elif name == "ConstantGray_OOD":
        size = data_dict.get("size", OOD_SIZE)
        channel = data_dict.get("channel", 3)
        dataset = ConstantGray_OOD(
            data_path,
            split=split_type,
            size=(size, size),
            channel=channel,
            transform=ToTensor(),
        )

    elif name == "Noise_OOD":
        channel = data_dict.get("channel", 3)
        size = data_dict.get("size", OOD_SIZE)
        dataset = Noise_OOD(
            data_path,
            split=split_type,
            transform=ToTensor(),
            channel=channel,
            size=(size, size),
        )

    elif name == "CelebA_OOD":
        size = data_dict.get("size", OOD_SIZE)
        l_aug = []
        l_aug.append(CenterCrop(140))
        l_aug.append(Resize(size))
        l_aug.append(ToTensor())
        if original_data_aug is not None:
            l_aug.append(original_data_aug)
        if dequant is not None:
            l_aug.append(dequant)
        data_aug = Compose(l_aug)
        dataset = CelebA_OOD(data_path, split=split_type, transform=data_aug)
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == "FashionMNIST":  # normal FashionMNIS
        dataset = FashionMNIST_OOD(
            data_path, split=split_type, download=True, transform=ToTensor()
        )
        dataset.img_size = (28, 28)
    elif name == "MNIST":  # normal  MNIST
        dataset = MNIST_OOD(
            data_path, split=split_type, download=True, transform=ToTensor()
        )
        dataset.img_size = (28, 28)
    elif name == "NotMNIST":
        dataset = NotMNIST(data_path, split=split_type, transform=ToTensor())
        dataset.img_size = (28, 28)
    elif name == "Omniglot":
        size = data_dict.get("size", OOD_SIZE)
        invert = data_dict.get("invert", True)  # invert pixel intensity: x -> 1 - x
        if split_type == "training":
            background = True
        else:
            background = False

        if invert:
            tr = Compose([Resize(size), ToTensor(), Invert()])
        else:
            tr = Compose([Resize(size), ToTensor()])

        dataset = Omniglot(
            data_path, background=background, download=False, transform=tr
        )
    elif name == "ImageNet32":
        train_split_ratio = data_dict.get("train_split_ratio", 0.8)
        seed = data_dict.get("seed", 1)
        dataset = ImageNet32(
            data_path,
            split=split_type,
            transform=ToTensor(),
            seed=seed,
            train_split_ratio=train_split_ratio,
        )
    elif name == "RImgNet":
        from loader.imagenet_augmentation import get_imageNet_augmentation
        from loader.imagenet_subsets import RestrictedImageNet

        augm_type = "none"
        size = 224
        path = os.path.join(data_dict["path"], "ImageNet2012")
        transform = get_imageNet_augmentation(
            type=augm_type, out_size=size, config_dict=None
        )
        split = "train" if split_type == "training" else "val"
        balanced = False
        dataset = RestrictedImageNet(
            path, split=split, transform=transform, balanced=balanced
        )

    elif name == "Flowers":
        from loader.flowers import Flowers
        from loader.imagenet_augmentation import get_imageNet_augmentation

        augm_type = "none"
        size = 224
        path = os.path.join(data_dict["path"], "flowers")
        if split_type == "training":
            split = "train"
        elif split_type == "validation":
            split = "val"
        elif split_type == "evaluation":
            # split = "test"
            split = "test_wo_insects"
        transform = get_imageNet_augmentation(
            type=augm_type, out_size=size, config_dict=None
        )
        dataset = Flowers(path, split, transform=transform)
    elif name == "Cars":
        from loader.imagenet_augmentation import get_imageNet_augmentation
        from loader.stanford_cars import StanfordCars

        augm_type = "none"
        size = 224
        transform = get_imageNet_augmentation(
            type=augm_type, out_size=size, config_dict=None
        )
        path = os.path.join(data_dict["path"], "stanford_cars")
        train = True if split_type == "training" else False
        dataset = StanfordCars(path, train, transform=transform)
    elif name == "FGVC":
        from loader.fgvc import FGVCAircraft
        from loader.imagenet_augmentation import get_imageNet_augmentation

        augm_type = "none"
        size = 224
        transform = get_imageNet_augmentation(
            type=augm_type, out_size=size, config_dict=None
        )
        path = os.path.join(data_dict["path"], "FGVC", "fgvc-aircraft-2013b")
        if split_type == "training":
            split = "train"
        elif split_type == "validation":
            split = "val"
        elif split_type == "evaluation":
            split = "test"
        dataset = FGVCAircraft(
            path, class_type="variant", split=split, transform=transform
        )
    elif name == "dtd":
        size = data_dict.get("size", 224)
        path = os.path.join(data_dict["path"], "dtd", "dtd", "images")
        dataset = ImageFolder(
            path, transform=Compose([Resize(size), CenterCrop(size), ToTensor()])
        )

    elif name == "OpenImages-O":
        from loader.openimages import OpenImages_O

        size = data_dict.get("size", 224)
        path = os.path.join(data_dict["path"], "openimagev3")
        transform = Compose([Resize(size), CenterCrop(size), ToTensor()])
        dataset = OpenImages_O(
            root=path,
            image_list_file="openimage_o.txt",
            transform=transform,
            split="evaluation",
        )

    elif name == "EuroSAT":
        size = data_dict.get("size", 224)
        path = os.path.join(data_dict["path"], "EuroSAT", "2750")
        transform = Compose([Resize(size), CenterCrop(size), ToTensor()])
        dataset = ImageFolder(path, transform=transform)
    else:
        n_classes = data_dict["n_classes"]
        split = data_dict["split"][split_type]

        param_dict = copy.deepcopy(data_dict)
        param_dict.pop("dataset")
        param_dict.pop("path")
        param_dict.pop("n_classes")
        param_dict.pop("split")
        param_dict.update({"split_type": split_type})

        dataset_instance = _get_dataset_instance(name)
        dataset = dataset_instance(
            data_path,
            n_classes,
            split=split,
            augmentations=data_aug,
            is_transform=True,
            **param_dict,
        )

    return dataset


def _get_dataset_instance(name):
    """get_loader

    :param name:
    """
    return {
        "basic": basic_dataset,
        "inmemory": InMemoryDataset,
    }[name]


def np_to_loader(l_tensors, batch_size, num_workers, load_all=False, shuffle=False):
    """Convert a list of numpy arrays to a torch.DataLoader"""
    if load_all:
        dataset = data.TensorDataset(*[torch.Tensor(X).cuda() for X in l_tensors])
        num_workers = 0
        return data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )
    else:
        dataset = data.TensorDataset(*[torch.Tensor(X) for X in l_tensors])
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=False,
        )
