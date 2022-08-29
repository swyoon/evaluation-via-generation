import sys

import pytest
import torch
from omegaconf import OmegaConf
from torch.optim import Adam

from attacks import Detector, EnsembleDetector, get_advdist, get_ensemble
from augmentations import get_composed_augmentations
from loader import get_dataloader
from models import load_pretrained


def test_ensemble():

    dataloader_cfg = {
        "dataset": "CIFAR10_OOD",
        "path": "datasets",
        "batch_size": 16,
        "n_workers": 4,
        "split": "validation",
    }

    d_dataloaders = {}
    d_dataloaders["cifar10"] = get_dataloader(dataloader_cfg)

    cfg_path = "configs_attack/cifar_detectors/cifar_a_c_g.yml"
    cfg = OmegaConf.load(cfg_path)
    device = "cpu"

    cfg_detector = cfg["detector"]
    do_ensemble = any([k.startswith("ensemble") for k in cfg_detector.keys()])

    detector = get_ensemble(**cfg_detector, device="cpu")
    # l_detector = []
    # l_no_grad = []
    # for key, cfg_detector in cfg['detector'].items():
    #     model, _ = load_pretrained(**cfg_detector, device=device)

    #     if 'detector_aug' in cfg_detector:
    #         aug = get_composed_augmentations(cfg_detector['detector_aug'])
    #     else:
    #         aug = None
    #     no_grad = cfg_detector.get('detector_no_grad', False)
    #     l_no_grad.append(no_grad)
    #     model = Detector(model, bound=-1, transform=aug, no_grad=no_grad, use_rank=False)
    #     model.to(device)
    #     l_detector.append(model)
    # detector = EnsembleDetector(l_detector, bound=-1, use_rank=False)
    # detector.to(device)
    # if False in l_no_grad: no_grad = False

    x, y = next(iter(d_dataloaders["cifar10"]))
    pred = detector.predict(x)
    assert len(pred) == len(x)
