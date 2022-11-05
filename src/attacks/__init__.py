import copy
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from attacks.advdist import (
    AdversarialDistribution,
    AdversarialDistributionAE,
    AdversarialDistributionLinf,
    AdversarialDistributionStyleGAN2,
    AdversarialDistributionTransform,
    AdversarialDistributionVQVAE,
)
from attacks.mcmc import (
    CoordinateDescentSampler,
    LangevinSampler,
    MHSampler,
    RandomSampler,
)
from augmentations import get_composed_augmentations
from models import get_model, load_pretrained

from .detector import Detector, EnsembleDetector


def get_sampler(**sampler_cfg):
    sampler_type = sampler_cfg.pop("name")
    if sampler_type == "mh":
        sampler = MHSampler(**sampler_cfg)
    elif sampler_type == "random":
        sampler = RandomSampler(**sampler_cfg)
    elif sampler_type == "coord":
        sampler = CoordinateDescentSampler(**sampler_cfg)
    elif sampler_type == "langevin":
        sampler = LangevinSampler(**sampler_cfg)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
    return sampler


def get_detector(device="cpu", normalize=False, root="./", **cfg_detector):
    d_detector_aug = cfg_detector.pop("detector_aug", None)
    no_grad_predict = cfg_detector.pop("no_grad_predict", True)
    blackbox_only = cfg_detector.pop("blackbox_only", False)
    indist_dataset = cfg_detector.pop("indist_dataset", "CIFAR10")
    alias = cfg_detector.pop("alias", None)
    detector, _ = load_pretrained(
        **cfg_detector["detector"], device=device, root=os.path.join(root, "pretrained")
    )

    aug = get_composed_augmentations(d_detector_aug)
    detector = Detector(
        detector,
        bound=-1,
        transform=aug,
        no_grad_predict=no_grad_predict,
        blackbox_only=blackbox_only,
        use_rank=False,
    )
    detector.to(device)

    if normalize:
        print("Normalizing detector score...")
        normalization_path = os.path.join(
            root, "results", indist_dataset, alias, f"IN_score.pkl"
        )
        detector.load_normalization(normalization_path, device=device)
    return detector


def get_ensemble(device="cpu", **cfg_detector):
    # cfg_detector : contains ensemble1, ensemble2, ...
    l_detector = []
    l_no_grad = []
    for key, cfg_ in cfg_detector.items():
        if key.startswith("ensemble"):
            l_detector.append(get_detector(**cfg_, device=device))
            l_no_grad.append(cfg_.get("detector_no_grad", False))
    no_grad = all(l_no_grad)
    print(no_grad)
    detector = EnsembleDetector(
        l_detector,
        bound=-1,
        use_rank=False,
        no_grad=no_grad,
        agg=cfg_detector.get("agg", "mean"),
    )
    detector.to(device)
    # detector.learn_normalization(dataloader=train_dl, device=device).detach().cpu().numpy()
    return detector


def get_advdist(cfg):
    cfg = copy.deepcopy(cfg)
    cfg_advdist = cfg["advdist"]
    cfg_model = cfg_advdist.pop("model", None)
    if cfg_model in {"mh"}:
        model = cfg_model
    elif cfg_model is None:
        model = None
    else:
        model = get_model(cfg_model)

    name = cfg_advdist.pop("name")

    if "sampler" in cfg_advdist:
        cfg_sampler = cfg_advdist.pop("sampler")
        sampler = get_sampler(**cfg_sampler)

    if name == "advq":
        cfg_vqvae = cfg_advdist.pop("vqvae")
        vqvae, _ = load_pretrained(**cfg_vqvae)
        advdist = AdversarialDistributionVQVAE(
            model, vqvae=vqvae, classifier=clsf, **cfg_advdist
        )
    elif name == "adae":
        cfg_ae = cfg_advdist.pop("ae")
        ae, _ = load_pretrained(**cfg_ae)
        # '''initialize actnorm parameter by forwarding arbitrary data'''
        # model(x=torch.randn(50,cfg_model.pop('x_dim'),1,1, dtype=torch.float), reverse=False)
        advdist = AdversarialDistributionAE(
            model, ae=ae, classifier=clsf, **cfg_advdist
        )
    elif name == "adtr":
        advdist = AdversarialDistributionTransform(sampler=sampler, **cfg_advdist)
    elif name == "adlinf":
        advdist = AdversarialDistributionLinf(sampler=sampler, **cfg_advdist)
    elif name == "adstylegan2":
        cfg_stylegan2_gen = cfg_advdist.pop("stylegan2_g")
        g, _ = load_pretrained(**cfg_stylegan2_gen)
        g.to(cfg["device"])
        g.eval()
        advdist = AdversarialDistributionStyleGAN2(
            generator=g, sampler=sampler, **cfg_advdist
        )
    else:
        advdist = AdversarialDistribution(model, **cfg_advdist)
    return advdist
