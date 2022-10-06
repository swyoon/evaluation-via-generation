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
from attacks.mcmc import CoordinateDescentSampler, MHSampler, RandomSampler
from augmentations import get_composed_augmentations
from models import get_model, load_pretrained
from models.test_time_aug import TestTimeAug


def get_sampler(**sampler_cfg):
    sampler_type = sampler_cfg.pop("name")
    if sampler_type == "mh":
        sampler = MHSampler(**sampler_cfg)
    elif sampler_type == "random":
        sampler = RandomSampler(**sampler_cfg)
    elif sampler_type == "coord":
        sampler = CoordinateDescentSampler(**sampler_cfg)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
    return sampler


def get_detector(device="cpu", normalize=False, **cfg_detector):
    d_detector_aug = cfg_detector.pop("detector_aug", None)
    no_grad = cfg_detector.pop("detector_no_grad", True)
    indist_dataset = cfg_detector.pop("indist_dataset", "CIFAR10")
    alias = cfg_detector.pop("alias", None)
    detector, _ = load_pretrained(**cfg_detector["detector"], device=device)

    aug = get_composed_augmentations(d_detector_aug)
    detector = Detector(
        detector, bound=-1, transform=aug, no_grad=no_grad, use_rank=False
    )
    detector.to(device)

    if normalize:
        print("Normalizing detector score...")
        normalization_path = os.path.join(
            "results", indist_dataset, alias, f"IN_score.pkl"
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

    if "classifier" in cfg_advdist:
        cfg_clsf = cfg_advdist.pop("classifier")
        if "testtimeaug" in cfg_clsf:
            testtimeaug = cfg_clsf.pop("testtimeaug")
        else:
            testtimeaug = False
        clsf, _ = load_pretrained(**cfg_clsf)
        if testtimeaug:
            clsf = TestTimeAug(clsf)
    else:
        clsf = None

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
        advdist = AdversarialDistributionStyleGAN2(
            generator=g, sampler=sampler, **cfg_advdist
        )
    else:
        advdist = AdversarialDistribution(model, **cfg_advdist)
    return advdist


class Detector(nn.Module):
    """A wrapper class for OOD detector.
    Main functions are:
        1. input pre-processing
        2. OOD score normalization"""

    def __init__(
        self,
        model,
        transform=None,
        mean=0,
        std=1,
        bound=-1,
        no_grad=False,
        up_bound=-1,
        use_rank=False,
    ):
        """
        no_grad: `torch.no_grad()` when predict
        """
        super().__init__()
        self.model = model
        self.transform = transform
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("bound", torch.tensor(bound))
        self.no_grad = no_grad
        self.up_bound = up_bound
        self.use_rank = use_rank

    def predict(self, x, normalize=True, binary_reward=None):
        if self.no_grad:
            f = torch.no_grad()(self._predict)
        else:
            f = self._predict
        return f(x, normalize=normalize, binary_reward=binary_reward)

    def _predict(self, x, normalize=True, binary_reward=None):
        if self.transform is not None:
            x = self.transform(x)
        score = self.model.predict(x)
        if normalize:
            score = self.normalize(score)
        if binary_reward is not None:
            index = score > binary_reward
            score[index] = 1
            score[~index] = -1
        return score

    def normalize(self, score):
        if self.use_rank:
            return torch.bucketize(score, self.in_score) / len(self.in_score) * 2 - 1

        else:
            normed = (score - self.mean) / self.std
            if self.bound > 0:
                normed.clip_(-self.bound, self.bound)
            if self.up_bound > 0:
                normed.clip_(max=self.up_bound)
            return normed

    # def learn_normalization(self, dataloader=None, device=None, use_grad=False):
    #     """compute normalization parameters for detector score"""
    #     l_score = []
    #     for xx, _ in tqdm(dataloader):
    #         if device is not None:
    #             xx = xx.to(device)
    #         if use_grad:
    #             l_score.append(self.predict(xx, normalize=False).detach())
    #         else:
    #             try:
    #                 with torch.no_grad():
    #                     l_score.append(self.predict(xx, normalize=False).detach())
    #             except:
    #                 l_score.append(self.predict(xx, normalize=False).detach())
    #     score = torch.cat(l_score)
    #     if self.use_rank:
    #         self.in_score = torch.sort(score).values
    #     else:
    #         mean_score = torch.mean(score)
    #         std_score = torch.std(score) + 1e-3
    #         self.mean = mean_score
    #         self.std = std_score

    #     normed_score = self.normalize(score)
    #     return normed_score

    # def save_normalization(self, norm_path):
    #     if self.use_rank:
    #         torch.save(self.in_score, norm_path)
    #         print("save rank normalization info")
    #     else:
    #         torch.save([self.mean, self.std], norm_path)
    #         print("save std normalization info")

    # def load_normalization(self, norm_path, device):
    #     if self.use_rank:
    #         self.in_score = torch.load(norm_path, map_location=device)
    #         print("load rank normalization info")
    #     else:
    #         self.mean, self.std = torch.load(norm_path, map_location=device)
    #         print("load std normalization info")

    def load_normalization(self, norm_path, device):
        in_score = torch.load(norm_path)
        mean_score = torch.mean(in_score)
        std_score = torch.std(in_score) + 1e-3
        self.mean = mean_score.to(device)
        self.std = std_score.to(device)


class EnsembleDetector(nn.Module):
    """A wrapper class for Ensemble model of OOD detectors.
    Main functions are:
        1. input pre-processing
        2. OOD score normalization"""

    def __init__(
        self,
        l_model,
        mean=0,
        std=1,
        bound=-1,
        up_bound=-1,
        use_rank=False,
        agg="max",
        no_grad=False,
    ):
        """
        no_grad: `torch.no_grad()` when predict
        agg: aggregation method
        """
        super().__init__()
        self.l_model = l_model
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("bound", torch.tensor(bound))
        self.up_bound = up_bound
        self.use_rank = use_rank
        self.agg = agg
        self.no_grad = no_grad

    def predict(self, x, normalize=True, binary_reward=None):
        l_predict = []
        for model in self.l_model:
            if model.no_grad:
                f = torch.no_grad()(model._predict)
            else:
                f = model._predict
            l_predict.append(f(x, normalize=True, binary_reward=binary_reward))
        pred = torch.stack(l_predict)
        pred = self.aggregate(pred)
        if normalize:
            return self.normalize(pred)
        else:
            return pred

    def aggregate(self, pred):
        """pred: (n_model) x (n_example)"""
        if self.agg == "mean":
            return pred.mean(dim=0)
        elif self.agg == "max":
            return pred.max(dim=0).values
        else:
            raise ValueError(f"Invalid aggregation {self.agg}")

    def normalize(self, score):
        if self.use_rank:
            return torch.bucketize(score, self.in_score) / len(self.in_score) * 2 - 1

        else:
            normed = (score - self.mean) / self.std
            if self.bound > 0:
                normed.clip_(-self.bound, self.bound)
            if self.up_bound > 0:
                normed.clip_(max=self.up_bound)
            return normed

    def learn_normalization(self, dataloader=None, device=None, use_grad=False):
        """compute normalization parameters for detector score"""
        l_score = []
        for xx, _ in tqdm(dataloader):
            if device is not None:
                xx = xx.to(device)
            l_score.append(self.predict(xx, normalize=False).detach())

        score = torch.cat(l_score)
        if self.use_rank:
            self.in_score = torch.sort(score).values
        else:
            mean_score = torch.mean(score)
            std_score = torch.std(score) + 1e-3
            self.mean = mean_score
            self.std = std_score

        normed_score = self.normalize(score)
        return score
