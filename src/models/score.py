"""
score.py
=====
Score matching
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# from models.modules import (ConvNet1, DeConvNet1, DCGANEncoder, DCGANDecoder, DCGANEncoder2, DCGANDecoder2,
#                                      ConvNet2, DeConvNet2, ConvNet3, DeConvNet3,
#                                      ConvClassifier, DCGANDecoder_Resnet,
#                                      ResNetEncoder18, ResNetEncoder34, ResNetEncoder50,
#                                      ResNetEncoder101, ResNetEncoder152, FCNet, ConvMLP,
#                                      VQVAEDecoder, VQVAEEncoder)
from models.discon_models import EM_AE_V1

# from models.unet import unet
from models.vqvae.modules import VQEmbedding
from optimizers import _get_optimizer_instance, get_optimizer


class DSM(nn.Module):
    """denoising score matching. predicts data score"""

    def __init__(self, net, sig):
        super(DSM, self).__init__()
        self.net = net
        self.sig = sig

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        """return norm of score"""
        return (self(x).view(len(x), -1) ** 2).mean(dim=1)

    def train_step(self, x, opt):
        opt.zero_grad()
        noise = torch.randn_like(x) * self.sig
        perturbed_x = x + noise

        target = -1 / (self.sig**2) * noise
        score = self(perturbed_x)
        target = target.view(len(target), -1)
        score = score.view(len(score), -1)

        loss = torch.mean((score - target) ** 2)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    def validation_step(self, x):
        noise = torch.randn_like(x) * self.sig
        score = self(x + noise)
        target = -1 / (self.sig**2) * noise
        score = score.view(len(score), -1)
        target = target.view(len(target), -1)
        loss = torch.mean((target - score) ** 2)
        return {"loss": loss.item()}


class MultiDSM(nn.Module):
    """denoising score matching. predicts data score"""

    def __init__(self, net, l_sig):
        super(MultiDSM, self).__init__()
        self.net = net
        self.l_sig = torch.tensor(l_sig)

    def _append_sig(self, x, sig):
        shape = list(x.shape)
        shape[1] = 1

        if not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig, dtype=torch.float).to(x.device)

        if len(sig.shape) == 0:
            sig = sig[None]
        sig = sig.reshape(len(sig), *([1] * len(shape[1:])))

        sig_input = (torch.ones(shape, dtype=torch.float32).to(x.device) * sig).to(
            x.device
        )
        x_sig = torch.cat([x, sig_input], dim=1)
        return x_sig

    def forward(self, x, sig):
        x_sig = self._append_sig(x, sig)
        return self.net(x_sig)

    def predict(self, x, sig):
        """return norm of score"""
        return (self(x, sig).view(len(x), -1) ** 2).mean(dim=1)

    def train_step(self, x, opt):
        opt.zero_grad()

        idx_sig = torch.multinomial(
            torch.ones_like(self.l_sig), len(x), replacement=True
        ).to(x.device)
        sig_1d = self.l_sig[idx_sig][:, None].to(x.device)
        sig = sig_1d.reshape(len(sig_1d), *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x) * sig
        perturbed_x = x + noise
        score = self(perturbed_x, sig_1d)
        target = -1 / (sig**2) * noise
        target = target.view(len(target), -1)
        score = score.view(len(score), -1)

        loss = torch.mean(((score - target) ** 2).mean(dim=-1) * sig_1d**2)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    def validation_step(self, x):
        idx_sig = torch.multinomial(
            torch.ones_like(self.l_sig), len(x), replacement=True
        ).to(x.device)
        sig_1d = self.l_sig[idx_sig][:, None].to(x.device)
        sig = sig_1d.reshape(len(sig_1d), *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x) * sig
        perturbed_x = x + noise
        score = self(perturbed_x, sig_1d)

        target = -1 / (sig**2) * noise
        score = self(perturbed_x)
        target = target.view(len(target), -1)
        score = score.view(len(score), -1)

        loss = torch.mean(((score - target) ** 2).mean(dim=-1) * sig_1d**2)
        loss.backward()
        return {"loss": loss.item()}
