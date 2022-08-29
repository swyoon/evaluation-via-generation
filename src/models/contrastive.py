"""
contrastive.py
==============
noise contrastive approach
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive(nn.Module):
    def __init__(self, net, sigma=0.1, uniform_jitter=None):
        super(Contrastive, self).__init__()
        self.net = net
        self.sigma = sigma  # std of noise
        self.own_optimizer = False
        self.uniform_jitter = uniform_jitter

    def forward(self, x):
        return self.net(x)

    def corrupt_forward(self, x, sigma=None):
        if sigma is None:
            sigma = self.sigma
        noise = sigma * torch.randn_like(x)

        if self.uniform_jitter:
            X = torch.cat(
                [
                    x * 255 / 256 + torch.rand_like(x) / 256,
                    x * 255 / 256 + torch.rand_like(x) / 256.0 + noise,
                ]
            )
        else:
            X = torch.cat([x, x + noise])
        y = torch.cat(
            [torch.ones(len(x), 1).to(x.device), torch.zeros(len(x), 1).to(x.device)]
        )
        pred = self.net(X)
        return X, y, pred

    def train_step(self, x, opt):
        opt.zero_grad()
        X, y, pred = self.corrupt_forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    def validation_step(self, x, reconstruction=None):
        X, y, pred = self.corrupt_forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        real_data_pred = pred[: len(x)]
        return {"loss": loss.item(), "predict": self.compute_lap(real_data_pred)}

    def predict(self, x, sigma=None):
        """compute \nabla^2 p / p"""
        pred = self(x)
        return self.compute_lap(pred)

    def compute_lap(self, pred, sigma=None):
        if sigma is None:
            sigma = self.sigma
        return -2 * pred / (sigma**2)


class ContrastiveV2(nn.Module):
    """noise is applied to original data as well."""

    def __init__(self, net, sigma_1=0.01, sigma_2=0.1):
        super(ContrastiveV2, self).__init__()
        self.net = net
        self.sigma_1 = sigma_1  # std of noise to data
        self.sigma_2 = sigma_2  # std of noise to contrastive
        self.own_optimizer = False

    def forward(self, x):
        return self.net(x)

    def corrupt_forward(self, x, sigma_1=None, sigma_2=None):
        if sigma_1 is None:
            sigma_1 = self.sigma_1
        if sigma_2 is None:
            sigma_2 = self.sigma_2

        if sigma_1 == 0:
            x_1 = x
        else:
            x_1 = x + sigma_1 * torch.randn_like(x)

        x_2 = x + sigma_2 * torch.randn_like(x)
        X = torch.cat([x_1, x_2])
        y = torch.cat(
            [torch.ones(len(x), 1).to(x.device), torch.zeros(len(x), 1).to(x.device)]
        )
        pred = self.net(X)
        return X, y, pred

    def train_step(self, x, opt):
        opt.zero_grad()
        X, y, pred = self.corrupt_forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    def validation_step(self, x):
        X, y, pred = self.corrupt_forward(x, sigma_1=0)  # use true data for inference
        loss = F.binary_cross_entropy_with_logits(pred, y)
        real_data_pred = pred[: len(x)]
        return {"loss": loss.item(), "predict": self.compute_lap(real_data_pred)}

    def predict(self, x, sigma=None):
        """compute \nabla^2 p / p"""
        pred = self(x)
        return self.compute_lap(pred)

    def compute_lap(self, pred, sigma_2=None):
        if sigma_2 is None:
            sigma_2 = self.sigma_2
        return -2 * pred / (sigma_2**2)


class ContrastiveMulti(nn.Module):
    def __init__(self, net, l_sigma=(0.1, 0.3, 0.5), sigma_0=0.01):
        super(ContrastiveMulti, self).__init__()
        self.net = net
        self.l_sigma = l_sigma  # std of noise
        self.sigma_0 = sigma_0
        self.own_optimizer = False

    def forward(self, x, arr_sigma=None, sigma=None):
        if arr_sigma is None:
            if sigma is None:
                sigma = self.l_sigma[0]
            arr_sigma = torch.ones(len(x)).to(x.device) * sigma

        # make appendable sigma
        sh = [len(x)] + [1 for i in range(len(x.shape) - 1)]
        arr_sigma = arr_sigma.view(sh).repeat([1, 1] + list(x.shape[2:]))

        x_ = torch.cat([x, arr_sigma], dim=1)
        return self.net(x_)

    def corrupt_forward(self, x, sigma=None, sigma_0=None):
        if sigma is None:
            arr_sigma = np.random.choice(self.l_sigma, size=(len(x),))
            arr_sigma = torch.tensor(arr_sigma, dtype=torch.float32).to(x.device)
        else:
            arr_sigma = torch.ones(len(x)).to(x.device) * sigma

        if sigma_0 is None:
            sigma_0 = self.sigma_0

        if sigma_0 == 0:
            x_0 = x
        else:
            x_0 = x + torch.randn_like(x) * sigma_0

        noise = torch.randn_like(x) * arr_sigma.view((-1, 1))

        X = torch.cat([x_0, x + noise])
        y = torch.cat(
            [torch.ones(len(x), 1).to(x.device), torch.zeros(len(x), 1).to(x.device)]
        )
        pred = self(X, torch.cat([arr_sigma, arr_sigma]))
        return X, y, pred

    def train_step(self, x, opt):
        opt.zero_grad()
        X, y, pred = self.corrupt_forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    def validation_step(self, x):
        X, y, pred = self.corrupt_forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        real_data_pred = self(x)
        return {"loss": loss.item(), "predict": self.compute_lap(real_data_pred)}

    def predict(self, x, sigma=None):
        """compute \nabla^2 p / p"""
        pred = self(x, sigma=sigma)
        return self.compute_lap(pred, sigma=sigma)

    def compute_lap(self, pred, sigma=None):
        if sigma is None:
            sigma = self.l_sigma[0]
        return -2 * pred / (sigma**2)
