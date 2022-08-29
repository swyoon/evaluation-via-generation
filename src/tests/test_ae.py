import numpy as np
import pytest
import torch
from torch.optim import Adam

from models import get_ae
from models.ae import AE, DAE, VAE
from models.glow.models import GlowV2
from models.modules import (
    ConvDenoiser,
    ConvMLP,
    ConvNet1,
    ConvNet2,
    ConvNet2FC,
    ConvNet3,
    DeConvNet1,
    DeConvNet2,
    DeConvNet3,
    FCNet,
)


def test_ae():
    # generate dummy data
    X = np.random.randn(2, 3, 128, 128)
    X = torch.tensor(X, dtype=torch.float32)

    # model
    encoder = ConvNet1(in_chan=3, out_chan=12)
    decoder = DeConvNet1(in_chan=12, out_chan=3)
    ae = AE(encoder, decoder)

    X_recon = ae.forward(X)
    assert X.shape == X_recon.shape

    optimizer = Adam(ae.parameters(), 1e-3)
    d_loss = ae.train_step(X, optimizer)
    assert "loss" in d_loss
    assert isinstance(d_loss["loss"], float)


def test_conv3():
    net1 = ConvNet3(in_chan=3, out_chan=16)
    net2 = DeConvNet3(in_chan=16, out_chan=3)
    x = torch.randn(1, 3, 32, 32)
    recon = net2(net1(x))
    assert x.shape == recon.shape


def test_vae():
    z_dim = 3
    cfg = {
        "model": {
            "arch": "vae",
            "encoder": {"arch": "conv2", "nh": 16, "out_activation": "tanh"},
            "decoder": {"arch": "deconv2", "nh": 16},
            "x_dim": 3,
            "z_dim": z_dim,
        }
    }

    ae = get_ae(**cfg["model"])
    assert isinstance(ae, VAE)

    X = np.random.randn(2, 3, 28, 28)
    X = torch.tensor(X, dtype=torch.float32)

    z = ae.encode(X)
    assert z.shape[1] == z_dim * 2

    X_recon = ae.forward(X)
    assert X.shape == X_recon.shape

    pred = ae.predict(X)
    assert len(pred) == len(X)

    optimizer = Adam(ae.parameters(), 1e-3)
    d_loss = ae.train_step(X, optimizer)
    assert "loss" in d_loss
    assert "vae/kl_loss_" in d_loss

    ll = ae.marginal_likelihood(X)
    assert len(ll) == len(X)


def test_new_ae():
    cfg = {
        "model": {
            "arch": "ae",
            "encoder": {"arch": "conv2", "nh": 16, "out_activation": "tanh"},
            "decoder": {"arch": "deconv2", "nh": 16},
            "x_dim": 1,
            "z_dim": 10,
        }
    }
    z_dim = cfg["model"]["z_dim"]

    ae = get_ae(**cfg["model"])
    assert isinstance(ae, AE)

    X = np.random.randn(2, 1, 28, 28)
    X = torch.tensor(X, dtype=torch.float32)

    z = ae.encode(X)
    assert z.shape[1] == z_dim

    X_recon = ae.forward(X)
    assert X.shape == X_recon.shape

    pred = ae.predict(X)
    assert len(pred) == len(X)

    optimizer = Adam(ae.parameters(), 1e-3)
    d_loss = ae.train_step(X, optimizer)
    assert "loss" in d_loss
