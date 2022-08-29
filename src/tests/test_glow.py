import pytest
import torch
from torch.optim import Adam

from models.glow.models import GlowV2


@pytest.mark.parametrize("coupling", ["affine", "affineV2"])
def test_glowv2_vector_mode(coupling):
    N = 5
    D = 20
    X = torch.randn(N, D, 1, 1, dtype=torch.float32)

    hparam = {
        "image_shape": [1, 1, 20],
        "hidden_channels": 64,
        "K": 32,
        "L": 3,
        "actnorm_scale": 1.0,
        "flow_permutation": "invconv",
        "flow_coupling": coupling,
        "LU_decomposed": False,
        "learn_top": False,
        "y_condition": False,
        "y_classes": 40,
        "prior_type": "gaussian",
        "vector_mode": True,
    }

    model = GlowV2(**hparam)

    d_forward = model(x=X)
    Z = d_forward["z"]
    nll = d_forward["nll"]
    l_zs = d_forward["l_zs"]
    logdet = d_forward["logdet"]
    prior_log_p = d_forward["prior_log_p"]
    assert Z.shape == (N, D, 1, 1)
    assert nll.shape == (N,)
    model.eval()

    # actnorm init
    model(torch.randn(1, D, 1, 1, dtype=torch.float), reverse=False)

    # check for invertibility
    d_reverse = model(z=Z, reverse=True, l_zs=l_zs)
    new_x = d_reverse["x"]
    new_logdet = d_reverse["logdet"]
    lik_reverse = d_reverse["logp"]
    diff = (new_x - X).abs().mean()
    assert diff < 1e-3
    assert (logdet - (-new_logdet)).abs().mean() < 1e-3
    assert (-nll - lik_reverse).abs().mean() < 1e-3

    # sampling
    d_sample = model.sample(N, device="cpu")
    sample_x = d_sample["x"]
    assert sample_x.shape == (N, D, 1, 1)
    assert not torch.isnan(sample_x).any()

    # training step
    opt = Adam(model.parameters(), lr=0.001)
    d_loss = model.train_step(X, optimizer=opt)
    assert "loss" in d_loss


@pytest.mark.parametrize("coupling", ["affine", "affineV2"])
def test_glowv2_img(coupling):
    N = 5
    L = 2
    W = 32  # image shape
    X = torch.rand(N, 3, W, W, dtype=torch.float32)

    hparam = {
        "image_shape": [W, W, 3],
        "hidden_channels": 64,
        "K": 32,
        "L": L,
        "actnorm_scale": 1.0,
        "flow_permutation": "shuffle",
        "flow_coupling": coupling,
        "LU_decomposed": False,
        "learn_top": False,
        "y_condition": False,
        "y_classes": 40,
        "prior_type": "gaussian",
        "vector_mode": False,
        "dequant_offset": False,
    }

    model = GlowV2(**hparam)

    d_forward = model(x=X)
    Z = d_forward["z"]
    nll = d_forward["nll"]
    l_zs = d_forward["l_zs"]
    logdet = d_forward["logdet"]
    prior_log_p = d_forward["prior_log_p"]
    assert Z.shape == (N, 3 * (2**L) * 2, W // (2**L), W // (2**L))
    assert nll.shape == (N,)
    assert len(l_zs) == (L - 1)
    model.eval()

    # check for invertibility
    d_reverse = model(z=Z, reverse=True, l_zs=l_zs)
    new_x = d_reverse["x"]
    new_logdet = d_reverse["logdet"]
    diff = (new_x - X).abs().mean()
    assert diff < 1e-3
    assert (logdet - (-new_logdet)).abs().mean() < 1e-2

    # sampling
    d_sample = model.sample(N, device="cpu")
    sample_x = d_sample["x"]
    assert sample_x.shape == (N, 3, W, W)
    assert "logdet" in d_sample
    assert "l_zs" in d_sample
    assert "l_logpz" in d_sample
    assert len(d_sample["l_zs"]) == len(d_sample["l_logpz"])
    assert not torch.isnan(sample_x).any()

    # training step
    opt = Adam(model.parameters(), lr=0.001)
    d_loss = model.train_step(X, optimizer=opt)
    assert "loss" in d_loss
