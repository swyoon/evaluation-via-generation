import pytest
import torch
from torch.optim import Adam

from models.glow_y0ast.models import Glow_y0ast


@pytest.mark.parametrize("prior_type", ["gaussian", "uniform"])
def test_glow_y0ast(prior_type):
    N = 5
    D = 20
    X = torch.randn(N, D, 2, 2, dtype=torch.float32)

    hparam = {
        "image_shape": [2, 2, 20],
        "hidden_channels": 64,
        "K": 32,
        "L": 1,
        "actnorm_scale": 1.0,
        "flow_permutation": "invconv",
        "flow_coupling": "affine",
        "LU_decomposed": False,
        "y_classes": 40,
        "learn_top": False,
        "y_condition": False,
    }

    model = Glow_y0ast(**hparam)

    Z, nll, y_logit = model(X)
    assert Z.shape == (N, 20 * 2 * 2, 2 // 2, 2 // 2)
    assert nll.shape == (N,)

    opt = Adam(model.parameters(), lr=0.001)

    d_loss = model.train_step(X, optimizer=opt)
    assert "loss" in d_loss
