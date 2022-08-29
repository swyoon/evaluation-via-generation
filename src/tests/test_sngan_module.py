import pytest
import torch

from models.modules import IGEBMEncoder
from models.modules_sngan import GeneratorGN


def test_groupnorm_generator():
    z_dim = 4
    channel = 3
    hidden_dim = 12
    num_groups = 3
    x = torch.rand(2, z_dim, 1, 1)
    m = GeneratorGN(
        z_dim,
        channel,
        hidden_dim=hidden_dim,
        out_activation="sigmoid",
        num_groups=num_groups,
    )
    out = m(x)


@pytest.mark.parametrize("z_spatial_dim", [1, 2, 4])
def test_grid_z(z_spatial_dim):
    z_dim = 4
    channel = 3
    hidden_dim = 12
    num_groups = 3
    x = torch.rand(2, 3, 32, 32)
    m1 = IGEBMEncoder(avg_pool_dim=z_spatial_dim, out_chan=z_dim)
    m2 = GeneratorGN(
        z_dim,
        channel,
        hidden_dim=hidden_dim,
        out_activation="sigmoid",
        num_groups=num_groups,
        spatial_dim=z_spatial_dim,
    )
    xhat = m2(m1(x))
    assert xhat.shape == x.shape
